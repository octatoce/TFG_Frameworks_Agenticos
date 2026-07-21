"""Pydantic AI implementation for ARCH_05_HANDOFF_SWARM."""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel
from pydantic_graph import GraphBuilder, StepContext

from benchmark_core.handoff_swarm import (
    HANDOFF_AGENTS,
    build_fallback_answer,
    choose_initial_handoff_agent,
    document_ids,
    extract_final_answer,
    get_handoff_limits,
    is_valid_handoff_decision,
    parse_handoff_decision,
    render_handoff_swarm_prompt,
)
from benchmark_core.schemas import AgentStep, ExperimentConfig, ExperimentInput
from benchmark_core.tracing import utc_now
from implementations.pydantic_ai.utils_pydantic_ai import (
    HandoffDecision,
    PydanticAIHandoffSwarmOutput,
    PydanticAIRunContext,
    PydanticAIRunOutput,
    build_typed_agent,
    complete_agent_step,
    framework_execution,
    pydantic_ai_architecture_runner,
    run_with_resource_monitor,
)


AGENT_INSTRUCTIONS = {
    "data_specialist": "Typed graph node DataSpecialist. Return HandoffDecision only.",
    "reasoning_specialist": "Typed graph node ReasoningSpecialist. Return HandoffDecision only.",
    "validation_specialist": "Typed graph node ValidationSpecialist. Return HandoffDecision only.",
    "synthesis_specialist": "Typed graph node SynthesisSpecialist. Return HandoffDecision only.",
}


class GraphSwarmState:
    def __init__(self) -> None:
        self.active_agent_history: list[str] = []
        self.handoff_history: list[dict[str, Any]] = []
        self.partial_results: list[dict[str, Any]] = []
        self.context_transferred = ""
        self.number_of_handoffs = 0
        self.number_of_agent_invocations = 0
        self.repeated_agent_visits: dict[str, int] = {}
        self.cycle_detected = False
        self.fallback_used = False
        self.finalizing_agent: str | None = None
        self.stop_reason: str | None = None
        self.warnings: list[str] = []
        self.last_decision: dict[str, Any] = {}
        self.final_answer = ""


class SwarmTransition(BaseModel):
    next_agent: str | None


class GraphDeps:
    def __init__(
        self,
        *,
        input_data: ExperimentInput,
        config: ExperimentConfig,
        context: PydanticAIRunContext,
        agents: dict[str, Any],
        limits: dict[str, int],
    ) -> None:
        self.input_data = input_data
        self.config = config
        self.context = context
        self.agents = agents
        self.limits = limits
        self.steps: list[AgentStep] = []
        self.llm_calls: list[Any] = []


async def _run_specialist_step(
    ctx: StepContext[GraphSwarmState, GraphDeps, SwarmTransition],
    agent_name: str,
) -> SwarmTransition:
    state = ctx.state
    deps = ctx.deps
    input_data = deps.input_data

    if state.number_of_agent_invocations >= deps.limits["max_agent_invocations"]:
        state.stop_reason = "max_agent_invocations_reached"
        state.finalizing_agent = agent_name
        state.final_answer = build_fallback_answer(input_data, state.partial_results)
        return SwarmTransition(next_agent=None)

    state.repeated_agent_visits[agent_name] = state.repeated_agent_visits.get(agent_name, 0) + 1
    if state.repeated_agent_visits[agent_name] > deps.limits["max_consecutive_visits_per_agent"]:
        state.cycle_detected = True
        state.fallback_used = True
        state.stop_reason = "max_consecutive_visits_per_agent_reached"
        state.finalizing_agent = agent_name
        state.final_answer = build_fallback_answer(input_data, state.partial_results)
        state.warnings.append(f"Visit limit reached for {agent_name}.")
        return SwarmTransition(next_agent=None)

    prompt_state = {
        "active_agent_history": state.active_agent_history,
        "handoff_history": state.handoff_history,
        "partial_results": state.partial_results,
        "context_transferred": state.context_transferred,
        "number_of_handoffs": state.number_of_handoffs,
        "number_of_agent_invocations": state.number_of_agent_invocations,
        "warnings": state.warnings,
    }
    prompt = render_handoff_swarm_prompt(input_data, agent_name, prompt_state)
    step_started_at = utc_now()
    step_id = len(deps.steps) + 1
    call_record = complete_agent_step(
        agent=deps.agents[agent_name],
        prompt=prompt,
        input_data=input_data,
        config=deps.config,
        step_id=step_id,
    )
    decision = HandoffDecision.model_validate(parse_handoff_decision(call_record.response.strip())).model_dump()
    state.last_decision = decision
    state.number_of_agent_invocations += 1
    state.active_agent_history.append(agent_name)
    state.partial_results.append({"agent": agent_name, "decision": decision})
    deps.llm_calls.append(call_record.metrics)
    deps.steps.append(
        AgentStep(
            step_id=step_id,
            name=agent_name,
            step_type="handoff_agent_llm_call",
            actor=f"pydantic_ai.graph.{agent_name}",
            input_data={"prompt": prompt, "active_agent": agent_name},
            output_data={"decision": decision},
            llm_call_ids=[call_record.metrics.call_id],
            started_at=step_started_at,
            finished_at=utc_now(),
            metadata={
                "architecture": "ARCH_05_HANDOFF_SWARM",
                "native_primitive": "pydantic_graph.GraphBuilder.Step",
                "native_graph_available": deps.context.native_graph_available,
            },
        )
    )

    if not is_valid_handoff_decision(decision):
        state.fallback_used = True
        state.warnings.append(f"Invalid handoff decision from {agent_name}.")
        if agent_name != "synthesis_specialist" and state.number_of_handoffs < deps.limits["max_handoffs"]:
            target_agent = "synthesis_specialist"
            state.handoff_history.append(
                {
                    "sequence_number": state.number_of_handoffs + 1,
                    "source_agent": agent_name,
                    "target_agent": target_agent,
                    "reason": "Fallback after invalid decision.",
                    "task": "Finalize from available context.",
                    "context_summary": "Fallback context after invalid decision.",
                    "timestamp": utc_now().isoformat(),
                }
            )
            state.number_of_handoffs += 1
            state.context_transferred = "Fallback context after invalid decision."
            return SwarmTransition(next_agent=target_agent)
        state.stop_reason = "invalid_decision_fallback_finalized"
        state.finalizing_agent = agent_name
        state.final_answer = build_fallback_answer(input_data, state.partial_results)
        return SwarmTransition(next_agent=None)

    if decision["action"] == "finalize":
        state.stop_reason = "agent_finalized"
        state.finalizing_agent = agent_name
        state.final_answer = extract_final_answer(str(decision["final_output"]))
        return SwarmTransition(next_agent=None)

    target_agent = str(decision["target_agent"])
    state.cycle_detected = state.cycle_detected or (
        len(state.active_agent_history) >= 2 and target_agent == state.active_agent_history[-2]
    )
    if state.number_of_handoffs >= deps.limits["max_handoffs"]:
        state.stop_reason = "max_handoffs_reached"
        state.finalizing_agent = agent_name
        state.final_answer = build_fallback_answer(input_data, state.partial_results)
        return SwarmTransition(next_agent=None)

    state.handoff_history.append(
        {
            "sequence_number": state.number_of_handoffs + 1,
            "source_agent": agent_name,
            "target_agent": target_agent,
            "reason": decision["reason"],
            "task": decision["task"],
            "context_summary": decision["context_summary"],
            "timestamp": utc_now().isoformat(),
        }
    )
    state.number_of_handoffs += 1
    state.context_transferred = str(decision["context_summary"])
    return SwarmTransition(next_agent=target_agent)


def _run_graph(
    graph,
    graph_input: SwarmTransition,
    *,
    state: GraphSwarmState,
    deps: GraphDeps,
    timeout_seconds: int,
):
    async def execute():
        return await asyncio.wait_for(
            graph.run(inputs=graph_input, state=state, deps=deps),
            timeout=float(timeout_seconds),
        )

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(execute())
    raise RuntimeError("Pydantic graph runner requires a synchronous entrypoint.")


@pydantic_ai_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: PydanticAIRunContext,
) -> PydanticAIRunOutput:
    """Execute a real pydantic-graph handoff swarm."""

    def execute() -> PydanticAIRunOutput:
        agents = {
            agent_name: build_typed_agent(
                name=agent_name,
                instructions=instructions,
                context=context,
                input_data=input_data,
                config=config,
            )
            for agent_name, instructions in AGENT_INSTRUCTIONS.items()
        }
        limits = get_handoff_limits(config)
        initial_agent = choose_initial_handoff_agent(input_data)
        state = GraphSwarmState()
        deps = GraphDeps(
            input_data=input_data,
            config=config,
            context=context,
            agents=agents,
            limits=limits,
        )
        builder = GraphBuilder(
            name="ARCH_05_HANDOFF_SWARM",
            state_type=GraphSwarmState,
            deps_type=GraphDeps,
            input_type=SwarmTransition,
            output_type=dict,
            auto_instrument=False,
        )

        async def data_specialist(
            ctx: StepContext[GraphSwarmState, GraphDeps, SwarmTransition],
        ) -> SwarmTransition:
            return await _run_specialist_step(ctx, "data_specialist")

        async def reasoning_specialist(
            ctx: StepContext[GraphSwarmState, GraphDeps, SwarmTransition],
        ) -> SwarmTransition:
            return await _run_specialist_step(ctx, "reasoning_specialist")

        async def validation_specialist(
            ctx: StepContext[GraphSwarmState, GraphDeps, SwarmTransition],
        ) -> SwarmTransition:
            return await _run_specialist_step(ctx, "validation_specialist")

        async def synthesis_specialist(
            ctx: StepContext[GraphSwarmState, GraphDeps, SwarmTransition],
        ) -> SwarmTransition:
            return await _run_specialist_step(ctx, "synthesis_specialist")

        async def finish_swarm(
            ctx: StepContext[GraphSwarmState, GraphDeps, SwarmTransition],
        ) -> dict[str, Any]:
            if not ctx.state.final_answer:
                ctx.state.final_answer = build_fallback_answer(
                    ctx.deps.input_data,
                    ctx.state.partial_results,
                )
            return {"final_answer": ctx.state.final_answer}

        specialist_steps = {
            "data_specialist": builder.step(data_specialist, node_id="data_specialist"),
            "reasoning_specialist": builder.step(reasoning_specialist, node_id="reasoning_specialist"),
            "validation_specialist": builder.step(validation_specialist, node_id="validation_specialist"),
            "synthesis_specialist": builder.step(synthesis_specialist, node_id="synthesis_specialist"),
        }
        finish_step = builder.step(finish_swarm, node_id="finish_swarm")
        handoff_decision = builder.decision(node_id="handoff_decision")
        for agent_name in HANDOFF_AGENTS:
            handoff_decision = handoff_decision.branch(
                builder.match(
                    SwarmTransition,
                    matches=lambda transition, target=agent_name: transition.next_agent == target,
                ).to(specialist_steps[agent_name])
            )
        handoff_decision = handoff_decision.branch(
            builder.match(
                SwarmTransition,
                matches=lambda transition: transition.next_agent is None,
            ).to(finish_step)
        )

        builder.add(builder.edge_from(builder.start_node).to(specialist_steps[initial_agent]))
        builder.add(builder.edge_from(*specialist_steps.values()).to(handoff_decision))
        builder.add(builder.edge_from(finish_step).to(builder.end_node))
        graph = builder.build()
        _run_graph(
            graph,
            SwarmTransition(next_agent=initial_agent),
            state=state,
            deps=deps,
            timeout_seconds=config.timeout_seconds,
        )

        final_answer = state.final_answer or build_fallback_answer(input_data, state.partial_results)
        structured_output = {
            "answer": final_answer,
            "mode": f"{config.model_provider}_handoff_swarm",
            "decision": state.last_decision,
            "confidence": state.last_decision.get("confidence", 0.0),
            "evidence": state.last_decision.get("evidence", "none"),
            "limitations": state.last_decision.get("limitations", "none"),
            "initial_agent": initial_agent,
            "active_agent_history": state.active_agent_history,
            "handoff_history": state.handoff_history,
            "number_of_handoffs": state.number_of_handoffs,
            "max_handoffs": limits["max_handoffs"],
            "number_of_agent_invocations": state.number_of_agent_invocations,
            "max_agent_invocations": limits["max_agent_invocations"],
            "unique_agents_executed": sorted(set(state.active_agent_history), key=HANDOFF_AGENTS.index),
            "finalizing_agent": state.finalizing_agent,
            "repeated_agent_visits": state.repeated_agent_visits,
            "cycle_detected": state.cycle_detected,
            "fallback_used": state.fallback_used,
            "stop_reason": state.stop_reason,
            "framework_native_primitives": [
                "pydantic_graph.GraphBuilder",
                "Step",
                "Decision",
            ],
            "native_automatic_behaviors": [],
            "parallelism_used": False,
            "warnings": state.warnings,
            "document_ids": document_ids(input_data),
            "framework_execution": framework_execution("handoff_swarm_graph", context),
        }
        structured_model = PydanticAIHandoffSwarmOutput.model_validate(structured_output)
        return PydanticAIRunOutput(
            final_answer=final_answer,
            structured_output=structured_model.model_dump(),
            steps=deps.steps,
            llm_calls=deps.llm_calls,
        )

    return run_with_resource_monitor(execute)
