"""Pydantic AI implementation for ARCH_05_HANDOFF_SWARM."""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic_graph import BaseNode, End, Graph, GraphRunContext

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


def _node_for_agent(agent_name: str) -> SwarmDataNode | SwarmReasoningNode | SwarmValidationNode | SwarmSynthesisNode:
    if agent_name == "data_specialist":
        return SwarmDataNode()
    if agent_name == "reasoning_specialist":
        return SwarmReasoningNode()
    if agent_name == "validation_specialist":
        return SwarmValidationNode()
    return SwarmSynthesisNode()


async def _run_specialist_node(
    ctx: GraphRunContext[GraphSwarmState, GraphDeps],
    agent_name: str,
) -> SwarmDataNode | SwarmReasoningNode | SwarmValidationNode | SwarmSynthesisNode | End[dict[str, Any]]:
    state = ctx.state
    deps = ctx.deps
    input_data = deps.input_data

    if state.number_of_agent_invocations >= deps.limits["max_agent_invocations"]:
        state.stop_reason = "max_agent_invocations_reached"
        state.finalizing_agent = agent_name
        state.final_answer = build_fallback_answer(input_data, state.partial_results)
        return End({"final_answer": state.final_answer})

    state.repeated_agent_visits[agent_name] = state.repeated_agent_visits.get(agent_name, 0) + 1
    if state.repeated_agent_visits[agent_name] > deps.limits["max_consecutive_visits_per_agent"]:
        state.cycle_detected = True
        state.fallback_used = True
        state.stop_reason = "max_consecutive_visits_per_agent_reached"
        state.finalizing_agent = agent_name
        state.final_answer = build_fallback_answer(input_data, state.partial_results)
        state.warnings.append(f"Visit limit reached for {agent_name}.")
        return End({"final_answer": state.final_answer})

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
                "native_primitive": "pydantic_graph.BaseNode",
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
            return _node_for_agent(target_agent)
        state.stop_reason = "invalid_decision_fallback_finalized"
        state.finalizing_agent = agent_name
        state.final_answer = build_fallback_answer(input_data, state.partial_results)
        return End({"final_answer": state.final_answer})

    if decision["action"] == "finalize":
        state.stop_reason = "agent_finalized"
        state.finalizing_agent = agent_name
        state.final_answer = extract_final_answer(str(decision["final_output"]))
        return End({"final_answer": state.final_answer})

    target_agent = str(decision["target_agent"])
    state.cycle_detected = state.cycle_detected or (
        len(state.active_agent_history) >= 2 and target_agent == state.active_agent_history[-2]
    )
    if state.number_of_handoffs >= deps.limits["max_handoffs"]:
        state.stop_reason = "max_handoffs_reached"
        state.finalizing_agent = agent_name
        state.final_answer = build_fallback_answer(input_data, state.partial_results)
        return End({"final_answer": state.final_answer})

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
    return _node_for_agent(target_agent)


class SwarmDataNode(BaseNode[GraphSwarmState, GraphDeps, dict[str, Any]]):
    async def run(
        self,
        ctx: GraphRunContext[GraphSwarmState, GraphDeps],
    ) -> SwarmReasoningNode | SwarmSynthesisNode | End[dict[str, Any]]:
        next_node = await _run_specialist_node(ctx, "data_specialist")
        if isinstance(next_node, (SwarmReasoningNode, SwarmSynthesisNode, End)):
            return next_node
        return SwarmSynthesisNode()


class SwarmReasoningNode(BaseNode[GraphSwarmState, GraphDeps, dict[str, Any]]):
    async def run(
        self,
        ctx: GraphRunContext[GraphSwarmState, GraphDeps],
    ) -> SwarmDataNode | SwarmValidationNode | SwarmSynthesisNode | End[dict[str, Any]]:
        next_node = await _run_specialist_node(ctx, "reasoning_specialist")
        if isinstance(next_node, (SwarmDataNode, SwarmValidationNode, SwarmSynthesisNode, End)):
            return next_node
        return SwarmSynthesisNode()


class SwarmValidationNode(BaseNode[GraphSwarmState, GraphDeps, dict[str, Any]]):
    async def run(
        self,
        ctx: GraphRunContext[GraphSwarmState, GraphDeps],
    ) -> SwarmReasoningNode | SwarmSynthesisNode | End[dict[str, Any]]:
        next_node = await _run_specialist_node(ctx, "validation_specialist")
        if isinstance(next_node, (SwarmReasoningNode, SwarmSynthesisNode, End)):
            return next_node
        return SwarmSynthesisNode()


class SwarmSynthesisNode(BaseNode[GraphSwarmState, GraphDeps, dict[str, Any]]):
    async def run(
        self,
        ctx: GraphRunContext[GraphSwarmState, GraphDeps],
    ) -> SwarmDataNode | SwarmReasoningNode | End[dict[str, Any]]:
        next_node = await _run_specialist_node(ctx, "synthesis_specialist")
        if isinstance(next_node, (SwarmDataNode, SwarmReasoningNode, End)):
            return next_node
        return End({"final_answer": ctx.state.final_answer or build_fallback_answer(ctx.deps.input_data, ctx.state.partial_results)})


def _run_graph(graph, start_node, *, state: GraphSwarmState, deps: GraphDeps):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(graph.run(start_node, state=state, deps=deps))
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
        graph = Graph(
            nodes=[SwarmDataNode, SwarmReasoningNode, SwarmValidationNode, SwarmSynthesisNode],
            name="ARCH_05_HANDOFF_SWARM",
            state_type=GraphSwarmState,
            run_end_type=dict,
        )
        _run_graph(graph, _node_for_agent(initial_agent), state=state, deps=deps)

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
            "framework_native_primitives": ["pydantic_graph.Graph", "BaseNode", "End"],
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
