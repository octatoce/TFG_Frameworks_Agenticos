"""CrewAI implementation for ARCH_05_HANDOFF_SWARM."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

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
from benchmark_core.resource_monitor import ResourceMonitor
from benchmark_core.schemas import AgentStep, ExperimentConfig, ExperimentInput
from benchmark_core.tracing import utc_now
from implementations.crewai.utils_crewai import (
    CrewAIRunContext,
    CrewAIRunOutput,
    create_agent,
    create_sequential_crew,
    create_task,
    crewai_architecture_runner,
)


AGENT_DEFINITIONS = {
    "data_specialist": {
        "role": "DataSpecialist",
        "goal": "Extract evidence and decide the next specialist.",
        "backstory": "A decentralized swarm specialist with no central supervisor.",
    },
    "reasoning_specialist": {
        "role": "ReasoningSpecialist",
        "goal": "Reason over context and decide whether to hand off or finalize.",
        "backstory": "A decentralized swarm specialist with no central supervisor.",
    },
    "validation_specialist": {
        "role": "ValidationSpecialist",
        "goal": "Validate consistency and transfer control when needed.",
        "backstory": "A decentralized swarm specialist with no central supervisor.",
    },
    "synthesis_specialist": {
        "role": "SynthesisSpecialist",
        "goal": "Synthesize and finalize when enough context is available.",
        "backstory": "A decentralized swarm specialist with no central supervisor.",
    },
}


class HandoffSwarmFlowState(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    initial_agent: str = ""
    active_agent: str = ""
    active_agent_history: list[str] = Field(default_factory=list)
    handoff_history: list[dict[str, Any]] = Field(default_factory=list)
    partial_results: list[dict[str, Any]] = Field(default_factory=list)
    repeated_agent_visits: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    number_of_handoffs: int = 0
    number_of_agent_invocations: int = 0
    cycle_detected: bool = False
    fallback_used: bool = False
    finalizing_agent: str | None = None
    stop_reason: str | None = None
    final_answer: str = ""
    last_decision: dict[str, Any] = Field(default_factory=dict)
    context_transferred: str = ""


HandoffSwarmFlowState.model_rebuild(_types_namespace={"Any": Any})


class HandoffSwarmFlow:
    """CrewAI Flow factory namespace used to keep imports after runtime setup."""

    @staticmethod
    def build(
        *,
        input_data: ExperimentInput,
        config: ExperimentConfig,
        context: CrewAIRunContext,
        agents: dict[str, Any],
        limits: dict[str, int],
        steps: list[AgentStep],
    ):
        from crewai.flow import Flow, listen, router, start

        class CrewAIHandoffFlow(Flow[HandoffSwarmFlowState]):
            input_data: ExperimentInput
            config: ExperimentConfig
            agents: dict[str, Any]
            limits: dict[str, int]
            steps: list[AgentStep]

            @start()
            def start_swarm(self) -> str:
                return self.state.active_agent

            @router("start_swarm")
            def route_from_start(self, next_route: str) -> str:
                return next_route or "done"

            @listen("data_specialist")
            def run_data_specialist(self) -> str:
                return self._run_active_specialist("data_specialist")

            @router("run_data_specialist")
            def route_from_data(self, next_route: str) -> str:
                return next_route or "done"

            @listen("reasoning_specialist")
            def run_reasoning_specialist(self) -> str:
                return self._run_active_specialist("reasoning_specialist")

            @router("run_reasoning_specialist")
            def route_from_reasoning(self, next_route: str) -> str:
                return next_route or "done"

            @listen("validation_specialist")
            def run_validation_specialist(self) -> str:
                return self._run_active_specialist("validation_specialist")

            @router("run_validation_specialist")
            def route_from_validation(self, next_route: str) -> str:
                return next_route or "done"

            @listen("synthesis_specialist")
            def run_synthesis_specialist(self) -> str:
                return self._run_active_specialist("synthesis_specialist")

            @router("run_synthesis_specialist")
            def route_from_synthesis(self, next_route: str) -> str:
                return next_route or "done"

            @listen("done")
            def finish_swarm(self) -> str:
                return self.state.final_answer

            def _run_active_specialist(self, active_agent: str) -> str:
                state = self.state
                if state.stop_reason is not None:
                    return "done"

                if state.number_of_agent_invocations >= self.limits["max_agent_invocations"]:
                    state.stop_reason = "max_agent_invocations_reached"
                    state.finalizing_agent = active_agent
                    state.final_answer = build_fallback_answer(self.input_data, state.partial_results)
                    return "done"

                state.repeated_agent_visits[active_agent] = state.repeated_agent_visits.get(active_agent, 0) + 1
                if state.repeated_agent_visits[active_agent] > self.limits["max_consecutive_visits_per_agent"]:
                    state.cycle_detected = True
                    state.fallback_used = True
                    state.stop_reason = "max_consecutive_visits_per_agent_reached"
                    state.finalizing_agent = active_agent
                    state.final_answer = build_fallback_answer(self.input_data, state.partial_results)
                    state.warnings.append(f"Visit limit reached for {active_agent}.")
                    return "done"

                prompt_state = {
                    "active_agent_history": state.active_agent_history,
                    "handoff_history": state.handoff_history,
                    "partial_results": state.partial_results,
                    "context_transferred": state.context_transferred,
                    "number_of_handoffs": state.number_of_handoffs,
                    "number_of_agent_invocations": state.number_of_agent_invocations,
                    "warnings": state.warnings,
                }
                prompt = render_handoff_swarm_prompt(self.input_data, active_agent, prompt_state)
                task = create_task(
                    description=prompt,
                    expected_output="A parseable ARCH_05 HandoffDecision.",
                    agent=self.agents[active_agent],
                    config=self.config,
                )
                crew = create_sequential_crew(agents=[self.agents[active_agent]], tasks=[task])
                crew.kickoff()
                call_record = context.crewai_llm.call_records[-1]
                decision = parse_handoff_decision(call_record.response.strip())
                state.last_decision = decision
                state.number_of_agent_invocations += 1
                state.active_agent_history.append(active_agent)
                state.partial_results.append({"agent": active_agent, "decision": decision})
                step_started_at = utc_now()
                self.steps.append(
                    AgentStep(
                        step_id=len(self.steps) + 1,
                        name=active_agent,
                        step_type="handoff_agent_llm_call",
                        actor=f"crewai.flow.{AGENT_DEFINITIONS[active_agent]['role']}",
                        input_data={"prompt": prompt, "active_agent": active_agent},
                        output_data={"decision": decision},
                        llm_call_ids=[call_record.metrics.call_id],
                        started_at=step_started_at,
                        finished_at=utc_now(),
                        metadata={
                            "architecture": "ARCH_05_HANDOFF_SWARM",
                            "native_primitive": "CrewAI Flow listener/router plus dynamic Agent/Task",
                            "crew_process": "flow_routed_single_agent_tasks",
                        },
                    )
                )

                if not is_valid_handoff_decision(decision):
                    state.fallback_used = True
                    state.warnings.append(f"Invalid handoff decision from {active_agent}.")
                    if active_agent != "synthesis_specialist" and state.number_of_handoffs < self.limits["max_handoffs"]:
                        target_agent = "synthesis_specialist"
                        state.handoff_history.append(
                            {
                                "sequence_number": state.number_of_handoffs + 1,
                                "source_agent": active_agent,
                                "target_agent": target_agent,
                                "reason": "Fallback after invalid decision.",
                                "task": "Finalize from available context.",
                                "context_summary": "Fallback context after invalid decision.",
                                "timestamp": utc_now().isoformat(),
                            }
                        )
                        state.number_of_handoffs += 1
                        state.context_transferred = "Fallback context after invalid decision."
                        state.active_agent = target_agent
                        return target_agent
                    state.stop_reason = "invalid_decision_fallback_finalized"
                    state.finalizing_agent = active_agent
                    state.final_answer = build_fallback_answer(self.input_data, state.partial_results)
                    return "done"

                if decision["action"] == "finalize":
                    state.stop_reason = "agent_finalized"
                    state.finalizing_agent = active_agent
                    state.final_answer = extract_final_answer(str(decision["final_output"]))
                    return "done"

                target_agent = str(decision["target_agent"])
                state.cycle_detected = state.cycle_detected or (
                    len(state.active_agent_history) >= 2 and target_agent == state.active_agent_history[-2]
                )
                if state.number_of_handoffs >= self.limits["max_handoffs"]:
                    state.stop_reason = "max_handoffs_reached"
                    state.finalizing_agent = active_agent
                    state.final_answer = build_fallback_answer(self.input_data, state.partial_results)
                    return "done"

                state.handoff_history.append(
                    {
                        "sequence_number": state.number_of_handoffs + 1,
                        "source_agent": active_agent,
                        "target_agent": target_agent,
                        "reason": decision["reason"],
                        "task": decision["task"],
                        "context_summary": decision["context_summary"],
                        "timestamp": utc_now().isoformat(),
                    }
                )
                state.number_of_handoffs += 1
                state.context_transferred = str(decision["context_summary"])
                state.active_agent = target_agent
                return target_agent

        CrewAIHandoffFlow.model_rebuild(
            _types_namespace={
                "Any": Any,
                "ExperimentInput": ExperimentInput,
                "ExperimentConfig": ExperimentConfig,
                "AgentStep": AgentStep,
            }
        )
        initial_agent = choose_initial_handoff_agent(input_data)
        return CrewAIHandoffFlow(
            initial_state=HandoffSwarmFlowState(
                initial_agent=initial_agent,
                active_agent=initial_agent,
            ),
            input_data=input_data,
            config=config,
            agents=agents,
            limits=limits,
            steps=steps,
            tracing=False,
            stream=False,
            suppress_flow_events=True,
            max_method_calls=limits["max_agent_invocations"] + limits["max_handoffs"] + 8,
        )


@crewai_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: CrewAIRunContext,
) -> CrewAIRunOutput:
    """Execute decentralized handoffs with CrewAI Flow routing."""

    agents = {
        agent_name: create_agent(
            **definition,
            crewai_llm=context.crewai_llm,
            config=config,
        )
        for agent_name, definition in AGENT_DEFINITIONS.items()
    }
    limits = get_handoff_limits(config)
    steps: list[AgentStep] = []

    with ResourceMonitor() as monitor:
        flow = HandoffSwarmFlow.build(
            input_data=input_data,
            config=config,
            context=context,
            agents=agents,
            limits=limits,
            steps=steps,
        )
        flow.kickoff()
        resource_usage = monitor.usage
    state = flow.state
    flow_steps = list(flow.steps)

    final_answer = state.final_answer or build_fallback_answer(input_data, state.partial_results)
    structured_output = {
        "answer": final_answer,
        "decision": state.last_decision,
        "confidence": state.last_decision.get("confidence", 0.0),
        "evidence": state.last_decision.get("evidence", "none"),
        "limitations": state.last_decision.get("limitations", "none"),
        "initial_agent": state.initial_agent,
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
        "framework_native_primitives": ["CrewAI Flow", "CrewAI @start", "CrewAI @listen", "CrewAI @router", "CrewAI Agent", "CrewAI Task"],
        "native_automatic_behaviors": [],
        "parallelism_used": False,
        "warnings": state.warnings,
        "document_ids": document_ids(input_data),
        "framework_execution": "crewai_handoff_swarm_flow",
    }
    return CrewAIRunOutput(
        final_answer=final_answer,
        structured_output=structured_output,
        steps=flow_steps,
        resource_usage=resource_usage,
    )
