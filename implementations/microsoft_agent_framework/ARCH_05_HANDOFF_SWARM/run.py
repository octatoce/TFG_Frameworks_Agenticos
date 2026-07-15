"""Microsoft Agent Framework implementation for ARCH_05_HANDOFF_SWARM."""

from __future__ import annotations

from typing import Any

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
from implementations.microsoft_agent_framework.utils_microsoft_agent_framework import (
    MicrosoftAgentFrameworkRunContext,
    MicrosoftAgentFrameworkRunOutput,
    framework_execution,
    run_async,
    microsoft_agent_framework_architecture_runner,
)


AGENT_INSTRUCTIONS = {
    "data_specialist": "You are DataSpecialist. Extract evidence, then hand off if reasoning is needed.",
    "reasoning_specialist": "You are ReasoningSpecialist. Reason over context and hand off only when useful.",
    "validation_specialist": "You are ValidationSpecialist. Validate consistency and hand off for synthesis.",
    "synthesis_specialist": "You are SynthesisSpecialist. Finalize when enough context is available.",
}


HANDOFF_TARGETS = {
    "data_specialist": ["reasoning_specialist", "synthesis_specialist"],
    "reasoning_specialist": ["validation_specialist", "data_specialist", "synthesis_specialist"],
    "validation_specialist": ["reasoning_specialist", "synthesis_specialist"],
    "synthesis_specialist": ["data_specialist", "reasoning_specialist"],
}


class NativeHandoffState:
    def __init__(self, *, initial_agent: str, limits: dict[str, int]) -> None:
        self.initial_agent = initial_agent
        self.active_agent = initial_agent
        self.active_agent_history: list[str] = []
        self.handoff_history: list[dict[str, Any]] = []
        self.partial_results: list[dict[str, Any]] = []
        self.repeated_agent_visits: dict[str, int] = {}
        self.warnings: list[str] = []
        self.steps: list[AgentStep] = []
        self.llm_calls: list[Any] = []
        self.number_of_handoffs = 0
        self.number_of_agent_invocations = 0
        self.cycle_detected = False
        self.fallback_used = False
        self.finalizing_agent: str | None = None
        self.stop_reason: str | None = None
        self.final_answer = ""
        self.last_decision: dict[str, Any] = {}
        self.context_transferred = ""
        self.limits = limits
        self.native_handoff_events: list[dict[str, Any]] = []


class BenchmarkHandoffChatClient:
    """Function-calling chat client used by Microsoft HandoffBuilder."""

    def __init__(
        self,
        *,
        agent_name: str,
        input_data: ExperimentInput,
        config: ExperimentConfig,
        context: MicrosoftAgentFrameworkRunContext,
        state: NativeHandoffState,
    ) -> None:
        from agent_framework import BaseChatClient, FunctionInvocationLayer

        class _Client(FunctionInvocationLayer, BaseChatClient):
            def _inner_get_response(inner_self, *, messages, stream=False, options=None, **kwargs):
                async def respond():
                    return self._respond()

                return respond()

        self.agent_name = agent_name
        self.input_data = input_data
        self.config = config
        self.context = context
        self.state = state
        self.client = _Client()

    def _respond(self):
        from agent_framework import ChatResponse, Content, Message

        agent_name = self.agent_name
        state = self.state
        if state.stop_reason is not None:
            text = f"Final Answer: {state.final_answer or build_fallback_answer(self.input_data, state.partial_results)}"
            message = Message(role="assistant", author_name=agent_name, contents=[Content.from_text(text)])
            return ChatResponse(messages=[message], response_id=f"{self.config.run_id}-{agent_name}-terminal")

        if state.number_of_agent_invocations >= state.limits["max_agent_invocations"]:
            state.stop_reason = "max_agent_invocations_reached"
            state.finalizing_agent = agent_name
            state.final_answer = build_fallback_answer(self.input_data, state.partial_results)
            message = Message(role="assistant", author_name=agent_name, contents=[Content.from_text(f"Final Answer: {state.final_answer}")])
            return ChatResponse(messages=[message], response_id=f"{self.config.run_id}-{agent_name}-max-invocations")

        state.repeated_agent_visits[agent_name] = state.repeated_agent_visits.get(agent_name, 0) + 1
        if state.repeated_agent_visits[agent_name] > state.limits["max_consecutive_visits_per_agent"]:
            state.cycle_detected = True
            state.fallback_used = True
            state.stop_reason = "max_consecutive_visits_per_agent_reached"
            state.finalizing_agent = agent_name
            state.final_answer = build_fallback_answer(self.input_data, state.partial_results)
            state.warnings.append(f"Visit limit reached for {agent_name}.")
            message = Message(role="assistant", author_name=agent_name, contents=[Content.from_text(f"Final Answer: {state.final_answer}")])
            return ChatResponse(messages=[message], response_id=f"{self.config.run_id}-{agent_name}-visit-limit")

        prompt_state = {
            "active_agent_history": state.active_agent_history,
            "handoff_history": state.handoff_history,
            "partial_results": state.partial_results,
            "context_transferred": state.context_transferred,
            "number_of_handoffs": state.number_of_handoffs,
            "number_of_agent_invocations": state.number_of_agent_invocations,
            "warnings": state.warnings,
        }
        prompt = render_handoff_swarm_prompt(self.input_data, agent_name, prompt_state)
        step_started_at = utc_now()
        step_id = len(state.steps) + 1
        call_record = self.context.llm.complete(
            prompt=prompt,
            input_data=self.input_data,
            call_id=f"{self.config.run_id}-llm-{step_id:03d}",
            step_id=step_id,
        )
        decision = parse_handoff_decision(call_record.response.strip())
        state.last_decision = decision
        state.number_of_agent_invocations += 1
        state.active_agent_history.append(agent_name)
        state.partial_results.append({"agent": agent_name, "decision": decision})
        state.llm_calls.append(call_record.metrics)
        state.steps.append(
            AgentStep(
                step_id=step_id,
                name=agent_name,
                step_type="handoff_agent_llm_call",
                actor=f"microsoft_agent_framework.handoff_builder.{agent_name}",
                input_data={"prompt": prompt, "active_agent": agent_name},
                output_data={"decision": decision},
                llm_call_ids=[call_record.metrics.call_id],
                started_at=step_started_at,
                finished_at=utc_now(),
                metadata={
                    "architecture": "ARCH_05_HANDOFF_SWARM",
                    "native_primitive": "HandoffBuilder handoff_to tool",
                },
            )
        )

        contents = [Content.from_text(call_record.response)]
        if not is_valid_handoff_decision(decision):
            state.fallback_used = True
            state.warnings.append(f"Invalid handoff decision from {agent_name}.")
            if agent_name != "synthesis_specialist" and state.number_of_handoffs < state.limits["max_handoffs"]:
                target_agent = "synthesis_specialist"
                self._record_handoff(agent_name, target_agent, "Fallback after invalid decision.", "Finalize from available context.", "Fallback context after invalid decision.")
                contents.append(Content.from_function_call(f"{call_record.metrics.call_id}-handoff", f"handoff_to_{target_agent}"))
            else:
                state.stop_reason = "invalid_decision_fallback_finalized"
                state.finalizing_agent = agent_name
                state.final_answer = build_fallback_answer(self.input_data, state.partial_results)
                contents = [Content.from_text(f"Final Answer: {state.final_answer}")]
            message = Message(role="assistant", author_name=agent_name, contents=contents)
            return ChatResponse(messages=[message], response_id=call_record.metrics.call_id, finish_reason="stop")

        if decision["action"] == "finalize":
            state.stop_reason = "agent_finalized"
            state.finalizing_agent = agent_name
            state.final_answer = extract_final_answer(str(decision["final_output"]))
            message = Message(role="assistant", author_name=agent_name, contents=[Content.from_text(call_record.response)])
            return ChatResponse(messages=[message], response_id=call_record.metrics.call_id, finish_reason="stop")

        target_agent = str(decision["target_agent"])
        state.cycle_detected = state.cycle_detected or (
            len(state.active_agent_history) >= 2 and target_agent == state.active_agent_history[-2]
        )
        if state.number_of_handoffs >= state.limits["max_handoffs"]:
            state.stop_reason = "max_handoffs_reached"
            state.finalizing_agent = agent_name
            state.final_answer = build_fallback_answer(self.input_data, state.partial_results)
            message = Message(role="assistant", author_name=agent_name, contents=[Content.from_text(f"Final Answer: {state.final_answer}")])
            return ChatResponse(messages=[message], response_id=call_record.metrics.call_id, finish_reason="stop")

        self._record_handoff(
            agent_name,
            target_agent,
            str(decision["reason"]),
            str(decision["task"]),
            str(decision["context_summary"]),
        )
        contents.append(Content.from_function_call(f"{call_record.metrics.call_id}-handoff", f"handoff_to_{target_agent}"))
        message = Message(role="assistant", author_name=agent_name, contents=contents)
        return ChatResponse(messages=[message], response_id=call_record.metrics.call_id, finish_reason="stop")

    def _record_handoff(
        self,
        source_agent: str,
        target_agent: str,
        reason: str,
        task: str,
        context_summary: str,
    ) -> None:
        state = self.state
        state.handoff_history.append(
            {
                "sequence_number": state.number_of_handoffs + 1,
                "source_agent": source_agent,
                "target_agent": target_agent,
                "reason": reason,
                "task": task,
                "context_summary": context_summary,
                "timestamp": utc_now().isoformat(),
            }
        )
        state.number_of_handoffs += 1
        state.context_transferred = context_summary
        state.active_agent = target_agent


def _build_native_agent(
    *,
    agent_name: str,
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: MicrosoftAgentFrameworkRunContext,
    state: NativeHandoffState,
):
    from agent_framework import Agent

    chat_client = BenchmarkHandoffChatClient(
        agent_name=agent_name,
        input_data=input_data,
        config=config,
        context=context,
        state=state,
    )
    return Agent(
        client=chat_client.client,
        id=agent_name,
        name=agent_name,
        description=AGENT_INSTRUCTIONS[agent_name],
        instructions=AGENT_INSTRUCTIONS[agent_name],
        require_per_service_call_history_persistence=True,
    )


@microsoft_agent_framework_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: MicrosoftAgentFrameworkRunContext,
) -> MicrosoftAgentFrameworkRunOutput:
    """Execute decentralized handoffs with Microsoft HandoffBuilder."""

    def execute() -> MicrosoftAgentFrameworkRunOutput:
        from agent_framework.orchestrations import HandoffBuilder

        limits = get_handoff_limits(config)
        initial_agent = choose_initial_handoff_agent(input_data)
        state = NativeHandoffState(initial_agent=initial_agent, limits=limits)
        agents = {
            agent_name: _build_native_agent(
                agent_name=agent_name,
                input_data=input_data,
                config=config,
                context=context,
                state=state,
            )
            for agent_name in AGENT_INSTRUCTIONS
        }

        def termination_condition(_messages) -> bool:
            if state.stop_reason is not None:
                return True
            if state.number_of_agent_invocations >= limits["max_agent_invocations"]:
                state.stop_reason = "max_agent_invocations_reached"
                state.finalizing_agent = state.active_agent
                state.final_answer = build_fallback_answer(input_data, state.partial_results)
                return True
            return False

        builder = (
            HandoffBuilder(
                name="ARCH_05_HANDOFF_SWARM",
                participants=list(agents.values()),
                termination_condition=termination_condition,
                output_from="all",
            )
            .with_start_agent(agents[initial_agent])
        )
        for source_agent, target_agents in HANDOFF_TARGETS.items():
            builder.add_handoff(
                agents[source_agent],
                [agents[target_agent] for target_agent in target_agents],
                description=f"ARCH_05 handoff from {source_agent}.",
            )

        workflow = builder.build()
        workflow_events = run_async(workflow.run(
            render_handoff_swarm_prompt(input_data, initial_agent, {}),
            stream=False,
            include_status_events=True,
        ))
        for event in workflow_events:
            if event.type == "handoff_sent":
                state.native_handoff_events.append(
                    {
                        "source_agent": event.data.source,
                        "target_agent": event.data.target,
                    }
                )

        final_answer = state.final_answer or build_fallback_answer(input_data, state.partial_results)
        structured_output = {
            "answer": final_answer,
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
            "framework_native_primitives": ["HandoffBuilder", "HandoffConfiguration", "handoff_to tool calls", framework_execution("handoff_swarm_native", context)],
            "native_automatic_behaviors": ["tool_call_interception", "handoff_sent_events"],
            "native_handoff_events": state.native_handoff_events,
            "parallelism_used": False,
            "warnings": state.warnings,
            "document_ids": document_ids(input_data),
            "framework_execution": framework_execution("handoff_swarm_native_handoff_builder", context),
        }
        return MicrosoftAgentFrameworkRunOutput(
            final_answer=final_answer,
            structured_output=structured_output,
            steps=state.steps,
            llm_calls=state.llm_calls,
        )

    with ResourceMonitor() as monitor:
        output = execute()
        output.resource_usage = monitor.usage
    return output
