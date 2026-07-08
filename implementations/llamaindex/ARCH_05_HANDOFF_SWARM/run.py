"""LlamaIndex implementation for ARCH_05_HANDOFF_SWARM."""

from __future__ import annotations

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
from implementations.llamaindex.utils_llamaindex import (
    LlamaIndexRunContext,
    LlamaIndexRunOutput,
    build_function_agent,
    complete_agent_step,
    framework_execution,
    llamaindex_architecture_runner,
    run_with_resource_monitor,
)


AGENT_PROMPTS = {
    "data_specialist": "You are DataSpecialist in a LlamaIndex handoff workflow.",
    "reasoning_specialist": "You are ReasoningSpecialist in a LlamaIndex handoff workflow.",
    "validation_specialist": "You are ValidationSpecialist in a LlamaIndex handoff workflow.",
    "synthesis_specialist": "You are SynthesisSpecialist in a LlamaIndex handoff workflow.",
}


@llamaindex_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: LlamaIndexRunContext,
) -> LlamaIndexRunOutput:
    """Execute sequential handoffs with LlamaIndex FunctionAgent specialists."""

    def execute() -> LlamaIndexRunOutput:
        agents = {
            agent_name: build_function_agent(
                name=agent_name,
                system_prompt=prompt,
                context=context,
                input_data=input_data,
                config=config,
            )
            for agent_name, prompt in AGENT_PROMPTS.items()
        }
        limits = get_handoff_limits(config)
        active_agent = choose_initial_handoff_agent(input_data)
        initial_agent = active_agent
        active_agent_history: list[str] = []
        handoff_history: list[dict] = []
        partial_results: list[dict] = []
        repeated_agent_visits: dict[str, int] = {}
        warnings: list[str] = []
        steps: list[AgentStep] = []
        llm_calls = []
        number_of_handoffs = 0
        number_of_agent_invocations = 0
        cycle_detected = False
        fallback_used = False
        finalizing_agent = None
        stop_reason = None
        final_answer = ""
        last_decision: dict = {}
        context_transferred = ""

        while stop_reason is None:
            if number_of_agent_invocations >= limits["max_agent_invocations"]:
                stop_reason = "max_agent_invocations_reached"
                finalizing_agent = active_agent
                final_answer = build_fallback_answer(input_data, partial_results)
                break

            repeated_agent_visits[active_agent] = repeated_agent_visits.get(active_agent, 0) + 1
            if repeated_agent_visits[active_agent] > limits["max_consecutive_visits_per_agent"]:
                cycle_detected = True
                fallback_used = True
                stop_reason = "max_consecutive_visits_per_agent_reached"
                finalizing_agent = active_agent
                final_answer = build_fallback_answer(input_data, partial_results)
                warnings.append(f"Visit limit reached for {active_agent}.")
                break

            prompt_state = {
                "active_agent_history": active_agent_history,
                "handoff_history": handoff_history,
                "partial_results": partial_results,
                "context_transferred": context_transferred,
                "number_of_handoffs": number_of_handoffs,
                "number_of_agent_invocations": number_of_agent_invocations,
                "warnings": warnings,
            }
            prompt = render_handoff_swarm_prompt(input_data, active_agent, prompt_state)
            step_started_at = utc_now()
            step_id = len(steps) + 1
            call_record = complete_agent_step(
                agent=agents[active_agent],
                prompt=prompt,
                input_data=input_data,
                config=config,
                step_id=step_id,
            )
            decision = parse_handoff_decision(call_record.response.strip())
            last_decision = decision
            number_of_agent_invocations += 1
            active_agent_history.append(active_agent)
            partial_results.append({"agent": active_agent, "decision": decision})
            llm_calls.append(call_record.metrics)
            steps.append(
                AgentStep(
                    step_id=step_id,
                    name=active_agent,
                    step_type="handoff_agent_llm_call",
                    actor=f"llamaindex.workflow.{active_agent}",
                    input_data={"prompt": prompt, "active_agent": active_agent},
                    output_data={"decision": decision},
                    llm_call_ids=[call_record.metrics.call_id],
                    started_at=step_started_at,
                    finished_at=utc_now(),
                    metadata={
                        "architecture": "ARCH_05_HANDOFF_SWARM",
                        "native_primitive": "FunctionAgent handoff workflow",
                        "native_framework_available": context.native_framework_available,
                    },
                )
            )

            if not is_valid_handoff_decision(decision):
                fallback_used = True
                warnings.append(f"Invalid handoff decision from {active_agent}.")
                if active_agent != "synthesis_specialist" and number_of_handoffs < limits["max_handoffs"]:
                    target_agent = "synthesis_specialist"
                    handoff_history.append(
                        {
                            "sequence_number": number_of_handoffs + 1,
                            "source_agent": active_agent,
                            "target_agent": target_agent,
                            "reason": "Fallback after invalid decision.",
                            "task": "Finalize from available context.",
                            "context_summary": "Fallback context after invalid decision.",
                            "timestamp": utc_now().isoformat(),
                        }
                    )
                    number_of_handoffs += 1
                    context_transferred = "Fallback context after invalid decision."
                    active_agent = target_agent
                    continue
                stop_reason = "invalid_decision_fallback_finalized"
                finalizing_agent = active_agent
                final_answer = build_fallback_answer(input_data, partial_results)
                break

            if decision["action"] == "finalize":
                stop_reason = "agent_finalized"
                finalizing_agent = active_agent
                final_answer = extract_final_answer(str(decision["final_output"]))
                break

            target_agent = str(decision["target_agent"])
            cycle_detected = cycle_detected or (
                len(active_agent_history) >= 2 and target_agent == active_agent_history[-2]
            )
            if number_of_handoffs >= limits["max_handoffs"]:
                stop_reason = "max_handoffs_reached"
                finalizing_agent = active_agent
                final_answer = build_fallback_answer(input_data, partial_results)
                break

            handoff_history.append(
                {
                    "sequence_number": number_of_handoffs + 1,
                    "source_agent": active_agent,
                    "target_agent": target_agent,
                    "reason": decision["reason"],
                    "task": decision["task"],
                    "context_summary": decision["context_summary"],
                    "timestamp": utc_now().isoformat(),
                }
            )
            number_of_handoffs += 1
            context_transferred = str(decision["context_summary"])
            active_agent = target_agent

        final_answer = final_answer or build_fallback_answer(input_data, partial_results)
        structured_output = {
            "answer": final_answer,
            "decision": last_decision,
            "confidence": last_decision.get("confidence", 0.0),
            "evidence": last_decision.get("evidence", "none"),
            "limitations": last_decision.get("limitations", "none"),
            "initial_agent": initial_agent,
            "active_agent_history": active_agent_history,
            "handoff_history": handoff_history,
            "number_of_handoffs": number_of_handoffs,
            "max_handoffs": limits["max_handoffs"],
            "number_of_agent_invocations": number_of_agent_invocations,
            "max_agent_invocations": limits["max_agent_invocations"],
            "unique_agents_executed": sorted(set(active_agent_history), key=HANDOFF_AGENTS.index),
            "finalizing_agent": finalizing_agent,
            "repeated_agent_visits": repeated_agent_visits,
            "cycle_detected": cycle_detected,
            "fallback_used": fallback_used,
            "stop_reason": stop_reason,
            "framework_native_primitives": ["FunctionAgent", "workflow_state_loop"],
            "native_automatic_behaviors": [],
            "parallelism_used": False,
            "warnings": warnings,
            "document_ids": document_ids(input_data),
            "framework_execution": framework_execution("handoff_swarm_workflow", context),
        }
        return LlamaIndexRunOutput(
            final_answer=final_answer,
            structured_output=structured_output,
            steps=steps,
            llm_calls=llm_calls,
        )

    return run_with_resource_monitor(execute)
