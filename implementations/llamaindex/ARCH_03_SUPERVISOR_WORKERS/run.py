"""LlamaIndex implementation for ARCH_03_SUPERVISOR_WORKERS."""

from __future__ import annotations

from benchmark_core.llm_wrapper import parse_worker_selection, render_supervisor_workers_prompt
from benchmark_core.schemas import AgentStep, ExperimentConfig, ExperimentInput
from benchmark_core.tracing import utc_now
from implementations.llamaindex.utils_llamaindex import (
    LlamaIndexRunContext,
    LlamaIndexRunOutput,
    build_function_agent,
    complete_agent_step,
    document_ids,
    extract_final_answer,
    framework_execution,
    llamaindex_architecture_runner,
    next_step_id,
    run_with_resource_monitor,
)


WORKERS = ["data_worker", "reasoning_worker", "validation_worker"]
OUTPUT_KEYS = {
    "supervisor_planning": "supervisor_plan",
    "data_worker": "evidence",
    "reasoning_worker": "preliminary_decision",
    "validation_worker": "validation_report",
    "supervisor_synthesis": "final_output",
}


@llamaindex_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: LlamaIndexRunContext,
) -> LlamaIndexRunOutput:
    """Execute a LlamaIndex orchestrator workflow with controlled workers."""

    def execute() -> LlamaIndexRunOutput:
        orchestrator = build_function_agent(
            name="supervisor_workers_workflow",
            system_prompt=(
                "You are a LlamaIndex supervisor orchestrator. "
                "All worker coordination must pass through the supervisor state."
            ),
            context=context,
            input_data=input_data,
            config=config,
        )
        state: dict[str, object] = {
            "query": input_data.query,
            "documents": list(input_data.documents),
            "selected_workers": [],
            "skipped_workers": [],
            "supervisor_plan": None,
            "evidence": [],
            "preliminary_decision": None,
            "validation_report": None,
            "final_output": None,
            "steps": [],
            "llm_calls": [],
        }

        def run_phase(phase: str) -> None:
            step_id = next_step_id(state)
            step_started_at = utc_now()
            prompt = render_supervisor_workers_prompt(input_data, phase=phase, state=state)
            call_record = complete_agent_step(
                agent=orchestrator,
                prompt=prompt,
                input_data=input_data,
                config=config,
                step_id=step_id,
            )
            phase_output = call_record.response.strip()

            if phase == "supervisor_planning":
                selected_workers, skipped_workers = parse_worker_selection(phase_output, WORKERS)
                state["selected_workers"] = selected_workers
                state["skipped_workers"] = skipped_workers

            state[OUTPUT_KEYS[phase]] = phase_output
            step = AgentStep(
                step_id=step_id,
                name=phase,
                step_type="supervisor_worker_llm_call",
                actor=f"llamaindex.workflow.{phase}",
                input_data={"phase": phase, "prompt": prompt},
                output_data={
                    "phase_output": phase_output,
                    "selected_workers": state["selected_workers"],
                    "skipped_workers": state["skipped_workers"],
                },
                llm_call_ids=[call_record.metrics.call_id],
                started_at=step_started_at,
                finished_at=utc_now(),
                metadata={
                    "pipeline_order": step_id,
                    "worker_role": phase if phase in WORKERS else None,
                    "handoff_policy": "supervisor_only",
                    "native_framework_available": context.native_framework_available,
                },
            )
            state["steps"] = [*state["steps"], step]
            state["llm_calls"] = [*state["llm_calls"], call_record.metrics]

        run_phase("supervisor_planning")
        for worker in WORKERS:
            if worker in state["selected_workers"]:
                run_phase(worker)
        run_phase("supervisor_synthesis")

        final_answer = extract_final_answer(str(state["final_output"]))
        structured_output = {
            "answer": final_answer,
            "mode": f"{config.model_provider}_supervisor_workers",
            "selected_workers": state["selected_workers"],
            "skipped_workers": state["skipped_workers"],
            "evidence": state["evidence"],
            "preliminary_decision": state["preliminary_decision"],
            "validation_report": state["validation_report"],
            "document_ids": document_ids(input_data),
            "framework_execution": framework_execution("agent_workflow_supervisor", context),
        }
        return LlamaIndexRunOutput(
            final_answer=final_answer,
            structured_output=structured_output,
            steps=state["steps"],
            llm_calls=state["llm_calls"],
        )

    return run_with_resource_monitor(execute)
