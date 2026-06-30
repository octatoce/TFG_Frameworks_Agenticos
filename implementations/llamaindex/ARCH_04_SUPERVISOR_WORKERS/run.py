"""LlamaIndex implementation for ARCH_04_SUPERVISOR_WORKERS."""

from __future__ import annotations

from benchmark_core.schemas import ExperimentConfig, ExperimentInput
from implementations.llamaindex.utils_llamaindex import (
    LlamaIndexRunContext,
    LlamaIndexRunOutput,
    build_function_agent,
    complete_agent_step,
    framework_execution,
    llamaindex_architecture_runner,
    run_with_resource_monitor,
)
from implementations.supervisor_workers_common import run_supervisor_workers_loop


@llamaindex_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: LlamaIndexRunContext,
) -> LlamaIndexRunOutput:
    """Execute a LlamaIndex-style supervised workflow with bounded state."""

    def execute() -> LlamaIndexRunOutput:
        supervisor_agent = build_function_agent(
            name="supervisor_workers_workflow",
            system_prompt=(
                "You are the centralized supervisor for ARCH_04_SUPERVISOR_WORKERS. "
                "Use workflow state to choose one next worker at a time, review outputs, "
                "request bounded revisions if needed, and finalize deterministically."
            ),
            context=context,
            input_data=input_data,
            config=config,
        )

        def run_prompt(prompt: str, step_id: int, phase: str, worker_name: str | None):
            return complete_agent_step(
                agent=supervisor_agent,
                prompt=prompt,
                input_data=input_data,
                config=config,
                step_id=step_id,
            )

        result = run_supervisor_workers_loop(
            input_data=input_data,
            config=config,
            framework_execution=framework_execution("workflow_supervisor_workers", context),
            actor_prefix="llamaindex.workflow",
            run_prompt=run_prompt,
        )
        return LlamaIndexRunOutput(
            final_answer=result.final_answer,
            structured_output=result.structured_output,
            steps=result.steps,
            llm_calls=result.llm_calls,
        )

    return run_with_resource_monitor(execute)
