"""Microsoft Agent Framework implementation for ARCH_04_SUPERVISOR_WORKERS."""

from __future__ import annotations

from benchmark_core.schemas import ExperimentConfig, ExperimentInput
from implementations.microsoft_agent_framework.utils_microsoft_agent_framework import (
    MicrosoftAgentFrameworkRunContext,
    MicrosoftAgentFrameworkRunOutput,
    build_agent,
    complete_agent_step,
    framework_execution,
    microsoft_agent_framework_architecture_runner,
    run_with_resource_monitor,
)
from implementations.supervisor_workers_common import run_supervisor_workers_loop


@microsoft_agent_framework_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: MicrosoftAgentFrameworkRunContext,
) -> MicrosoftAgentFrameworkRunOutput:
    """Execute a centralized supervisor/workers workflow."""

    def execute() -> MicrosoftAgentFrameworkRunOutput:
        supervisor_agent = build_agent(
            name="supervisor_workers_orchestrator",
            instructions=(
                "You are the centralized supervisor for ARCH_04_SUPERVISOR_WORKERS. "
                "Plan worker use, review every worker output, request bounded revisions only when needed, "
                "and finalize when quality criteria are satisfied or the iteration limit is reached."
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
            framework_execution=framework_execution("supervisor_workers", context),
            actor_prefix="microsoft_agent_framework",
            run_prompt=run_prompt,
        )
        return MicrosoftAgentFrameworkRunOutput(
            final_answer=result.final_answer,
            structured_output=result.structured_output,
            steps=result.steps,
            llm_calls=result.llm_calls,
        )

    return run_with_resource_monitor(execute)
