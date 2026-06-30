"""Pydantic AI implementation for ARCH_04_SUPERVISOR_WORKERS."""

from __future__ import annotations

from benchmark_core.schemas import ExperimentConfig, ExperimentInput
from implementations.pydantic_ai.utils_pydantic_ai import (
    PydanticAIRunContext,
    PydanticAIRunOutput,
    PydanticAISupervisorWorkersOutput,
    build_typed_agent,
    complete_agent_step,
    framework_execution,
    pydantic_ai_architecture_runner,
    run_with_resource_monitor,
)
from implementations.supervisor_workers_common import run_supervisor_workers_loop


@pydantic_ai_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: PydanticAIRunContext,
) -> PydanticAIRunOutput:
    """Execute a typed supervisor/workers graph-shaped flow."""

    def execute() -> PydanticAIRunOutput:
        supervisor_agent = build_typed_agent(
            name="supervisor_workers_graph",
            instructions=(
                "You are a typed Pydantic AI supervisor for ARCH_04_SUPERVISOR_WORKERS. "
                "Maintain bounded state, validate each decision shape, review worker outputs, "
                "and finalize only after quality control or iteration exhaustion."
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
            actor_prefix="pydantic_ai.graph",
            run_prompt=run_prompt,
        )
        structured_model = PydanticAISupervisorWorkersOutput.model_validate(result.structured_output)
        return PydanticAIRunOutput(
            final_answer=result.final_answer,
            structured_output=structured_model.model_dump(),
            steps=result.steps,
            llm_calls=result.llm_calls,
        )

    return run_with_resource_monitor(execute)
