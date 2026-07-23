"""Shared result construction helpers."""

from __future__ import annotations

from pathlib import Path

from benchmark_core.environment import build_environment_info
from benchmark_core.metrics import build_metrics
from benchmark_core.schemas import (
    AgentStep,
    ExperimentConfig,
    ExperimentError,
    ExperimentInput,
    ExperimentResult,
    LLMCallMetrics,
    ResourceUsage,
    RunStatus,
)


def build_experiment_result(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    status: RunStatus,
    final_answer: str,
    structured_output: dict,
    steps: list[AgentStep],
    llm_calls: list[LLMCallMetrics],
    errors: list[ExperimentError],
    started_at,
    finished_at,
    resource_usage: ResourceUsage | None = None,
    environment_packages: list[str] | None = None,
    repo_root: Path | None = None,
) -> ExperimentResult:
    """Build a complete canonical result object."""

    metrics = build_metrics(
        started_at=started_at,
        finished_at=finished_at,
        steps=steps,
        errors=errors,
        llm_calls=llm_calls,
        resource_usage=resource_usage,
    )
    metrics.metadata["latency_total_ms"] = metrics.total_latency_seconds * 1000
    parallel_execution = structured_output.get("parallel_execution")
    if isinstance(parallel_execution, dict):
        metrics.metadata["parallel_execution"] = parallel_execution
    map_reduce_execution = structured_output.get("map_reduce_execution")
    if isinstance(map_reduce_execution, dict):
        metrics.metadata["map_reduce_execution"] = map_reduce_execution
    debate_execution = structured_output.get("debate_execution")
    if isinstance(debate_execution, dict):
        metrics.metadata["debate_execution"] = debate_execution
    reflection_execution = structured_output.get("reflection_execution")
    if isinstance(reflection_execution, dict):
        metrics.metadata["reflection_execution"] = reflection_execution
    recovery_execution = structured_output.get("recovery_execution")
    if isinstance(recovery_execution, dict):
        metrics.metadata["recovery_execution"] = recovery_execution

    return ExperimentResult(
        case_id=input_data.case_id,
        dataset_id=input_data.dataset_id,
        framework=config.framework,
        architecture=config.architecture,
        run_id=config.run_id,
        status=status,
        final_answer=final_answer,
        structured_output=structured_output,
        input_snapshot=input_data,
        config_snapshot=config,
        metrics=metrics,
        steps=steps,
        llm_calls=llm_calls,
        errors=errors,
        environment=build_environment_info(
            package_names=environment_packages,
            repo_root=repo_root,
        ),
        started_at=started_at,
        finished_at=finished_at,
    )
