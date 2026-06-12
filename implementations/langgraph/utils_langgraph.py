"""Shared utilities for LangGraph benchmark implementations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from benchmark_core.llm_wrapper import InstrumentedLLM, build_llm_from_config
from benchmark_core.resource_monitor import ResourceMonitor
from benchmark_core.result_builders import build_experiment_result
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
from benchmark_core.tracing import utc_now


@dataclass
class LangGraphRunContext:
    repo_root: Path
    llm: InstrumentedLLM


@dataclass
class LangGraphRunOutput:
    final_answer: str
    structured_output: dict[str, Any]
    steps: list[AgentStep]
    llm_calls: list[LLMCallMetrics]
    resource_usage: ResourceUsage | None = None


LangGraphImplementation = Callable[
    [ExperimentInput, ExperimentConfig, LangGraphRunContext],
    LangGraphRunOutput,
]


def get_repo_root() -> Path:
    """Return the repository root from this shared LangGraph module."""

    return Path(__file__).resolve().parents[2]


def extract_final_answer(response: str) -> str:
    marker = "Final Answer:"
    if marker not in response:
        return response.strip()
    return response.split(marker, maxsplit=1)[1].strip()


def document_ids(input_data: ExperimentInput) -> list[str]:
    return [document.document_id for document in input_data.documents]


def next_step_id(state: dict[str, Any]) -> int:
    return len(state.get("steps", [])) + 1


def complete_llm_step(
    *,
    llm: InstrumentedLLM,
    input_data: ExperimentInput,
    config: ExperimentConfig,
    prompt: str,
    step_id: int,
):
    return llm.complete(
        prompt=prompt,
        input_data=input_data,
        call_id=f"{config.run_id}-llm-{step_id:03d}",
        step_id=step_id,
    )


def invoke_with_resource_monitor(compiled_graph: Any, initial_state: dict[str, Any]) -> tuple[dict[str, Any], ResourceUsage]:
    with ResourceMonitor() as monitor:
        state = compiled_graph.invoke(initial_state)
        resource_usage = monitor.usage
    return state, resource_usage


def langgraph_architecture_runner(run_impl: LangGraphImplementation):
    def wrapper(
        input_data: ExperimentInput,
        config: ExperimentConfig,
    ) -> ExperimentResult:
        repo_root = get_repo_root()
        started_at = utc_now()
        resource_usage = None

        try:
            context = LangGraphRunContext(
                repo_root=repo_root,
                llm=build_llm_from_config(config),
            )
            output = run_impl(input_data, config, context)
            status = RunStatus.SUCCESS
            final_answer = output.final_answer
            structured_output = output.structured_output
            steps = output.steps
            llm_calls = output.llm_calls
            errors: list[ExperimentError] = []
            resource_usage = output.resource_usage
        except Exception as exc:  # pragma: no cover - exercised by integration failures
            status = RunStatus.ERROR
            final_answer = ""
            structured_output = {"error": True}
            errors = [
                ExperimentError(
                    error_type=type(exc).__name__,
                    message=str(exc),
                    recoverable=False,
                )
            ]
            steps = []
            llm_calls = []

        return build_experiment_result(
            input_data=input_data,
            config=config,
            status=status,
            final_answer=final_answer,
            structured_output=structured_output,
            steps=steps,
            llm_calls=llm_calls,
            errors=errors,
            started_at=started_at,
            finished_at=utc_now(),
            resource_usage=resource_usage,
            environment_packages=["langgraph"],
            repo_root=repo_root,
        )

    return wrapper
