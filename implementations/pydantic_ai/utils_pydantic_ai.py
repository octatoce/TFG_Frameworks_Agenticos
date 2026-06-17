"""Shared utilities for Pydantic AI benchmark implementations."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

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


class PydanticAIStructuredOutput(BaseModel):
    answer: str
    mode: str
    document_ids: list[str] = Field(default_factory=list)
    framework_execution: str


class PydanticAIPipelineOutput(PydanticAIStructuredOutput):
    plan: str
    evidence: str
    analysis: str


class PydanticAISupervisorOutput(PydanticAIStructuredOutput):
    selected_workers: list[str]
    skipped_workers: list[str]
    evidence: str | list[Any]
    preliminary_decision: str | None
    validation_report: str | None


@dataclass
class PydanticAIRunContext:
    repo_root: Path
    llm: InstrumentedLLM
    native_framework_available: bool
    native_graph_available: bool


@dataclass
class PydanticAIRunOutput:
    final_answer: str
    structured_output: dict[str, Any]
    steps: list[AgentStep]
    llm_calls: list[LLMCallMetrics]
    resource_usage: ResourceUsage | None = None


PydanticAIImplementation = Callable[
    [ExperimentInput, ExperimentConfig, PydanticAIRunContext],
    PydanticAIRunOutput,
]


class DeterministicPydanticAgent:
    """Small typed-agent surface used for the reproducible baseline."""

    def __init__(self, *, name: str, llm: InstrumentedLLM, input_data: ExperimentInput, config: ExperimentConfig) -> None:
        self.name = name
        self.llm = llm
        self.input_data = input_data
        self.config = config

    def run_sync(self, prompt: str, *, step_id: int) -> Any:
        return self.llm.complete(
            prompt=prompt,
            input_data=self.input_data,
            call_id=f"{self.config.run_id}-llm-{step_id:03d}",
            step_id=step_id,
        )


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def native_framework_available() -> bool:
    return find_spec("pydantic_ai") is not None


def native_graph_available() -> bool:
    return find_spec("pydantic_graph") is not None


def extract_final_answer(response: str) -> str:
    marker = "Final Answer:"
    if marker not in response:
        return response.strip()
    return response.split(marker, maxsplit=1)[1].strip()


def document_ids(input_data: ExperimentInput) -> list[str]:
    return [document.document_id for document in input_data.documents]


def next_step_id(state: dict[str, Any]) -> int:
    return len(state.get("steps", [])) + 1


def framework_execution(label: str, context: PydanticAIRunContext) -> str:
    suffix = "native_available" if context.native_framework_available else "deterministic_adapter"
    if label == "supervisor_workers" and context.native_graph_available:
        suffix = f"{suffix}_graph_available"
    return f"pydantic_ai_{label}_{suffix}"


def run_with_resource_monitor(callback: Callable[[], PydanticAIRunOutput]) -> PydanticAIRunOutput:
    with ResourceMonitor() as monitor:
        output = callback()
        output.resource_usage = monitor.usage
    return output


def pydantic_ai_architecture_runner(run_impl: PydanticAIImplementation):
    def wrapper(
        input_data: ExperimentInput,
        config: ExperimentConfig,
    ) -> ExperimentResult:
        repo_root = get_repo_root()
        started_at = utc_now()
        resource_usage = None

        try:
            context = PydanticAIRunContext(
                repo_root=repo_root,
                llm=build_llm_from_config(config),
                native_framework_available=native_framework_available(),
                native_graph_available=native_graph_available(),
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
            environment_packages=["pydantic-ai", "pydantic-graph"],
            repo_root=repo_root,
        )

    return wrapper

