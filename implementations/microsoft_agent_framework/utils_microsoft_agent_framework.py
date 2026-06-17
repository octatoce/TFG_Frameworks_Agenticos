"""Shared utilities for Microsoft Agent Framework benchmark implementations."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
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
class MicrosoftAgentFrameworkRunContext:
    repo_root: Path
    llm: InstrumentedLLM
    native_framework_available: bool


@dataclass
class MicrosoftAgentFrameworkRunOutput:
    final_answer: str
    structured_output: dict[str, Any]
    steps: list[AgentStep]
    llm_calls: list[LLMCallMetrics]
    resource_usage: ResourceUsage | None = None


MicrosoftAgentFrameworkImplementation = Callable[
    [ExperimentInput, ExperimentConfig, MicrosoftAgentFrameworkRunContext],
    MicrosoftAgentFrameworkRunOutput,
]


class DeterministicMicrosoftAgent:
    """Small local agent surface used for the reproducible baseline."""

    def __init__(self, *, name: str, llm: InstrumentedLLM, input_data: ExperimentInput, config: ExperimentConfig) -> None:
        self.name = name
        self.llm = llm
        self.input_data = input_data
        self.config = config

    def run(self, prompt: str, *, step_id: int) -> Any:
        return self.llm.complete(
            prompt=prompt,
            input_data=self.input_data,
            call_id=f"{self.config.run_id}-llm-{step_id:03d}",
            step_id=step_id,
        )


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def native_framework_available() -> bool:
    return find_spec("agent_framework") is not None


def extract_final_answer(response: str) -> str:
    marker = "Final Answer:"
    if marker not in response:
        return response.strip()
    return response.split(marker, maxsplit=1)[1].strip()


def document_ids(input_data: ExperimentInput) -> list[str]:
    return [document.document_id for document in input_data.documents]


def next_step_id(state: dict[str, Any]) -> int:
    return len(state.get("steps", [])) + 1


def framework_execution(label: str, context: MicrosoftAgentFrameworkRunContext) -> str:
    suffix = "native_available" if context.native_framework_available else "deterministic_adapter"
    return f"microsoft_agent_framework_{label}_{suffix}"


def run_with_resource_monitor(callback: Callable[[], MicrosoftAgentFrameworkRunOutput]) -> MicrosoftAgentFrameworkRunOutput:
    with ResourceMonitor() as monitor:
        output = callback()
        output.resource_usage = monitor.usage
    return output


def microsoft_agent_framework_architecture_runner(run_impl: MicrosoftAgentFrameworkImplementation):
    def wrapper(
        input_data: ExperimentInput,
        config: ExperimentConfig,
    ) -> ExperimentResult:
        repo_root = get_repo_root()
        started_at = utc_now()
        resource_usage = None

        try:
            context = MicrosoftAgentFrameworkRunContext(
                repo_root=repo_root,
                llm=build_llm_from_config(config),
                native_framework_available=native_framework_available(),
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
            environment_packages=["agent-framework"],
            repo_root=repo_root,
        )

    return wrapper

