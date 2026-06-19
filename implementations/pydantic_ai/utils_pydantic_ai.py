"""Shared utilities for Pydantic AI benchmark implementations."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from pydantic import BaseModel, Field

from benchmark_core.llm_wrapper import (
    InstrumentedLLM,
    LLMCallRecord,
    build_llm_from_config,
    estimate_token_usage,
)
from benchmark_core.metrics import estimate_cost_usd
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
    TokenUsage,
)
from benchmark_core.tracing import utc_now

try:
    from pydantic_ai import Agent
    from pydantic_graph import BaseNode, End, GraphBuilder, GraphRunContext, StepContext
except ImportError:  # pragma: no cover - optional dependency guard
    Agent = None
    BaseNode = None
    End = None
    GraphBuilder = None
    GraphRunContext = None
    StepContext = None


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


def uses_openai(config: ExperimentConfig) -> bool:
    return config.model_provider.lower() == "openai"


def build_typed_agent(*, name: str, instructions: str, context: PydanticAIRunContext, input_data: ExperimentInput, config: ExperimentConfig):
    if uses_openai(config):
        if Agent is None:
            raise RuntimeError("pydantic-ai is required for Pydantic AI OpenAI runs.")
        return Agent(
            f"openai:{config.model_name}",
            output_type=str,
            instructions=instructions,
            name=name,
            retries=config.retry_count,
        )
    return DeterministicPydanticAgent(
        name=name,
        llm=context.llm,
        input_data=input_data,
        config=config,
    )


def complete_openai_agent_step(
    *,
    agent: Any,
    prompt: str,
    input_data: ExperimentInput,
    config: ExperimentConfig,
    step_id: int,
) -> LLMCallRecord:
    started = perf_counter()
    result = agent.run_sync(prompt)
    latency_seconds = max(perf_counter() - started, 0.0)
    response_text = str(result.output).strip()
    token_usage = TokenUsage(
        input_tokens=estimate_token_usage(prompt),
        output_tokens=estimate_token_usage(response_text),
        total_tokens=estimate_token_usage(prompt) + estimate_token_usage(response_text),
    )
    metrics = LLMCallMetrics(
        call_id=f"{config.run_id}-llm-{step_id:03d}",
        step_id=step_id,
        model_provider=config.model_provider,
        model_name=config.model_name,
        latency_seconds=latency_seconds,
        token_usage=token_usage,
        estimated_cost_usd=estimate_cost_usd(
            token_usage,
            input_cost_per_1k=float(config.metadata.get("input_cost_per_1k_tokens", 0.0)),
            output_cost_per_1k=float(config.metadata.get("output_cost_per_1k_tokens", 0.0)),
        ),
        finish_reason=None,
        metadata={
            "deterministic": False,
            "framework_api": "pydantic_ai",
            "agent_type": "Agent[str]",
            "token_counting_method": "whitespace_proxy",
        },
    )
    return LLMCallRecord(
        model_name=config.model_name,
        prompt=prompt,
        response=response_text,
        metrics=metrics,
    )


def complete_agent_step(
    *,
    agent: Any,
    prompt: str,
    input_data: ExperimentInput,
    config: ExperimentConfig,
    step_id: int,
) -> LLMCallRecord:
    if uses_openai(config):
        return complete_openai_agent_step(
            agent=agent,
            prompt=prompt,
            input_data=input_data,
            config=config,
            step_id=step_id,
        )
    return agent.run_sync(prompt, step_id=step_id)


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
