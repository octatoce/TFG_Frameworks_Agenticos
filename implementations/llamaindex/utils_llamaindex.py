"""Shared utilities for LlamaIndex benchmark implementations."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

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
    from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
    from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step
    from llama_index.llms.openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency guard
    AgentWorkflow = None
    FunctionAgent = None
    Event = None
    StartEvent = None
    StopEvent = None
    Workflow = None
    step = None
    OpenAI = None


@dataclass
class LlamaIndexRunContext:
    repo_root: Path
    llm: InstrumentedLLM
    native_framework_available: bool


@dataclass
class LlamaIndexRunOutput:
    final_answer: str
    structured_output: dict[str, Any]
    steps: list[AgentStep]
    llm_calls: list[LLMCallMetrics]
    resource_usage: ResourceUsage | None = None


LlamaIndexImplementation = Callable[
    [ExperimentInput, ExperimentConfig, LlamaIndexRunContext],
    LlamaIndexRunOutput,
]


class DeterministicLlamaIndexAgent:
    """Small local FunctionAgent-like surface used for the reproducible baseline."""

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


def run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("LlamaIndex runners require a synchronous entrypoint.")


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def native_framework_available() -> bool:
    return find_spec("llama_index") is not None and find_spec("llama_index.llms.openai") is not None


def uses_openai(config: ExperimentConfig) -> bool:
    return config.model_provider.lower() == "openai"


def build_openai_llm(config: ExperimentConfig):
    if OpenAI is None:
        raise RuntimeError("llama-index-llms-openai is required for LlamaIndex OpenAI runs.")
    return OpenAI(
        model=config.model_name,
        api_key=config.metadata.get("openai_api_key"),
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=float(config.timeout_seconds),
        max_retries=config.retry_count,
    )


def build_function_agent(*, name: str, system_prompt: str, context: LlamaIndexRunContext, input_data: ExperimentInput, config: ExperimentConfig):
    if uses_openai(config):
        if FunctionAgent is None:
            raise RuntimeError("llama-index is required for LlamaIndex OpenAI runs.")
        return FunctionAgent(
            name=name,
            description=f"Benchmark component {name}",
            llm=build_openai_llm(config),
            system_prompt=system_prompt,
            tools=[],
            allow_parallel_tool_calls=False,
            timeout=float(config.timeout_seconds),
            verbose=False,
        )
    return DeterministicLlamaIndexAgent(
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
    async def run_agent() -> Any:
        handler = agent.run(user_msg=prompt)
        return await handler

    started = perf_counter()
    response = run_async(run_agent())
    latency_seconds = max(perf_counter() - started, 0.0)
    response_text = str(response).strip()
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
            "framework_api": "llamaindex",
            "agent_type": "FunctionAgent",
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
    return agent.run(prompt, step_id=step_id)



def extract_final_answer(response: str) -> str:
    marker = "Final Answer:"
    if marker not in response:
        return response.strip()
    return response.split(marker, maxsplit=1)[1].strip()


def document_ids(input_data: ExperimentInput) -> list[str]:
    return [document.document_id for document in input_data.documents]


def next_step_id(state: dict[str, Any]) -> int:
    return len(state.get("steps", [])) + 1


def framework_execution(label: str, context: LlamaIndexRunContext) -> str:
    suffix = "native_available" if context.native_framework_available else "deterministic_adapter"
    return f"llamaindex_{label}_{suffix}"


def run_with_resource_monitor(callback: Callable[[], LlamaIndexRunOutput]) -> LlamaIndexRunOutput:
    with ResourceMonitor() as monitor:
        output = callback()
        output.resource_usage = monitor.usage
    return output


def llamaindex_architecture_runner(run_impl: LlamaIndexImplementation):
    def wrapper(
        input_data: ExperimentInput,
        config: ExperimentConfig,
    ) -> ExperimentResult:
        repo_root = get_repo_root()
        started_at = utc_now()
        resource_usage = None

        try:
            context = LlamaIndexRunContext(
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
            environment_packages=["llama-index"],
            repo_root=repo_root,
        )

    return wrapper
