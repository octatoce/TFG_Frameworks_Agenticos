"""Shared utilities for Microsoft Agent Framework benchmark implementations."""

from __future__ import annotations

import asyncio
import os
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
from benchmark_core.result_writer import save_result_json
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

os.environ.setdefault("AGENT_FRAMEWORK_USER_AGENT_DISABLED", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

try:
    from agent_framework import (
        Agent,
        AgentResponse,
        BaseAgent,
        Content,
        Executor,
        Message,
        ResponseStream,
        WorkflowBuilder,
        WorkflowContext,
        executor,
        handler,
        normalize_messages,
    )
    from agent_framework.openai import OpenAIChatClient, OpenAIChatCompletionClient
except ImportError:  # pragma: no cover - optional dependency guard
    Agent = None
    AgentResponse = None
    BaseAgent = object
    Content = None
    Executor = None
    Message = None
    ResponseStream = None
    WorkflowBuilder = None
    WorkflowContext = None
    executor = None
    handler = None
    normalize_messages = None
    OpenAIChatClient = None
    OpenAIChatCompletionClient = None


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


class OpenAIBenchmarkMicrosoftAgent(BaseAgent):
    """Agent Framework custom agent that calls OpenAI through the official SDK."""

    def __init__(
        self,
        *,
        name: str,
        instructions: str,
        config: ExperimentConfig,
    ) -> None:
        if BaseAgent is object:
            raise RuntimeError("agent-framework-core is required for Microsoft custom agents.")
        super().__init__(name=name, description="Benchmark custom OpenAI agent")
        self.instructions = instructions
        self.config = config

    def run(self, messages=None, *, stream: bool = False, session=None, **kwargs):
        if stream:
            raise NotImplementedError("Streaming is not used in the benchmark runner.")
        return self._run(messages)

    async def _run(self, messages=None):
        from openai import AsyncOpenAI

        normalized_messages = normalize_messages(messages)
        prompt = normalized_messages[-1].text if normalized_messages else ""
        full_input = f"{self.instructions}\n\n{prompt}".strip()
        client = AsyncOpenAI(api_key=self.config.metadata.get("openai_api_key"))
        request_kwargs = {
            "model": self.config.model_name,
            "input": full_input,
        }
        if self.config.max_tokens is not None:
            request_kwargs["max_output_tokens"] = self.config.max_tokens

        response = await client.responses.create(**request_kwargs)
        response_text = response.output_text.strip()
        message = Message(
            role="assistant",
            contents=[Content.from_text(response_text)],
        )
        return AgentResponse(
            messages=[message],
            agent_id=self.id,
            raw_representation=response,
        )


def run_async(coro):
    """Run an async framework call from the synchronous benchmark contract."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("Microsoft Agent Framework runners require a synchronous entrypoint.")


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def native_framework_available() -> bool:
    return find_spec("agent_framework") is not None and find_spec("agent_framework.openai") is not None


def uses_openai(config: ExperimentConfig) -> bool:
    return config.model_provider.lower() == "openai"


def build_openai_agent(*, name: str, instructions: str, config: ExperimentConfig):
    """Build a real Microsoft Agent Framework OpenAI agent."""

    if BaseAgent is object:
        raise RuntimeError(
            "agent-framework-core is required for Microsoft Agent Framework OpenAI runs."
        )
    return OpenAIBenchmarkMicrosoftAgent(
        name=name,
        instructions=instructions,
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
    """Execute a real Agent Framework agent and normalize benchmark metrics."""

    started = perf_counter()
    response = run_async(agent.run(prompt))
    latency_seconds = max(perf_counter() - started, 0.0)
    response_text = (getattr(response, "text", None) or str(response)).strip()
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
            "framework_api": "microsoft_agent_framework",
            "client": "custom_BaseAgent_openai_responses",
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


def build_agent(*, name: str, instructions: str, context: MicrosoftAgentFrameworkRunContext, input_data: ExperimentInput, config: ExperimentConfig):
    if uses_openai(config):
        return build_openai_agent(name=name, instructions=instructions, config=config)
    return DeterministicMicrosoftAgent(
        name=name,
        llm=context.llm,
        input_data=input_data,
        config=config,
    )



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

        result = build_experiment_result(
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
        save_result_json(result, base_dir=repo_root / "results" / "raw")
        return result

    return wrapper
