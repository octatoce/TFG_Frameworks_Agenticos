"""Shared utilities for CrewAI benchmark implementations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import PrivateAttr

from benchmark_core.llm_wrapper import InstrumentedLLM, build_llm_from_config
from benchmark_core.llm_wrapper import LLMCallRecord, estimate_token_usage
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


@dataclass
class CrewAIRunContext:
    repo_root: Path
    crewai_llm: Any


@dataclass
class CrewAIRunOutput:
    final_answer: str
    structured_output: dict[str, Any]
    steps: list[AgentStep]
    resource_usage: ResourceUsage | None = None


CrewAIImplementation = Callable[
    [ExperimentInput, ExperimentConfig, CrewAIRunContext],
    CrewAIRunOutput,
]


def get_repo_root() -> Path:
    """Return the repository root from this shared CrewAI module."""

    return Path(__file__).resolve().parents[2]


def prepare_crewai_runtime(repo_root: Path) -> None:
    """Keep CrewAI side-effect storage inside the benchmark repository."""

    local_data_dir = repo_root / ".crewai_data"
    os.environ["LOCALAPPDATA"] = str(local_data_dir / "localappdata")
    os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
    os.environ["CREWAI_DISABLE_TRACKING"] = "true"
    os.environ["CREWAI_DISABLE_VERSION_CHECK"] = "true"
    os.environ["CREWAI_TESTING"] = "true"
    os.environ["OTEL_SDK_DISABLED"] = "true"

    import appdirs

    appdirs.user_data_dir = lambda appname=None, appauthor=None, version=None, roaming=False: str(
        local_data_dir / "appdirs" / (appauthor or "app") / (appname or "app")
    )


def messages_to_text(messages: Any) -> str:
    if isinstance(messages, str):
        return messages
    if isinstance(messages, list):
        parts = []
        for message in messages:
            if isinstance(message, dict):
                parts.append(f"{message.get('role', 'unknown')}: {message.get('content', '')}")
            else:
                parts.append(str(message))
        return "\n".join(parts)
    return str(messages)


def extract_final_answer(response: str) -> str:
    marker = "Final Answer:"
    if marker not in response:
        return response.strip()
    return response.split(marker, maxsplit=1)[1].strip()


def create_benchmark_crewai_llm(
    input_data: ExperimentInput,
    config: ExperimentConfig,
) -> Any:
    """Build the CrewAI LLM adapter around the benchmark LLM wrapper."""

    from crewai.llms.base_llm import BaseLLM

    class BenchmarkCrewAILLM(BaseLLM):
        _call_records: list = PrivateAttr(default_factory=list)
        _config: ExperimentConfig = PrivateAttr()
        _input_data: ExperimentInput = PrivateAttr()
        _instrumented_llm: InstrumentedLLM = PrivateAttr()

        def __init__(
            self,
            input_data: ExperimentInput,
            config: ExperimentConfig,
            instrumented_llm: InstrumentedLLM,
        ) -> None:
            super().__init__(
                model=instrumented_llm.model_name,
                provider=instrumented_llm.model_provider,
                temperature=config.temperature,
            )
            self._config = config
            self._input_data = input_data
            self._instrumented_llm = instrumented_llm

        @property
        def call_records(self) -> list:
            return self._call_records

        def call(
            self,
            messages,
            tools=None,
            callbacks=None,
            available_functions=None,
            from_task=None,
            from_agent=None,
            response_model=None,
        ) -> str:
            prompt = messages_to_text(messages)
            call_number = len(self._call_records) + 1
            deterministic_response = build_crewai_native_hierarchical_response(
                prompt=prompt,
                input_data=self._input_data,
                config=self._config,
                from_agent=from_agent,
                tools=tools,
            )
            if deterministic_response is not None:
                token_usage = TokenUsage(
                    input_tokens=estimate_token_usage(prompt),
                    output_tokens=estimate_token_usage(deterministic_response),
                    total_tokens=estimate_token_usage(prompt) + estimate_token_usage(deterministic_response),
                )
                call_record = LLMCallRecord(
                    model_name=self._instrumented_llm.model_name,
                    prompt=prompt,
                    response=deterministic_response,
                    metrics=LLMCallMetrics(
                        call_id=f"{self._config.run_id}-llm-{call_number:03d}",
                        step_id=call_number,
                        model_provider=self._instrumented_llm.model_provider,
                        model_name=self._instrumented_llm.model_name,
                        latency_seconds=0.0,
                        token_usage=token_usage,
                        estimated_cost_usd=estimate_cost_usd(
                            token_usage,
                            input_cost_per_1k=self._instrumented_llm.input_cost_per_1k,
                            output_cost_per_1k=self._instrumented_llm.output_cost_per_1k,
                        ),
                        finish_reason="stop",
                        metadata={
                            "deterministic": True,
                            "token_counting_method": "whitespace_proxy",
                            "framework_api": "crewai_native_hierarchical",
                            "crewai_from_agent_role": getattr(from_agent, "role", None),
                        },
                    ),
                )
                self._call_records.append(call_record)
                return call_record.response

            call_record = self._instrumented_llm.complete(
                prompt=prompt,
                input_data=self._input_data,
                call_id=f"{self._config.run_id}-llm-{call_number:03d}",
                step_id=call_number,
            )
            self._call_records.append(call_record)
            return call_record.response

    return BenchmarkCrewAILLM(
        input_data=input_data,
        config=config,
        instrumented_llm=build_llm_from_config(config),
    )


def build_crewai_native_hierarchical_response(
    *,
    prompt: str,
    input_data: ExperimentInput,
    config: ExperimentConfig,
    from_agent: Any = None,
    tools: Any = None,
) -> str | None:
    """Return local deterministic responses for CrewAI's native hierarchical manager."""

    if config.model_provider.lower() == "openai" or "ARCH_04_SUPERVISOR_WORKERS" not in prompt:
        return None

    role = getattr(from_agent, "role", "") or ""
    if not role:
        for candidate_role in ["Supervisor", "DataWorker", "ReasoningWorker", "ValidationWorker", "SynthesisWorker"]:
            if f"You are {candidate_role}." in prompt:
                role = candidate_role
                break
    document_ids_text = ", ".join(document_ids(input_data)) if input_data.documents else "no-documents"
    evidence = (
        input_data.documents[0].content.strip().replace("\n", " ")[:240]
        if input_data.documents
        else "No document context was provided."
    )

    if role == "DataWorker":
        return f"DataWorker output: sources={document_ids_text}; evidence={evidence}"
    if role == "ReasoningWorker":
        return (
            "ReasoningWorker output: the task can be answered from the approved evidence. "
            f"The relevant sources are {document_ids_text}."
        )
    if role == "ValidationWorker":
        return (
            "ValidationWorker output: no contradictions detected; confidence=high; "
            "limitations=synthetic deterministic benchmark case."
        )
    if role == "SynthesisWorker":
        return (
            "SynthesisWorker output: Final Answer: The native CrewAI hierarchical manager delegated "
            "work to the available workers and reviewed their responses. "
            f"Sources used: {document_ids_text}."
        )

    has_delegation_tools = (
        tools is not None
        or "Delegate work to coworker" in prompt
        or "delegate_work_to_coworker" in prompt
    )
    if role == "Supervisor" and has_delegation_tools:
        query_text = f"{input_data.query} {input_data.task_type}".lower()
        needs_validation = any(
            keyword in query_text
            for keyword in [
                "valid",
                "confidence",
                "confianza",
                "risk",
                "riesgo",
                "error",
                "crit",
                "compar",
                "evaluate",
                "evaluacion",
                "evaluar",
                "revis",
                "contradic",
            ]
        )

        def delegate(coworker: str, task: str) -> str:
            return (
                f"Thought: I should delegate this part to {coworker} and then validate the outcome.\n"
                "Action: delegate_work_to_coworker\n"
                f"Action Input: {{\"coworker\": \"{coworker}\", \"task\": \"{task}\", "
                f"\"context\": \"ARCH_04_SUPERVISOR_WORKERS case {input_data.case_id}. "
                f"Question: {input_data.query}. Sources: {document_ids_text}.\"}}"
            )

        if (
            "MUST give your absolute best final answer" in prompt
            or '"coworker": "SynthesisWorker"' in prompt
            or "SynthesisWorker output:" in prompt
        ):
            return (
                "Thought: I now know the final answer\n"
                "Final Answer: The native CrewAI hierarchical manager coordinated the workers, "
                "validated the available outcomes, and approved the final answer. "
                f"Sources used: {document_ids_text}."
            )
        if input_data.documents and "DataWorker output:" not in prompt:
            return delegate("DataWorker", "Extract evidence and cite source ids.")
        if "ReasoningWorker output:" not in prompt:
            return delegate("ReasoningWorker", "Reason over the approved evidence and query.")
        if needs_validation and "ValidationWorker output:" not in prompt:
            return delegate("ValidationWorker", "Validate the reasoning and identify limitations.")
        if "SynthesisWorker output:" not in prompt:
            return delegate("SynthesisWorker", "Synthesize the approved outputs into the final answer.")
        return (
            "Thought: I now know the final answer\n"
            "Final Answer: The native CrewAI hierarchical manager coordinated the workers, "
            "validated the available outcomes, and approved the final answer. "
            f"Sources used: {document_ids_text}."
        )

    return None


def get_llm_call_metrics(crewai_llm: Any) -> list[LLMCallMetrics]:
    return [record.metrics for record in crewai_llm.call_records]


def create_agent(
    *,
    role: str,
    goal: str,
    backstory: str,
    crewai_llm: Any,
    config: ExperimentConfig,
    allow_delegation: bool = False,
    max_iter: int | None = None,
):
    from crewai import Agent

    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=crewai_llm,
        verbose=False,
        max_iter=max_iter or config.max_agent_iterations,
        allow_delegation=allow_delegation,
        max_retry_limit=config.retry_count,
        memory=False,
    )


def create_task(
    *,
    description: str,
    expected_output: str,
    agent: Any | None,
    config: ExperimentConfig,
    context: list[Any] | None = None,
):
    from crewai import Task

    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
        context=context,
        guardrail_max_retries=config.retry_count,
    )


def create_sequential_crew(*, agents: list[Any], tasks: list[Any]):
    from crewai import Crew, Process

    return Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
        memory=False,
        cache=False,
        tracing=False,
    )


def create_hierarchical_crew(*, agents: list[Any], tasks: list[Any], manager_agent: Any):
    from crewai import Crew, Process

    return Crew(
        agents=agents,
        tasks=tasks,
        manager_agent=manager_agent,
        process=Process.hierarchical,
        verbose=False,
        memory=False,
        cache=False,
        planning=False,
        tracing=False,
        checkpoint=False,
    )


def kickoff_with_resource_monitor(crew: Any) -> tuple[Any, ResourceUsage]:
    with ResourceMonitor() as monitor:
        crew_output = crew.kickoff()
        resource_usage = monitor.usage
    return crew_output, resource_usage


def crew_process_value() -> str:
    from crewai import Process

    return Process.sequential.value


def unique_agents(agents: list[Any]) -> list[Any]:
    return list({id(agent): agent for agent in agents}.values())


def document_ids(input_data: ExperimentInput) -> list[str]:
    return [document.document_id for document in input_data.documents]


def crewai_architecture_runner(run_impl: CrewAIImplementation):
    def wrapper(
        input_data: ExperimentInput,
        config: ExperimentConfig,
    ) -> ExperimentResult:
        repo_root = get_repo_root()
        started_at = utc_now()
        resource_usage = None

        try:
            prepare_crewai_runtime(repo_root)
            context = CrewAIRunContext(
                repo_root=repo_root,
                crewai_llm=create_benchmark_crewai_llm(input_data, config),
            )
            output = run_impl(input_data, config, context)
            status = RunStatus.SUCCESS
            final_answer = output.final_answer
            structured_output = output.structured_output
            steps = output.steps
            llm_calls = get_llm_call_metrics(context.crewai_llm)
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
            environment_packages=["crewai"],
            repo_root=repo_root,
        )
        save_result_json(result, base_dir=repo_root / "results" / "raw")
        return result

    return wrapper
