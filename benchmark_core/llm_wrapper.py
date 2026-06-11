"""Framework-neutral LLM helpers for benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from time import perf_counter

from benchmark_core.metrics import estimate_cost_usd
from benchmark_core.schemas import ExperimentConfig, ExperimentInput, LLMCallMetrics, TokenUsage


@dataclass
class LLMCallRecord:
    """Record returned by one instrumented LLM call."""

    model_name: str
    prompt: str
    response: str
    metrics: LLMCallMetrics


def render_single_react_prompt(input_data: ExperimentInput) -> str:
    """Render the canonical ARCH_01 task prompt used by both frameworks."""

    document_blocks = "\n".join(
        f"[{document.document_id}] {document.content}" for document in input_data.documents
    )
    if not document_blocks:
        document_blocks = "No documents provided."

    return (
        "You are executing ARCH_01_SINGLE_REACT for a benchmark.\n"
        "Use one ReAct-style reasoning step internally and provide a concise final answer.\n\n"
        f"Task type: {input_data.task_type}\n"
        f"Question: {input_data.query}\n\n"
        f"Documents:\n{document_blocks}\n"
    )


def render_sequential_pipeline_prompt(
    input_data: ExperimentInput,
    phase: str,
    state: dict[str, object],
) -> str:
    """Render a canonical ARCH_02 phase prompt."""

    document_blocks = "\n".join(
        f"[{document.document_id}] {document.content}" for document in input_data.documents
    )
    if not document_blocks:
        document_blocks = "No documents provided."

    return (
        "You are executing ARCH_02_SEQUENTIAL_PIPELINE for a benchmark.\n"
        f"Pipeline phase: {phase}\n"
        "Return only the output for this phase. Keep it concise and traceable.\n\n"
        f"Task type: {input_data.task_type}\n"
        f"Question: {input_data.query}\n\n"
        f"Current plan:\n{state.get('plan') or 'None'}\n\n"
        f"Current evidence:\n{state.get('evidence') or 'None'}\n\n"
        f"Current analysis:\n{state.get('analysis') or 'None'}\n\n"
        f"Documents:\n{document_blocks}\n"
    )


def render_supervisor_workers_prompt(
    input_data: ExperimentInput,
    phase: str,
    state: dict[str, object],
) -> str:
    """Render a canonical ARCH_03 supervisor/workers prompt."""

    document_blocks = "\n".join(
        f"[{document.document_id}] {document.content}" for document in input_data.documents
    )
    if not document_blocks:
        document_blocks = "No documents provided."

    return (
        "You are executing ARCH_03_SUPERVISOR_WORKERS for a benchmark.\n"
        f"Supervisor phase: {phase}\n"
        "Return only the output for this role. Keep it concise and traceable.\n\n"
        "For supervisor_planning, use exactly this format:\n"
        "SELECTED_WORKERS=comma_separated_worker_ids\n"
        "SKIPPED_WORKERS=comma_separated_worker_ids_or_none\n"
        "RATIONALE=one short sentence\n\n"
        "Available workers: data_worker, reasoning_worker, validation_worker.\n\n"
        f"Task type: {input_data.task_type}\n"
        f"Question: {input_data.query}\n\n"
        f"Selected workers:\n{state.get('selected_workers') or 'None'}\n\n"
        f"Skipped workers:\n{state.get('skipped_workers') or 'None'}\n\n"
        f"Evidence:\n{state.get('evidence') or 'None'}\n\n"
        f"Preliminary decision:\n{state.get('preliminary_decision') or 'None'}\n\n"
        f"Validation report:\n{state.get('validation_report') or 'None'}\n\n"
        f"Documents:\n{document_blocks}\n"
    )


def choose_supervisor_workers(input_data: ExperimentInput) -> tuple[list[str], list[str], str]:
    """Choose ARCH_03 workers for the deterministic local supervisor."""

    available_workers = ["data_worker", "reasoning_worker", "validation_worker"]
    query_text = f"{input_data.query} {input_data.task_type}".lower()
    selected_workers = ["reasoning_worker"]

    if input_data.documents:
        selected_workers.insert(0, "data_worker")

    validation_keywords = [
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
    ]
    if any(keyword in query_text for keyword in validation_keywords):
        selected_workers.append("validation_worker")

    selected_workers = [
        worker for worker in available_workers if worker in selected_workers
    ]
    skipped_workers = [
        worker for worker in available_workers if worker not in selected_workers
    ]

    if not input_data.documents:
        rationale = "No documents were provided, so data retrieval is skipped."
    elif "validation_worker" in selected_workers:
        rationale = "The query benefits from evidence, reasoning, and validation."
    else:
        rationale = "The query can be answered with evidence and reasoning only."

    return selected_workers, skipped_workers, rationale


def parse_worker_selection(
    supervisor_output: str,
    available_workers: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Parse and normalize supervisor worker selection output."""

    workers = available_workers or ["data_worker", "reasoning_worker", "validation_worker"]

    def parse_line(name: str) -> list[str]:
        match = re.search(rf"^{name}\s*=\s*(.+)$", supervisor_output, flags=re.MULTILINE)
        if not match:
            return []
        raw_items = re.split(r"[,;]", match.group(1).strip())
        return [
            item.strip()
            for item in raw_items
            if item.strip() and item.strip().lower() not in {"none", "null", "[]"}
        ]

    selected = [worker for worker in parse_line("SELECTED_WORKERS") if worker in workers]
    skipped = [worker for worker in parse_line("SKIPPED_WORKERS") if worker in workers]

    if not selected:
        selected = [worker for worker in workers if worker in supervisor_output]

    if not selected:
        selected = list(workers)

    selected = [worker for worker in workers if worker in selected]
    skipped = [worker for worker in workers if worker not in selected or worker in skipped]
    skipped = [worker for worker in workers if worker in skipped and worker not in selected]
    return selected, skipped


def build_deterministic_answer(input_data: ExperimentInput) -> str:
    """Build a stable local answer for smoke tests without external APIs."""

    document_ids = [document.document_id for document in input_data.documents]
    if input_data.documents:
        first_document = input_data.documents[0].content.strip().replace("\n", " ")
        evidence = first_document[:240]
    else:
        evidence = "No document context was provided."

    source_text = ", ".join(document_ids) if document_ids else "no-documents"
    return (
        f"Deterministic benchmark answer for case {input_data.case_id}. "
        f"Query: {input_data.query.strip()} "
        f"Sources used: {source_text}. "
        f"Main evidence: {evidence}"
    )


def build_deterministic_pipeline_output(
    input_data: ExperimentInput,
    phase: str,
) -> str:
    """Build a stable local output for one ARCH_02 phase."""

    document_ids = [document.document_id for document in input_data.documents]
    source_text = ", ".join(document_ids) if document_ids else "no-documents"
    if input_data.documents:
        evidence = input_data.documents[0].content.strip().replace("\n", " ")[:240]
    else:
        evidence = "No document context was provided."

    normalized_phase = phase.lower()
    if normalized_phase == "planner":
        return (
            "Plan: identify the requested objective, inspect the provided documents, "
            "extract relevant evidence, and produce a concise structured answer."
        )
    if normalized_phase == "retriever":
        return f"Evidence: sources={source_text}; relevant_excerpt={evidence}"
    if normalized_phase == "analyst":
        return (
            "Analysis: the case asks for the benchmark objective and validation criterion. "
            f"The available evidence from {source_text} supports answering from the provided context."
        )
    if normalized_phase == "writer":
        return (
            "Final Answer: "
            f"The document states that the TFG compares modern agentic frameworks through equivalent "
            f"prototypes. The validated criterion is whether the repository structure, common schemas, "
            f"metric collection, comparable execution, and raw JSON persistence work correctly. "
            f"Sources used: {source_text}."
        )

    return build_deterministic_answer(input_data)


def build_deterministic_supervisor_output(
    input_data: ExperimentInput,
    phase: str,
) -> str:
    """Build a stable local output for one ARCH_03 role."""

    document_ids = [document.document_id for document in input_data.documents]
    source_text = ", ".join(document_ids) if document_ids else "no-documents"
    if input_data.documents:
        evidence = input_data.documents[0].content.strip().replace("\n", " ")[:240]
    else:
        evidence = "No document context was provided."

    normalized_phase = phase.lower()
    if normalized_phase == "supervisor_planning":
        selected_workers, skipped_workers, rationale = choose_supervisor_workers(input_data)
        skipped_text = ", ".join(skipped_workers) if skipped_workers else "none"
        return (
            f"SELECTED_WORKERS={', '.join(selected_workers)}\n"
            f"SKIPPED_WORKERS={skipped_text}\n"
            f"RATIONALE={rationale}"
        )
    if normalized_phase == "data_worker":
        return f"Data report: sources={source_text}; evidence={evidence}"
    if normalized_phase == "reasoning_worker":
        return (
            "Reasoning report: the case asks for the benchmark objective and validation criterion. "
            f"The evidence from {source_text} supports a direct answer grounded in the provided context."
        )
    if normalized_phase == "validation_worker":
        return (
            "Validation report: the preliminary decision is consistent with the evidence; "
            "confidence=high; limitations=synthetic smoke case."
        )
    if normalized_phase == "supervisor_synthesis":
        selected_workers, _, _ = choose_supervisor_workers(input_data)
        validation_text = (
            " and validation"
            if "validation_worker" in selected_workers
            else ""
        )
        return (
            "Final Answer: "
            f"The supervisor integrated the selected worker reports ({', '.join(selected_workers)})"
            f"{validation_text}. "
            "The document states that the TFG compares modern agentic frameworks through equivalent "
            "prototypes, and the validated criterion is whether common schemas, metric collection, "
            f"comparable execution, and raw JSON persistence work correctly. Sources used: {source_text}."
        )

    return build_deterministic_answer(input_data)


def detect_pipeline_phase(prompt: str) -> str | None:
    """Return the ARCH_02 phase embedded in a prompt, if present."""

    prefix = "Pipeline phase:"
    for line in prompt.splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix).strip().lower()
    return None


def detect_supervisor_phase(prompt: str) -> str | None:
    """Return the ARCH_03 phase embedded in a prompt, if present."""

    prefix = "Supervisor phase:"
    for line in prompt.splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix).strip().lower()
    return None


def estimate_token_usage(text: str) -> int:
    """Estimate tokens with a simple reproducible whitespace proxy."""

    return len([part for part in text.replace("\n", " ").split(" ") if part])


class InstrumentedLLM:
    """Deterministic local LLM used to validate benchmark instrumentation."""

    def __init__(
        self,
        model_provider: str,
        model_name: str,
        input_cost_per_1k: float = 0.0,
        output_cost_per_1k: float = 0.0,
    ) -> None:
        self.model_provider = model_provider
        self.model_name = model_name
        self.input_cost_per_1k = input_cost_per_1k
        self.output_cost_per_1k = output_cost_per_1k

    def complete(
        self,
        prompt: str,
        input_data: ExperimentInput,
        call_id: str,
        step_id: int | None = None,
    ) -> LLMCallRecord:
        """Return a deterministic ReAct-shaped response and call metrics."""

        started = perf_counter()
        phase = detect_pipeline_phase(prompt)
        supervisor_phase = detect_supervisor_phase(prompt)
        if phase is not None:
            response = build_deterministic_pipeline_output(input_data, phase)
        elif supervisor_phase is not None:
            response = build_deterministic_supervisor_output(input_data, supervisor_phase)
        else:
            final_answer = build_deterministic_answer(input_data)
            response = f"Thought: I can answer from the provided benchmark input.\nFinal Answer: {final_answer}"
        latency_seconds = max(perf_counter() - started, 0.0)
        token_usage = TokenUsage(
            input_tokens=estimate_token_usage(prompt),
            output_tokens=estimate_token_usage(response),
            total_tokens=estimate_token_usage(prompt) + estimate_token_usage(response),
        )
        metrics = LLMCallMetrics(
            call_id=call_id,
            step_id=step_id,
            model_provider=self.model_provider,
            model_name=self.model_name,
            latency_seconds=latency_seconds,
            token_usage=token_usage,
            estimated_cost_usd=estimate_cost_usd(
                token_usage,
                input_cost_per_1k=self.input_cost_per_1k,
                output_cost_per_1k=self.output_cost_per_1k,
            ),
            finish_reason="stop",
            metadata={
                "deterministic": True,
                "token_counting_method": "whitespace_proxy",
            },
        )
        return LLMCallRecord(
            model_name=self.model_name,
            prompt=prompt,
            response=response,
            metrics=metrics,
        )


class OpenAIInstrumentedLLM(InstrumentedLLM):
    """OpenAI Responses API-backed LLM with benchmark metric extraction."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
        input_cost_per_1k: float = 0.0,
        output_cost_per_1k: float = 0.0,
        env_file: str | Path | None = ".env",
    ) -> None:
        super().__init__(
            model_provider="openai",
            model_name=model_name,
            input_cost_per_1k=input_cost_per_1k,
            output_cost_per_1k=output_cost_per_1k,
        )
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.env_file = Path(env_file) if env_file is not None else None

    def complete(
        self,
        prompt: str,
        input_data: ExperimentInput,
        call_id: str,
        step_id: int | None = None,
    ) -> LLMCallRecord:
        """Call OpenAI and return normalized benchmark metrics."""

        self._load_env_file()

        from openai import OpenAI

        request_kwargs = {
            "model": self.model_name,
            "input": prompt,
            "temperature": self.temperature,
        }
        if self.max_output_tokens is not None:
            request_kwargs["max_output_tokens"] = self.max_output_tokens

        client = OpenAI()
        started = perf_counter()
        response = client.responses.create(**request_kwargs)
        latency_seconds = max(perf_counter() - started, 0.0)

        response_text = response.output_text.strip()
        token_usage = self._extract_token_usage(response)
        metrics = LLMCallMetrics(
            call_id=call_id,
            step_id=step_id,
            model_provider=self.model_provider,
            model_name=self.model_name,
            latency_seconds=latency_seconds,
            token_usage=token_usage,
            estimated_cost_usd=estimate_cost_usd(
                token_usage,
                input_cost_per_1k=self.input_cost_per_1k,
                output_cost_per_1k=self.output_cost_per_1k,
            ),
            finish_reason=self._extract_finish_reason(response),
            metadata={
                "deterministic": False,
                "api": "openai.responses",
                "response_id": getattr(response, "id", None),
                "token_counting_method": "openai_usage",
            },
        )
        return LLMCallRecord(
            model_name=self.model_name,
            prompt=prompt,
            response=response_text,
            metrics=metrics,
        )

    def _load_env_file(self) -> None:
        if self.env_file is None or not self.env_file.exists():
            return

        try:
            from dotenv import load_dotenv
        except ImportError:
            return

        load_dotenv(self.env_file)

    def _extract_token_usage(self, response) -> TokenUsage:
        usage = getattr(response, "usage", None)
        if usage is None:
            return TokenUsage()

        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", input_tokens + output_tokens) or 0)
        input_details = getattr(usage, "input_tokens_details", None)
        output_details = getattr(usage, "output_tokens_details", None)
        cached_input_tokens = int(getattr(input_details, "cached_tokens", 0) or 0)
        reasoning_tokens = int(getattr(output_details, "reasoning_tokens", 0) or 0)
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=total_tokens,
        )

    def _extract_finish_reason(self, response) -> str | None:
        output = getattr(response, "output", None)
        if not output:
            return None

        finish_reasons = []
        for item in output:
            finish_reason = getattr(item, "finish_reason", None)
            if finish_reason:
                finish_reasons.append(finish_reason)
        return ",".join(finish_reasons) if finish_reasons else None


def build_llm_from_config(config: ExperimentConfig) -> InstrumentedLLM:
    """Build the configured instrumented LLM implementation."""

    input_cost = float(config.metadata.get("input_cost_per_1k_tokens", 0.0))
    output_cost = float(config.metadata.get("output_cost_per_1k_tokens", 0.0))
    provider = config.model_provider.lower()

    if provider == "openai":
        return OpenAIInstrumentedLLM(
            model_name=config.model_name,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
            input_cost_per_1k=input_cost,
            output_cost_per_1k=output_cost,
            env_file=config.metadata.get("env_file", ".env"),
        )

    return InstrumentedLLM(
        model_provider=config.model_provider,
        model_name=config.model_name,
        input_cost_per_1k=input_cost,
        output_cost_per_1k=output_cost,
    )
