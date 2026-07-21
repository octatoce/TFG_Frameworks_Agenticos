"""Framework-neutral LLM helpers for benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from time import perf_counter

from benchmark_core.metrics import estimate_cost_usd
from benchmark_core.schemas import ExperimentConfig, ExperimentInput, LLMCallMetrics, TokenUsage
from benchmark_core.handoff_swarm import (
    AGENT_DISPLAY_NAMES,
    HANDOFF_AGENTS,
    build_deterministic_handoff_output,
)


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


def render_router_specialists_prompt(
    input_data: ExperimentInput,
    phase: str,
    state: dict[str, object],
) -> str:
    """Render a canonical ARCH_03 router/specialists prompt."""

    document_blocks = "\n".join(
        f"[{document.document_id}] {document.content}" for document in input_data.documents
    )
    if not document_blocks:
        document_blocks = "No documents provided."

    return (
        "You are executing ARCH_03_ROUTER_SPECIALISTS for a benchmark.\n"
        f"Router phase: {phase}\n"
        "Return only the output for this role. Keep it concise and traceable.\n\n"
        "For router_routing, use exactly this format:\n"
        "SELECTED_SPECIALISTS=comma_separated_specialist_ids\n"
        "SKIPPED_SPECIALISTS=comma_separated_specialist_ids_or_none\n"
        "RATIONALE=one short sentence\n\n"
        "Available specialists: data_specialist, reasoning_specialist, validation_specialist.\n\n"
        f"Task type: {input_data.task_type}\n"
        f"Question: {input_data.query}\n\n"
        f"Selected specialists:\n{state.get('selected_specialists') or 'None'}\n\n"
        f"Skipped specialists:\n{state.get('skipped_specialists') or 'None'}\n\n"
        f"Evidence:\n{state.get('evidence') or 'None'}\n\n"
        f"Preliminary decision:\n{state.get('preliminary_decision') or 'None'}\n\n"
        f"Validation report:\n{state.get('validation_report') or 'None'}\n\n"
        f"Documents:\n{document_blocks}\n"
    )


SUPERVISOR_WORKERS = ["data_worker", "reasoning_worker", "validation_worker", "synthesis_worker"]


def render_supervisor_workers_prompt(
    input_data: ExperimentInput,
    phase: str,
    state: dict[str, object],
    *,
    worker_name: str | None = None,
    task: str | None = None,
    revision_instructions: str | None = None,
) -> str:
    """Render a canonical ARCH_04 supervisor/workers prompt."""

    document_blocks = "\n".join(
        f"[{document.document_id}] {document.content}" for document in input_data.documents
    )
    if not document_blocks:
        document_blocks = "No documents provided."

    worker_outputs = state.get("worker_outputs") or []
    executed_workers = state.get("workers_executed") or []
    plan = state.get("plan") or {}
    plan_workers = []
    if isinstance(plan, dict):
        plan_workers = list(plan.get("workers_to_run") or [])

    return (
        "You are executing ARCH_04_SUPERVISOR_WORKERS for a benchmark.\n"
        f"Supervisor phase: {phase}\n"
        "The supervisor must actively plan, review worker outputs, request bounded revisions if needed, "
        "and decide when to finalize.\n\n"
        "For supervisor_plan, use exactly this format:\n"
        "WORKERS_TO_RUN=comma_separated_worker_ids\n"
        "TASK_ASSIGNMENT_<worker_id>=short task\n"
        "EXPECTED_OUTPUT_<worker_id>=short expected output\n"
        "QUALITY_CRITERIA=comma_separated_criteria\n\n"
        "For supervisor_decision, use exactly this format:\n"
        "ACCEPTED=true_or_false\n"
        "NEEDS_REVISION=true_or_false\n"
        "REVISION_INSTRUCTIONS=short instructions or none\n"
        "MISSING_INFORMATION=short note or none\n"
        "ACTION=run_worker|request_revision|finalize\n"
        "WORKER_NAME=worker_id_or_none\n"
        "TASK=short task or none\n"
        "STOP_REASON=short reason or none\n\n"
        "For worker phases, return only the requested worker output. "
        "For synthesis_worker and supervisor_finalize, include 'Final Answer:' when producing the final answer.\n\n"
        f"Available workers: {', '.join(SUPERVISOR_WORKERS)}.\n"
        f"Task type: {input_data.task_type}\n"
        f"Question: {input_data.query}\n\n"
        f"Current iteration: {state.get('iterations', 0)}\n"
        f"Max supervisor iterations: {state.get('max_supervisor_iterations', 3)}\n"
        f"Plan workers: {', '.join(plan_workers) if plan_workers else 'None'}\n"
        f"Executed workers: {', '.join(executed_workers) if executed_workers else 'None'}\n"
        f"Revisions requested: {state.get('revisions_requested', 0)}\n"
        f"Current worker: {worker_name or 'None'}\n"
        f"Current task: {task or 'None'}\n"
        f"Revision instructions: {revision_instructions or 'None'}\n\n"
        f"Worker outputs:\n{worker_outputs or 'None'}\n\n"
        f"Warnings:\n{state.get('warnings') or 'None'}\n\n"
        f"Documents:\n{document_blocks}\n"
    )


def choose_router_specialists(input_data: ExperimentInput) -> tuple[list[str], list[str], str]:
    """Choose ARCH_03 specialists for the deterministic local router."""

    available_specialists = ["data_specialist", "reasoning_specialist", "validation_specialist"]
    query_text = f"{input_data.query} {input_data.task_type}".lower()
    selected_specialists = ["reasoning_specialist"]

    if input_data.documents:
        selected_specialists.insert(0, "data_specialist")

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
        selected_specialists.append("validation_specialist")

    selected_specialists = [
        specialist for specialist in available_specialists if specialist in selected_specialists
    ]
    skipped_specialists = [
        specialist for specialist in available_specialists if specialist not in selected_specialists
    ]

    if not input_data.documents:
        rationale = "No documents were provided, so data retrieval is skipped."
    elif "validation_specialist" in selected_specialists:
        rationale = "The query benefits from evidence, reasoning, and validation."
    else:
        rationale = "The query can be answered with evidence and reasoning only."

    return selected_specialists, skipped_specialists, rationale


def choose_supervisor_workers(input_data: ExperimentInput) -> tuple[list[str], list[str], str]:
    """Choose ARCH_04 workers for the deterministic local supervisor."""

    query_text = f"{input_data.query} {input_data.task_type}".lower()
    workers_to_run = ["reasoning_worker", "synthesis_worker"]
    if input_data.documents:
        workers_to_run.insert(0, "data_worker")

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
        "revis",
        "contradic",
    ]
    if any(keyword in query_text for keyword in validation_keywords):
        workers_to_run.insert(-1, "validation_worker")

    workers_to_run = [worker for worker in SUPERVISOR_WORKERS if worker in workers_to_run]
    workers_not_used = [worker for worker in SUPERVISOR_WORKERS if worker not in workers_to_run]
    if "validation_worker" in workers_to_run:
        rationale = "The task benefits from evidence, reasoning, validation, and supervised synthesis."
    elif input_data.documents:
        rationale = "The task needs evidence extraction, reasoning, and synthesis."
    else:
        rationale = "No documents were provided, so the supervisor skips evidence extraction."
    return workers_to_run, workers_not_used, rationale


def parse_specialist_selection(
    router_output: str,
    available_specialists: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Parse and normalize router specialist selection output."""

    specialists = available_specialists or ["data_specialist", "reasoning_specialist", "validation_specialist"]

    def parse_line(name: str) -> list[str]:
        match = re.search(rf"^{name}\s*=\s*(.+)$", router_output, flags=re.MULTILINE)
        if not match:
            return []
        raw_items = re.split(r"[,;]", match.group(1).strip())
        return [
            item.strip()
            for item in raw_items
            if item.strip() and item.strip().lower() not in {"none", "null", "[]"}
        ]

    selected = [specialist for specialist in parse_line("SELECTED_SPECIALISTS") if specialist in specialists]
    skipped = [specialist for specialist in parse_line("SKIPPED_SPECIALISTS") if specialist in specialists]

    if not selected:
        selected = [specialist for specialist in specialists if specialist in router_output]

    if not selected:
        selected = list(specialists)

    selected = [specialist for specialist in specialists if specialist in selected]
    skipped = [specialist for specialist in specialists if specialist not in selected or specialist in skipped]
    skipped = [specialist for specialist in specialists if specialist in skipped and specialist not in selected]
    return selected, skipped


def parse_supervisor_plan(plan_output: str) -> dict[str, object]:
    """Parse and normalize an ARCH_04 supervisor plan."""

    def parse_line(name: str) -> str | None:
        match = re.search(rf"^{name}\s*=\s*(.+)$", plan_output, flags=re.MULTILINE)
        return match.group(1).strip() if match else None

    workers_line = parse_line("WORKERS_TO_RUN") or ""
    workers_to_run = [
        item.strip()
        for item in re.split(r"[,;]", workers_line)
        if item.strip() in SUPERVISOR_WORKERS
    ]
    if not workers_to_run:
        workers_to_run = [worker for worker in SUPERVISOR_WORKERS if worker in plan_output]
    if not workers_to_run:
        workers_to_run = ["reasoning_worker", "synthesis_worker"]

    task_assignments = {
        worker: parse_line(f"TASK_ASSIGNMENT_{worker}") or f"Complete the {worker} responsibility."
        for worker in workers_to_run
    }
    expected_outputs = {
        worker: parse_line(f"EXPECTED_OUTPUT_{worker}") or f"Concise {worker} output."
        for worker in workers_to_run
    }
    criteria_line = parse_line("QUALITY_CRITERIA") or "evidence, consistency, completeness, clarity"
    quality_criteria = [
        item.strip()
        for item in re.split(r"[,;]", criteria_line)
        if item.strip()
    ]
    return {
        "workers_to_run": workers_to_run,
        "task_assignments": task_assignments,
        "expected_outputs": expected_outputs,
        "quality_criteria": quality_criteria,
        "raw_plan": plan_output,
    }


def parse_supervisor_action(action_output: str) -> dict[str, object]:
    """Parse and normalize an ARCH_04 supervisor decision/review."""

    def parse_line(name: str) -> str | None:
        match = re.search(rf"^{name}\s*=\s*(.+)$", action_output, flags=re.MULTILINE)
        return match.group(1).strip() if match else None

    def parse_bool(value: str | None) -> bool:
        return (value or "").strip().lower() in {"true", "yes", "1", "accepted", "si", "sí"}

    action = (parse_line("ACTION") or "").lower()
    if action not in {"run_worker", "request_revision", "finalize"}:
        lowered = action_output.lower()
        if "request_revision" in lowered or "revision" in lowered:
            action = "request_revision"
        elif "finalize" in lowered or "final answer" in lowered:
            action = "finalize"
        else:
            action = "run_worker"

    worker_name = parse_line("WORKER_NAME") or None
    if worker_name not in SUPERVISOR_WORKERS:
        worker_name = None

    return {
        "accepted": parse_bool(parse_line("ACCEPTED")),
        "needs_revision": parse_bool(parse_line("NEEDS_REVISION")),
        "revision_instructions": parse_line("REVISION_INSTRUCTIONS") or "none",
        "missing_information": parse_line("MISSING_INFORMATION") or "none",
        "action": action,
        "worker_name": worker_name,
        "task": parse_line("TASK") or "none",
        "stop_reason": parse_line("STOP_REASON") or "none",
        "raw_decision": action_output,
    }


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


def build_deterministic_router_output(
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
    if normalized_phase == "router_routing":
        selected_specialists, skipped_specialists, rationale = choose_router_specialists(input_data)
        skipped_text = ", ".join(skipped_specialists) if skipped_specialists else "none"
        return (
            f"SELECTED_SPECIALISTS={', '.join(selected_specialists)}\n"
            f"SKIPPED_SPECIALISTS={skipped_text}\n"
            f"RATIONALE={rationale}"
        )
    if normalized_phase == "data_specialist":
        return f"Data report: sources={source_text}; evidence={evidence}"
    if normalized_phase == "reasoning_specialist":
        return (
            "Reasoning report: the case asks for the benchmark objective and validation criterion. "
            f"The evidence from {source_text} supports a direct answer grounded in the provided context."
        )
    if normalized_phase == "validation_specialist":
        return (
            "Validation report: the preliminary decision is consistent with the evidence; "
            "confidence=high; limitations=synthetic smoke case."
        )
    if normalized_phase == "router_synthesis":
        selected_specialists, _, _ = choose_router_specialists(input_data)
        validation_text = (
            " and validation"
            if "validation_specialist" in selected_specialists
            else ""
        )
        return (
            "Final Answer: "
            f"The router integrated the selected specialist reports ({', '.join(selected_specialists)})"
            f"{validation_text}. "
            "The document states that the TFG compares modern agentic frameworks through equivalent "
            "prototypes, and the validated criterion is whether common schemas, metric collection, "
            f"comparable execution, and raw JSON persistence work correctly. Sources used: {source_text}."
        )

    return build_deterministic_answer(input_data)


def _parse_prompt_line(prompt: str, prefix: str) -> str:
    for line in prompt.splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix).strip()
    return ""


def build_deterministic_supervisor_output(
    input_data: ExperimentInput,
    phase: str,
    prompt: str,
) -> str:
    """Build a stable local output for one ARCH_04 supervisor/worker phase."""

    document_ids = [document.document_id for document in input_data.documents]
    source_text = ", ".join(document_ids) if document_ids else "no-documents"
    if input_data.documents:
        evidence = input_data.documents[0].content.strip().replace("\n", " ")[:240]
    else:
        evidence = "No document context was provided."

    normalized_phase = phase.lower()
    if normalized_phase == "supervisor_plan":
        workers_to_run, _, rationale = choose_supervisor_workers(input_data)
        lines = [
            f"WORKERS_TO_RUN={', '.join(workers_to_run)}",
        ]
        assignments = {
            "data_worker": "Extract documentary evidence and source references.",
            "reasoning_worker": "Analyze the evidence and answer requirements.",
            "validation_worker": "Check contradictions, missing evidence, and risk.",
            "synthesis_worker": "Build the final structured answer from approved material.",
        }
        expected_outputs = {
            "data_worker": "Evidence report with source ids.",
            "reasoning_worker": "Technical reasoning report.",
            "validation_worker": "Validation report with limitations.",
            "synthesis_worker": "Final answer draft.",
        }
        for worker in workers_to_run:
            lines.append(f"TASK_ASSIGNMENT_{worker}={assignments[worker]}")
            lines.append(f"EXPECTED_OUTPUT_{worker}={expected_outputs[worker]}")
        lines.append("QUALITY_CRITERIA=evidence, consistency, completeness, clarity")
        lines.append(f"RATIONALE={rationale}")
        return "\n".join(lines)

    if normalized_phase == "supervisor_decision":
        plan_workers = [
            item.strip()
            for item in re.split(r"[,;]", _parse_prompt_line(prompt, "Plan workers:"))
            if item.strip() and item.strip().lower() != "none"
        ]
        executed_workers = [
            item.strip()
            for item in re.split(r"[,;]", _parse_prompt_line(prompt, "Executed workers:"))
            if item.strip() and item.strip().lower() != "none"
        ]
        current_iteration = int(_parse_prompt_line(prompt, "Current iteration:") or 0)
        max_iterations = int(_parse_prompt_line(prompt, "Max supervisor iterations:") or 3)
        accepted = "true" if executed_workers else "false"

        next_worker = next((worker for worker in plan_workers if worker not in executed_workers), None)
        if current_iteration >= max_iterations - 1 and next_worker != "synthesis_worker":
            next_worker = "synthesis_worker" if "synthesis_worker" in plan_workers and "synthesis_worker" not in executed_workers else None

        if next_worker is None:
            return (
                f"ACCEPTED={accepted}\n"
                "NEEDS_REVISION=false\n"
                "REVISION_INSTRUCTIONS=none\n"
                "MISSING_INFORMATION=none\n"
                "ACTION=finalize\n"
                "WORKER_NAME=none\n"
                "TASK=none\n"
                "STOP_REASON=quality_criteria_satisfied"
            )

        tasks = {
            "data_worker": "Extract evidence and cite source ids.",
            "reasoning_worker": "Reason over the approved evidence and query.",
            "validation_worker": "Validate the reasoning and identify limitations.",
            "synthesis_worker": "Synthesize the approved outputs into the final answer.",
        }
        missing = "none" if executed_workers else "initial worker output required"
        return (
            f"ACCEPTED={accepted}\n"
            "NEEDS_REVISION=false\n"
            "REVISION_INSTRUCTIONS=none\n"
            f"MISSING_INFORMATION={missing}\n"
            "ACTION=run_worker\n"
            f"WORKER_NAME={next_worker}\n"
            f"TASK={tasks[next_worker]}\n"
            "STOP_REASON=none"
        )

    if normalized_phase == "worker":
        worker_name = _parse_prompt_line(prompt, "Current worker:")
        if worker_name == "data_worker":
            return f"DataWorker output: sources={source_text}; evidence={evidence}"
        if worker_name == "reasoning_worker":
            return (
                "ReasoningWorker output: the task can be answered from the approved evidence. "
                f"The relevant sources are {source_text}."
            )
        if worker_name == "validation_worker":
            return (
                "ValidationWorker output: no contradictions detected; confidence=high; "
                "limitations=synthetic deterministic benchmark case."
            )
        if worker_name == "synthesis_worker":
            return (
                "Final Answer: The supervised workers produced evidence, reasoning, and any requested validation. "
                "The supervisor can approve a concise answer grounded in the provided documents. "
                f"Sources used: {source_text}."
            )
        return build_deterministic_answer(input_data)

    if normalized_phase == "supervisor_finalize":
        return (
            "Final Answer: The supervisor reviewed the available worker outputs and finalized the answer. "
            "The result is grounded in the executed workers' evidence, reasoning, and synthesis. "
            f"Sources used: {source_text}."
        )

    return build_deterministic_answer(input_data)


def detect_pipeline_phase(prompt: str) -> str | None:
    """Return the ARCH_02 phase embedded in a prompt, if present."""

    prefix = "Pipeline phase:"
    for line in prompt.splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix).strip().lower()
    return None


def detect_router_phase(prompt: str) -> str | None:
    """Return the ARCH_03 phase embedded in a prompt, if present."""

    prefix = "Router phase:"
    for line in prompt.splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix).strip().lower()
    return None


def detect_supervisor_phase(prompt: str) -> str | None:
    """Return the ARCH_04 supervisor/worker phase embedded in a prompt, if present."""

    prefix = "Supervisor phase:"
    for line in prompt.splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix).strip().lower()
    return None


def detect_handoff_agent(prompt: str) -> str | None:
    """Return the ARCH_05 active agent embedded in a prompt, if present."""

    if "ARCH_05_HANDOFF_SWARM" not in prompt:
        return None
    prefix = "Active agent:"
    for line in prompt.splitlines():
        if line.startswith(prefix):
            value = line.removeprefix(prefix).strip()
            lowered = value.lower()
            if lowered in HANDOFF_AGENTS:
                return lowered
            normalized = re.sub(r"[^a-z_]", "", lowered)
            if normalized in HANDOFF_AGENTS:
                return normalized
            for agent_name, display_name in AGENT_DISPLAY_NAMES.items():
                if normalized in {display_name.lower(), agent_name.replace("_", "")}:
                    return agent_name
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
        router_phase = detect_router_phase(prompt)
        supervisor_phase = detect_supervisor_phase(prompt)
        handoff_agent = detect_handoff_agent(prompt)
        if phase is not None:
            response = build_deterministic_pipeline_output(input_data, phase)
        elif router_phase is not None:
            response = build_deterministic_router_output(input_data, router_phase)
        elif supervisor_phase is not None:
            response = build_deterministic_supervisor_output(input_data, supervisor_phase, prompt)
        elif handoff_agent is not None:
            response = build_deterministic_handoff_output(input_data, handoff_agent, prompt)
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
