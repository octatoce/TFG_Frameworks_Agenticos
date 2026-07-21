"""Shared semantics and trace helpers for ARCH_07_MAP_REDUCE_AGENTIC.

Framework orchestration stays in each implementation. This module only keeps
partitioning, prompts, normalized outputs, traces, and metrics comparable.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from benchmark_core.schemas import (
    AgentStep,
    DocumentInput,
    ExperimentConfig,
    ExperimentInput,
    LLMCallMetrics,
)


ARCHITECTURE = "ARCH_07_MAP_REDUCE_AGENTIC"
DEFAULT_BATCH_SIZE = 3
PARTITIONER = "document_partitioner"
MAPPER = "mapper"
REDUCER = "reducer"


class DocumentBatch(BaseModel):
    batch_id: str
    batch_index: int
    documents: list[DocumentInput]

    @property
    def document_ids(self) -> list[str]:
        return [document.document_id for document in self.documents]


class MapperAnalysis(BaseModel):
    batch_id: str
    document_ids: list[str]
    partial_answer: str
    findings: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    raw_output: str
    error: str | None = None


class ReducedAnalysis(BaseModel):
    answer: str
    resolved_contradictions: list[str] = Field(default_factory=list)
    prioritized_evidence: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    raw_output: str


def get_batch_size(config: ExperimentConfig) -> int:
    configured = config.metadata.get(
        "map_reduce_batch_size",
        config.metadata.get("batch_size", DEFAULT_BATCH_SIZE),
    )
    try:
        batch_size = int(configured)
    except (TypeError, ValueError) as exc:
        raise ValueError("map_reduce_batch_size must be a positive integer") from exc
    if batch_size <= 0:
        raise ValueError("map_reduce_batch_size must be a positive integer")
    return batch_size


def partition_documents(
    input_data: ExperimentInput,
    config: ExperimentConfig,
) -> list[DocumentBatch]:
    """Create stable contiguous batches while preserving document order."""

    batch_size = get_batch_size(config)
    documents = input_data.documents
    if not documents:
        return [DocumentBatch(batch_id="batch_001", batch_index=0, documents=[])]
    return [
        DocumentBatch(
            batch_id=f"batch_{start // batch_size + 1:03d}",
            batch_index=start // batch_size,
            documents=documents[start : start + batch_size],
        )
        for start in range(0, len(documents), batch_size)
    ]


def mapper_name(batch: DocumentBatch) -> str:
    return f"mapper_{batch.batch_index + 1:03d}"


def render_map_reduce_prompt(
    input_data: ExperimentInput,
    component: str,
    *,
    batch: DocumentBatch | None = None,
    partial_outputs: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Render one canonical mapper or reducer prompt."""

    if component not in {MAPPER, REDUCER}:
        raise ValueError(f"Unknown ARCH_07 component: {component}")
    common = (
        f"You are executing {ARCHITECTURE} for a benchmark.\n"
        f"Map-reduce component: {component}\n"
        f"Task type: {input_data.task_type}\n"
        f"Question: {input_data.query}\n"
    )
    if component == MAPPER:
        if batch is None:
            raise ValueError("Mapper prompt requires one document batch")
        document_blocks = "\n".join(
            f"[{document.document_id}] {document.content}" for document in batch.documents
        ) or "No documents in this batch."
        return common + (
            f"Batch id: {batch.batch_id}\n"
            f"Batch index: {batch.batch_index}\n"
            f"Batch document ids: {json.dumps(batch.document_ids, ensure_ascii=False)}\n\n"
            "Process only this batch. Use the same mapper logic used for every other batch. "
            "Extract relevant evidence, findings, a bounded partial answer, and limitations. "
            "Do not access other batches, aggregate globally, route, hand off, debate, or supervise.\n\n"
            f"Batch documents:\n{document_blocks}\n\n"
            "Return exactly these labeled fields:\n"
            "PARTIAL_ANSWER=bounded answer from this batch only\n"
            "FINDINGS=item 1 | item 2 | none\n"
            "EVIDENCE=item 1 | item 2 | none\n"
            "LIMITATIONS=item 1 | item 2 | none\n"
        )

    if partial_outputs is None:
        raise ValueError("Reducer prompt requires mapper partial outputs")
    partial_json = json.dumps(partial_outputs, ensure_ascii=False, sort_keys=True)
    all_document_ids = [document.document_id for document in input_data.documents]
    return common + (
        f"Original document ids: {json.dumps(all_document_ids, ensure_ascii=False)}\n\n"
        "You are the single reducer. Consume all mapper outputs exactly once, deduplicate findings, "
        "resolve contradictions, prioritize relevant evidence, and synthesize the final answer. "
        "Do not reread the complete original documents, start another round, route, hand off, or supervise.\n\n"
        "Return exactly these labeled fields:\n"
        "FINAL_ANSWER=concise global answer\n"
        "RESOLVED_CONTRADICTIONS=item 1 | item 2 | none\n"
        "PRIORITIZED_EVIDENCE=item 1 | item 2 | none\n"
        "LIMITATIONS=item 1 | item 2 | none\n\n"
        f"Mapper outputs JSON:\n{partial_json}\n"
    )


def detect_map_reduce_component(prompt: str) -> str | None:
    if ARCHITECTURE not in prompt:
        return None
    prefix = "Map-reduce component:"
    for line in prompt.splitlines():
        if line.startswith(prefix):
            component = line.removeprefix(prefix).strip().lower()
            return component if component in {MAPPER, REDUCER} else None
    return None


def detect_map_reduce_batch_id(prompt: str) -> str | None:
    if detect_map_reduce_component(prompt) != MAPPER:
        return None
    prefix = "Batch id:"
    for line in prompt.splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix).strip()
    return None


def _labeled_value(output: str, label: str) -> str | None:
    prefix = f"{label}="
    for line in output.splitlines():
        if line.strip().startswith(prefix):
            return line.strip().removeprefix(prefix).strip()
    return None


def _split_items(value: str | None) -> list[str]:
    if value is None or value.strip().lower() in {"", "none", "n/a"}:
        return []
    return [item.strip() for item in value.split("|") if item.strip()]


def parse_mapper_analysis(
    batch: DocumentBatch,
    output: str,
    *,
    error: str | None = None,
) -> MapperAnalysis:
    return MapperAnalysis(
        batch_id=batch.batch_id,
        document_ids=batch.document_ids,
        partial_answer=_labeled_value(output, "PARTIAL_ANSWER") or output.strip(),
        findings=_split_items(_labeled_value(output, "FINDINGS")),
        evidence=_split_items(_labeled_value(output, "EVIDENCE")),
        limitations=_split_items(_labeled_value(output, "LIMITATIONS")),
        raw_output=output.strip(),
        error=error,
    )


def parse_reduced_analysis(output: str) -> ReducedAnalysis:
    answer = _labeled_value(output, "FINAL_ANSWER")
    if not answer and "Final Answer:" in output:
        answer = output.split("Final Answer:", maxsplit=1)[1].strip()
    return ReducedAnalysis(
        answer=answer or output.strip(),
        resolved_contradictions=_split_items(_labeled_value(output, "RESOLVED_CONTRADICTIONS")),
        prioritized_evidence=_split_items(_labeled_value(output, "PRIORITIZED_EVIDENCE")),
        limitations=_split_items(_labeled_value(output, "LIMITATIONS")),
        raw_output=output.strip(),
    )


def build_deterministic_map_reduce_output(
    input_data: ExperimentInput,
    component: str,
    prompt: str,
) -> str:
    if component == MAPPER:
        batch_id = detect_map_reduce_batch_id(prompt) or "unknown_batch"
        ids_line = next(
            (line for line in prompt.splitlines() if line.startswith("Batch document ids:")),
            "Batch document ids: []",
        )
        ids_text = ids_line.removeprefix("Batch document ids:").strip()
        return (
            f"PARTIAL_ANSWER={batch_id} provides bounded evidence relevant to: {input_data.query}\n"
            f"FINDINGS=Processed the assigned batch only | Preserved source traceability for {ids_text}\n"
            f"EVIDENCE=Internal document ids {ids_text}\n"
            "LIMITATIONS=Conclusions are limited to this batch"
        )
    if component == REDUCER:
        document_ids = ", ".join(document.document_id for document in input_data.documents)
        return (
            "FINAL_ANSWER=The mapper outputs provide a global answer synthesized from deterministic document batches.\n"
            "RESOLVED_CONTRADICTIONS=No material contradiction in the deterministic mapper outputs\n"
            f"PRIORITIZED_EVIDENCE=Traceable evidence from {document_ids or 'no documents'}\n"
            "LIMITATIONS=The conclusion is limited to the supplied benchmark documents"
        )
    raise ValueError(f"Unknown ARCH_07 component: {component}")


def make_partitioner_step(
    *,
    input_data: ExperimentInput,
    batches: list[DocumentBatch],
    batch_size: int,
    started_at: datetime,
    finished_at: datetime,
    actor: str,
    framework_primitive: str,
) -> AgentStep:
    return AgentStep(
        step_id=1,
        name=PARTITIONER,
        step_type="document_partition",
        actor=actor,
        input_data={
            "total_documents": len(input_data.documents),
            "batch_size": batch_size,
            "document_ids": [document.document_id for document in input_data.documents],
        },
        output_data={
            "batch_count": len(batches),
            "batches": [
                {
                    "batch_id": batch.batch_id,
                    "batch_index": batch.batch_index,
                    "document_ids": batch.document_ids,
                    "document_count": len(batch.documents),
                }
                for batch in batches
            ],
        },
        started_at=started_at,
        finished_at=finished_at,
        metadata={
            "architecture": ARCHITECTURE,
            "deterministic_partitioning": True,
            "framework_primitive": framework_primitive,
        },
    )


def make_mapper_step(
    *,
    batch: DocumentBatch,
    prompt: str,
    partial_output: dict[str, Any],
    llm_call_id: str | None,
    started_at: datetime,
    finished_at: datetime,
    actor: str,
    framework_primitive: str,
    parallelism_used: bool,
    error: str | None = None,
) -> AgentStep:
    return AgentStep(
        step_id=batch.batch_index + 2,
        name=mapper_name(batch),
        step_type="map_batch_llm_call",
        actor=actor,
        input_data={
            "prompt": prompt,
            "component": MAPPER,
            "batch_id": batch.batch_id,
            "batch_index": batch.batch_index,
            "document_ids": batch.document_ids,
            "document_count": len(batch.documents),
            "depends_on": [PARTITIONER],
        },
        output_data={"partial_output": partial_output},
        llm_call_ids=[llm_call_id] if llm_call_id else [],
        started_at=started_at,
        finished_at=finished_at,
        error=error,
        metadata={
            "architecture": ARCHITECTURE,
            "mapper_equivalence_group": "document_batch_mapper",
            "parallelism_used": parallelism_used,
            "framework_primitive": framework_primitive,
            "latency_ms": max((finished_at - started_at).total_seconds() * 1000, 0.0),
        },
    )


def make_reducer_step(
    *,
    batches: list[DocumentBatch],
    prompt: str,
    reducer_output: str,
    partial_outputs: dict[str, dict[str, Any]],
    llm_call_id: str,
    started_at: datetime,
    finished_at: datetime,
    actor: str,
    framework_primitive: str,
    parallelism_used: bool,
) -> AgentStep:
    return AgentStep(
        step_id=len(batches) + 2,
        name=REDUCER,
        step_type="reduce_llm_call",
        actor=actor,
        input_data={
            "prompt": prompt,
            "component": REDUCER,
            "depends_on": [mapper_name(batch) for batch in batches],
            "partial_outputs": partial_outputs,
            "original_documents_included": False,
        },
        output_data={"reducer_output": reducer_output.strip()},
        llm_call_ids=[llm_call_id],
        started_at=started_at,
        finished_at=finished_at,
        metadata={
            "architecture": ARCHITECTURE,
            "parallelism_used": parallelism_used,
            "framework_primitive": framework_primitive,
            "latency_ms": max((finished_at - started_at).total_seconds() * 1000, 0.0),
        },
    )


def build_map_reduce_execution_metadata(
    *,
    input_data: ExperimentInput,
    batch_size: int,
    batches: list[DocumentBatch],
    steps: list[AgentStep],
    llm_calls: list[LLMCallMetrics],
    parallelism_used: bool,
    fallback_sequential: bool,
    framework_primitive: str,
) -> dict[str, Any]:
    calls_by_id = {call.call_id: call for call in llm_calls}

    def metrics_for(step: AgentStep) -> dict[str, Any]:
        calls = [calls_by_id[call_id] for call_id in step.llm_call_ids if call_id in calls_by_id]
        return {
            "latency_ms": max((step.finished_at - step.started_at).total_seconds() * 1000, 0.0),
            "llm_call_count": len(calls),
            "input_tokens": sum(call.token_usage.input_tokens for call in calls),
            "output_tokens": sum(call.token_usage.output_tokens for call in calls),
            "error": step.error,
        }

    mapper_steps = {step.name: step for step in steps if step.step_type == "map_batch_llm_call"}
    mapper_metrics = {}
    for batch in batches:
        step = mapper_steps[mapper_name(batch)]
        mapper_metrics[step.name] = {
            **metrics_for(step),
            "batch_id": batch.batch_id,
            "document_ids": batch.document_ids,
            "document_count": len(batch.documents),
            "partial_output_size_chars": len(
                json.dumps(step.output_data.get("partial_output", {}), ensure_ascii=False)
            ),
        }
    batches_completed = [
        values["batch_id"] for values in mapper_metrics.values() if values["error"] is None
    ]
    batches_failed = [
        batch.batch_id for batch in batches if batch.batch_id not in batches_completed
    ]
    reducer_step = next(step for step in steps if step.name == REDUCER)
    elapsed_seconds = max(
        (reducer_step.finished_at - steps[0].started_at).total_seconds(),
        0.0,
    )
    return {
        "total_documents": len(input_data.documents),
        "batch_size": batch_size,
        "batch_count": len(batches),
        "mapper_count": len(mapper_steps),
        "parallelism_used": parallelism_used,
        "fallback_sequential": fallback_sequential,
        "framework_primitive": framework_primitive,
        "batches_completed": batches_completed,
        "batches_failed": batches_failed,
        "mapper_metrics": mapper_metrics,
        "reducer_metrics": metrics_for(reducer_step),
        "throughput_docs_per_second": (
            len(input_data.documents) / elapsed_seconds if elapsed_seconds > 0 else 0.0
        ),
    }


def build_map_reduce_structured_output(
    *,
    input_data: ExperimentInput,
    config: ExperimentConfig,
    batches: list[DocumentBatch],
    partial_outputs: dict[str, dict[str, Any]],
    reducer_output: str,
    steps: list[AgentStep],
    llm_calls: list[LLMCallMetrics],
    framework_execution: str,
    framework_primitive: str,
    parallelism_used: bool,
    fallback_sequential: bool = False,
) -> tuple[str, dict[str, Any]]:
    reduced = parse_reduced_analysis(reducer_output)
    batch_size = get_batch_size(config)
    execution = build_map_reduce_execution_metadata(
        input_data=input_data,
        batch_size=batch_size,
        batches=batches,
        steps=steps,
        llm_calls=llm_calls,
        parallelism_used=parallelism_used,
        fallback_sequential=fallback_sequential,
        framework_primitive=framework_primitive,
    )
    structured_output = {
        "answer": reduced.answer,
        "mode": f"{config.model_provider}_map_reduce_agentic",
        "partitioning": {
            "strategy": "stable_contiguous_batches",
            "total_documents": len(input_data.documents),
            "batch_size": batch_size,
            "batch_count": len(batches),
            "batches": [
                {
                    "batch_id": batch.batch_id,
                    "batch_index": batch.batch_index,
                    "document_ids": batch.document_ids,
                }
                for batch in batches
            ],
        },
        "partial_outputs": partial_outputs,
        "reducer": reduced.model_dump(),
        "parallelism_used": parallelism_used,
        "fallback_sequential": fallback_sequential,
        "map_reduce_execution": execution,
        "framework_execution": framework_execution,
    }
    return reduced.answer, structured_output
