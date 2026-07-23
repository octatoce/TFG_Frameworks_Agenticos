"""Shared semantics, validation, traces, and metrics for ARCH_10.

Framework runners own orchestration and the actual checkpoint backend.  This
module keeps prompts, typed domain state, deterministic local responses,
portable checkpoint helpers, integrity checks, and additive metrics common.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from benchmark_core.schemas import AgentStep, ExperimentConfig, ExperimentInput, LLMCallMetrics


ARCHITECTURE = "ARCH_10_CHECKPOINT_MEMORY_RECOVERY"
STATE_INITIALIZER = "state_initializer"
PLANNING_STEP = "planning_or_analysis_step"
CHECKPOINT_WRITER = "checkpoint_writer"
FAILURE_INJECTOR = "failure_injector"
RECOVERY_LOADER = "recovery_loader"
CONTINUATION_STEP = "continuation_step"
FINALIZER = "finalizer"
CHECKPOINT_STAGE = PLANNING_STEP
CONTROLLED_FAILURE_STAGE = FAILURE_INJECTOR
RECOVERY_COMPONENTS = (
    STATE_INITIALIZER,
    PLANNING_STEP,
    CHECKPOINT_WRITER,
    FAILURE_INJECTOR,
    RECOVERY_LOADER,
    CONTINUATION_STEP,
    FINALIZER,
)


class ControlledFailure(RuntimeError):
    """Expected, deterministic interruption used to exercise recovery."""


class RecoverySettings(BaseModel):
    inject_failure: bool = True


class PlanningAnalysis(BaseModel):
    analysis: str
    evidence: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    raw_output: str
    error: str | None = None


class ContinuationResult(BaseModel):
    answer: str
    decision: str
    evidence: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    raw_output: str
    error: str | None = None


class RecoveryWorkflowState(BaseModel):
    """Serializable domain state that must survive the checkpoint boundary."""

    query: str
    documents: list[dict[str, Any]]
    input_metadata: dict[str, Any]
    current_stage: str = STATE_INITIALIZER
    planning: PlanningAnalysis | None = None
    continuation: ContinuationResult | None = None
    checkpoint_id: str | None = None
    checkpoint_stage: str | None = None
    checkpoint_timestamp: datetime | None = None
    state_digest: str | None = None
    recovered: bool = False
    recovery_reason: str | None = None
    result_generated_after_recovery: bool = False


class PortableCheckpoint(BaseModel):
    """Portable JSON fallback used when a framework has no suitable backend."""

    schema_version: str = "1.0"
    architecture: str = ARCHITECTURE
    framework: str
    run_id: str
    checkpoint_id: str
    checkpoint_stage: str
    created_at: datetime
    state_digest: str
    state: RecoveryWorkflowState


def _parse_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "si", "sí", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise ValueError("checkpoint_inject_failure must be boolean-like.")


def get_recovery_settings(config: ExperimentConfig) -> RecoverySettings:
    return RecoverySettings(
        inject_failure=_parse_bool(
            config.metadata.get("checkpoint_inject_failure"),
            default=True,
        )
    )


def initialize_recovery_state(input_data: ExperimentInput) -> RecoveryWorkflowState:
    return RecoveryWorkflowState(
        query=input_data.query,
        documents=[document.model_dump(mode="json") for document in input_data.documents],
        input_metadata=dict(input_data.metadata),
    )


def document_ids(input_data: ExperimentInput) -> list[str]:
    return [document.document_id for document in input_data.documents]


def logical_checkpoint_id(config: ExperimentConfig) -> str:
    safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", config.run_id).strip(".-") or "run"
    return f"{safe_run_id}-checkpoint-001"


def checkpoint_directory(repo_root: Path, framework: str) -> Path:
    return repo_root / "results" / "checkpoints" / framework / ARCHITECTURE


def checkpoint_path(repo_root: Path, framework: str, checkpoint_id: str) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", checkpoint_id).strip(".-")
    if not safe_id:
        raise ValueError("checkpoint_id cannot be empty after normalization.")
    base = checkpoint_directory(repo_root, framework).resolve()
    path = (base / f"{safe_id}.json").resolve()
    if not path.is_relative_to(base):
        raise ValueError("Checkpoint path escapes the benchmark checkpoint directory.")
    return path


def state_digest(state: RecoveryWorkflowState) -> str:
    payload = state.model_dump(mode="json", exclude={"state_digest"})
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def seal_state_for_checkpoint(
    state: RecoveryWorkflowState,
    *,
    checkpoint_id: str,
    created_at: datetime,
) -> RecoveryWorkflowState:
    staged = state.model_copy(
        update={
            "current_stage": CHECKPOINT_WRITER,
            "checkpoint_id": checkpoint_id,
            "checkpoint_stage": CHECKPOINT_STAGE,
            "checkpoint_timestamp": created_at,
            "state_digest": None,
        }
    )
    return staged.model_copy(update={"state_digest": state_digest(staged)})


def verify_recovered_state(state: RecoveryWorkflowState) -> str:
    if state.planning is None:
        raise ValueError("Recovered checkpoint does not contain planning analysis.")
    if not state.checkpoint_id or state.checkpoint_stage != CHECKPOINT_STAGE:
        raise ValueError("Recovered checkpoint identity or stage is invalid.")
    expected = state.state_digest
    if not expected:
        raise ValueError("Recovered checkpoint does not contain an integrity digest.")
    candidate = state.model_copy(update={"state_digest": None})
    actual = state_digest(candidate)
    if actual != expected:
        raise ValueError("Recovered checkpoint state failed its SHA-256 integrity check.")
    return actual


def build_portable_checkpoint(
    *,
    framework: str,
    config: ExperimentConfig,
    state: RecoveryWorkflowState,
    created_at: datetime,
) -> PortableCheckpoint:
    if not state.checkpoint_id or not state.state_digest:
        raise ValueError("State must be sealed before building a portable checkpoint.")
    return PortableCheckpoint(
        framework=framework,
        run_id=config.run_id,
        checkpoint_id=state.checkpoint_id,
        checkpoint_stage=CHECKPOINT_STAGE,
        created_at=created_at,
        state_digest=state.state_digest,
        state=state,
    )


def write_json_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json_checkpoint(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Checkpoint payload must be a JSON object.")
    return value


def write_portable_checkpoint(path: Path, checkpoint: PortableCheckpoint) -> None:
    write_json_checkpoint(path, checkpoint.model_dump(mode="json"))


def load_portable_checkpoint(path: Path) -> PortableCheckpoint:
    checkpoint = PortableCheckpoint.model_validate(read_json_checkpoint(path))
    actual = verify_recovered_state(checkpoint.state)
    if actual != checkpoint.state_digest:
        raise ValueError("Checkpoint envelope and state digests differ.")
    return checkpoint


def render_recovery_prompt(
    input_data: ExperimentInput,
    component: Literal["planning_or_analysis_step", "continuation_step"],
    *,
    planning: PlanningAnalysis | None = None,
) -> str:
    if component == CONTINUATION_STEP and planning is None:
        raise ValueError("Continuation requires recovered planning analysis.")
    documents = "\n".join(
        f"[{document.document_id}] {document.content}" for document in input_data.documents
    ) or "No documents provided."
    metadata_json = json.dumps(input_data.metadata, ensure_ascii=False, sort_keys=True)
    common = (
        f"You are executing {ARCHITECTURE} for a benchmark.\n"
        f"Recovery component: {component}\n"
        "Use only the common query, documents, and metadata. Do not debate, reflect, route, hand off, "
        "fan out, use Map-Reduce, or start hidden loops.\n\n"
        f"Task type: {input_data.task_type}\n"
        f"Question: {input_data.query}\n"
        f"Input metadata: {metadata_json}\n\n"
        f"Documents:\n{documents}\n\n"
    )
    if component == PLANNING_STEP:
        return common + (
            "Produce a concise initial analysis that is useful after a process restart. Do not write the "
            "final answer yet.\n\n"
            "Return exactly these labeled fields:\n"
            "ANALYSIS=concise preliminary analysis\n"
            "EVIDENCE=item 1 | item 2 | none\n"
            "OPEN_QUESTIONS=item 1 | item 2 | none\n"
        )
    planning_json = json.dumps(planning.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    return common + (
        "Continue from the recovered analysis below and produce the final result. The recovered analysis "
        "is authoritative workflow state; do not redo the pre-checkpoint phase.\n\n"
        "Return exactly these labeled fields:\n"
        "ANSWER=concise final answer\n"
        "DECISION=short synthesis or decision\n"
        "EVIDENCE=item 1 | item 2 | none\n"
        "LIMITATIONS=item 1 | item 2 | none\n\n"
        f"Recovered planning JSON:\n{planning_json}\n"
    )


def detect_recovery_component(prompt: str) -> str | None:
    if ARCHITECTURE not in prompt:
        return None
    for line in prompt.splitlines():
        if line.startswith("Recovery component:"):
            component = line.removeprefix("Recovery component:").strip()
            return component if component in {PLANNING_STEP, CONTINUATION_STEP} else None
    return None


def _labeled_value(output: str, label: str) -> str | None:
    prefix = f"{label}="
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped.removeprefix(prefix).strip()
    return None


def _split_items(value: str | None) -> list[str]:
    if value is None or value.strip().lower() in {"", "none", "null", "n/a"}:
        return []
    return [item.strip() for item in value.split("|") if item.strip()]


def parse_planning_analysis(output: str, *, error: str | None = None) -> PlanningAnalysis:
    return PlanningAnalysis(
        analysis=_labeled_value(output, "ANALYSIS") or output.strip(),
        evidence=_split_items(_labeled_value(output, "EVIDENCE")),
        open_questions=_split_items(_labeled_value(output, "OPEN_QUESTIONS")),
        raw_output=output.strip(),
        error=error,
    )


def parse_continuation_result(output: str, *, error: str | None = None) -> ContinuationResult:
    answer = _labeled_value(output, "ANSWER")
    if not answer and "Final Answer:" in output:
        answer = output.split("Final Answer:", maxsplit=1)[1].strip()
    return ContinuationResult(
        answer=answer or output.strip(),
        decision=_labeled_value(output, "DECISION") or "Finalized from recovered state.",
        evidence=_split_items(_labeled_value(output, "EVIDENCE")),
        limitations=_split_items(_labeled_value(output, "LIMITATIONS")),
        raw_output=output.strip(),
        error=error,
    )


def build_deterministic_recovery_output(input_data: ExperimentInput, component: str) -> str:
    sources = document_ids(input_data)
    source_text = ", ".join(sources) if sources else "no documents"
    if component == PLANNING_STEP:
        return (
            f"ANALYSIS=Preliminary analysis for '{input_data.query}' preserves the facts needed after recovery.\n"
            f"EVIDENCE=Source ids: {source_text}\n"
            "OPEN_QUESTIONS=Confirm that the recovered digest and stage match before finalizing"
        )
    if component == CONTINUATION_STEP:
        return (
            f"ANSWER=Recovered execution completed the answer to '{input_data.query}' using {source_text}.\n"
            "DECISION=Continue from the verified checkpoint and synthesize the preserved analysis.\n"
            f"EVIDENCE=Source ids: {source_text} | Checkpoint integrity verified\n"
            "LIMITATIONS=The checkpoint backend is local to this benchmark run"
        )
    raise ValueError(f"Unknown ARCH_10 LLM component: {component}")


def controlled_failure_message(checkpoint_id: str) -> str:
    return f"ControlledFailure: deterministic interruption after checkpoint {checkpoint_id}"


def make_recovery_step(
    *,
    step_id: int,
    component: str,
    actor: str,
    started_at: datetime,
    finished_at: datetime,
    framework_primitive: str,
    output: dict[str, Any],
    input_payload: dict[str, Any] | None = None,
    llm_call_ids: list[str] | None = None,
    depends_on: list[str] | None = None,
    error: str | None = None,
    checkpoint_backend: str | None = None,
    native_checkpointing: bool | None = None,
) -> AgentStep:
    step_types = {
        STATE_INITIALIZER: "recovery_state_initialization",
        PLANNING_STEP: "recovery_planning_llm_call",
        CHECKPOINT_WRITER: "recovery_checkpoint_write",
        FAILURE_INJECTOR: "recovery_controlled_failure",
        RECOVERY_LOADER: "recovery_checkpoint_load",
        CONTINUATION_STEP: "recovery_continuation_llm_call",
        FINALIZER: "recovery_finalization",
    }
    if component not in step_types:
        raise ValueError(f"Unknown ARCH_10 trace component: {component}")
    payload = dict(input_payload or {})
    payload.setdefault("component", component)
    payload.setdefault("depends_on", depends_on or [])
    metadata: dict[str, Any] = {
        "architecture": ARCHITECTURE,
        "component": component,
        "framework_primitive": framework_primitive,
        "deterministic_component": component not in {PLANNING_STEP, CONTINUATION_STEP},
        "latency_ms": max((finished_at - started_at).total_seconds() * 1000, 0.0),
    }
    if checkpoint_backend is not None:
        metadata["checkpoint_backend"] = checkpoint_backend
    if native_checkpointing is not None:
        metadata["native_checkpointing"] = native_checkpointing
    if component == FAILURE_INJECTOR:
        metadata["controlled_failure"] = True
        metadata["failure_stage"] = CONTROLLED_FAILURE_STAGE
    return AgentStep(
        step_id=step_id,
        name=component,
        step_type=step_types[component],
        actor=actor,
        input_data=payload,
        output_data=output,
        llm_call_ids=llm_call_ids or [],
        started_at=started_at,
        finished_at=finished_at,
        error=error,
        metadata=metadata,
    )


def _step_metrics(step: AgentStep, calls_by_id: dict[str, LLMCallMetrics]) -> dict[str, Any]:
    calls = [calls_by_id[call_id] for call_id in step.llm_call_ids if call_id in calls_by_id]
    return {
        "latency_ms": (
            max((step.finished_at - step.started_at).total_seconds() * 1000, 0.0)
            if step.started_at and step.finished_at
            else 0.0
        ),
        "llm_call_count": len(calls),
        "input_tokens": sum(call.token_usage.input_tokens for call in calls),
        "output_tokens": sum(call.token_usage.output_tokens for call in calls),
        "total_tokens": sum(call.token_usage.total_tokens for call in calls),
        "error": step.error,
    }


def build_recovery_execution_metadata(
    *,
    state: RecoveryWorkflowState,
    steps: list[AgentStep],
    llm_calls: list[LLMCallMetrics],
    framework_primitive: str,
    checkpoint_backend: str,
    native_checkpointing: bool,
    recovery_source: str,
    failure_injected: bool,
    recovery_attempted: bool,
    recovery_successful: bool,
    native_checkpoint_id: str | None = None,
    native_checkpoints_created: int | None = None,
) -> dict[str, Any]:
    calls_by_id = {call.call_id: call for call in llm_calls}
    by_component = {step.name: _step_metrics(step, calls_by_id) for step in steps}
    before = [STATE_INITIALIZER, PLANNING_STEP]
    after = [CONTINUATION_STEP, FINALIZER]
    return {
        "checkpoint_used": state.checkpoint_id is not None,
        "checkpoints_created": 1 if state.checkpoint_id else 0,
        "checkpoint_id": state.checkpoint_id,
        "native_checkpoint_id": native_checkpoint_id,
        "checkpoint_stage": state.checkpoint_stage,
        "checkpoint_timestamp": (
            state.checkpoint_timestamp.isoformat() if state.checkpoint_timestamp else None
        ),
        "checkpoint_backend": checkpoint_backend,
        "native_checkpointing": native_checkpointing,
        "native_checkpoints_created": native_checkpoints_created,
        "state_digest": state.state_digest,
        "state_digest_verified": recovery_successful,
        "failure_injected": failure_injected,
        "failure_stage": CONTROLLED_FAILURE_STAGE if failure_injected else None,
        "controlled_error_count": sum(1 for step in steps if step.name == FAILURE_INJECTOR and step.error),
        "uncontrolled_error_count": sum(1 for step in steps if step.name != FAILURE_INJECTOR and step.error),
        "recovery_attempted": recovery_attempted,
        "recovery_successful": recovery_successful,
        "recovery_source": recovery_source,
        "recovery_reason": state.recovery_reason,
        "result_generated_after_recovery": state.result_generated_after_recovery,
        "steps_before_failure": len([name for name in (STATE_INITIALIZER, PLANNING_STEP, CHECKPOINT_WRITER) if name in by_component]),
        "steps_after_recovery": len([name for name in after if name in by_component]),
        "latency_before_checkpoint_ms": sum(by_component[name]["latency_ms"] for name in before if name in by_component),
        "checkpoint_write_latency_ms": by_component.get(CHECKPOINT_WRITER, {}).get("latency_ms", 0.0),
        "recovery_latency_ms": by_component.get(RECOVERY_LOADER, {}).get("latency_ms", 0.0),
        "latency_after_recovery_ms": sum(by_component[name]["latency_ms"] for name in after if name in by_component),
        "llm_call_count_by_component": {
            component: by_component.get(component, {}).get("llm_call_count", 0)
            for component in RECOVERY_COMPONENTS
        },
        "component_metrics": by_component,
        "framework_primitive": framework_primitive,
    }


def build_recovery_structured_output(
    *,
    input_data: ExperimentInput,
    config: ExperimentConfig,
    state: RecoveryWorkflowState,
    steps: list[AgentStep],
    llm_calls: list[LLMCallMetrics],
    framework_execution: str,
    framework_primitive: str,
    checkpoint_backend: str,
    native_checkpointing: bool,
    recovery_source: str,
    failure_injected: bool,
    recovery_attempted: bool,
    recovery_successful: bool,
    native_checkpoint_id: str | None = None,
    native_checkpoints_created: int | None = None,
) -> tuple[str, dict[str, Any]]:
    if state.planning is None or state.continuation is None:
        raise ValueError("Finalization requires planning and continuation outputs.")
    execution = build_recovery_execution_metadata(
        state=state,
        steps=steps,
        llm_calls=llm_calls,
        framework_primitive=framework_primitive,
        checkpoint_backend=checkpoint_backend,
        native_checkpointing=native_checkpointing,
        recovery_source=recovery_source,
        failure_injected=failure_injected,
        recovery_attempted=recovery_attempted,
        recovery_successful=recovery_successful,
        native_checkpoint_id=native_checkpoint_id,
        native_checkpoints_created=native_checkpoints_created,
    )
    final_answer = state.continuation.answer
    return final_answer, {
        "answer": final_answer,
        "decision": state.continuation.decision,
        "evidence": state.continuation.evidence,
        "limitations": state.continuation.limitations,
        "initial_analysis": state.planning.model_dump(),
        "continuation": state.continuation.model_dump(),
        "document_ids": document_ids(input_data),
        "mode": f"{config.model_provider}_checkpoint_memory_recovery",
        "checkpoint_used": execution["checkpoint_used"],
        "failure_injected": execution["failure_injected"],
        "recovery_attempted": execution["recovery_attempted"],
        "recovery_successful": execution["recovery_successful"],
        "checkpoint_id": execution["checkpoint_id"],
        "checkpoint_stage": execution["checkpoint_stage"],
        "result_generated_after_recovery": execution["result_generated_after_recovery"],
        "recovery_execution": execution,
        "framework_execution": framework_execution,
    }
