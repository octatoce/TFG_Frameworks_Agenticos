"""Shared semantics and trace helpers for ARCH_09_REFLECTION_CRITIC_LOOP.

The framework implementations own their loops.  This module only centralizes
configuration, prompts, typed intermediate values, deterministic local
responses, the stop policy, canonical traces, and additive metrics.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from benchmark_core.schemas import AgentStep, ExperimentConfig, ExperimentInput, LLMCallMetrics


GENERATOR = "generator"
CRITIC = "critic"
REVISER = "reviser"
STOP_CONTROLLER = "stop_controller"
REFLECTION_COMPONENTS = (GENERATOR, CRITIC, REVISER)
DEFAULT_MAX_ITERATIONS = 2
DEFAULT_QUALITY_THRESHOLD = 0.85


class ReflectionSettings(BaseModel):
    """Validated bounded settings shared by every framework runner."""

    max_iterations: int
    quality_threshold: float


class ReflectionVersion(BaseModel):
    """One generator or reviser version of the single evolving answer."""

    version_index: int
    iteration: int
    created_by: Literal["generator", "reviser"]
    answer: str
    evidence: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    limitations: list[str] = Field(default_factory=list)
    changes_applied: list[str] = Field(default_factory=list)
    raw_output: str
    error: str | None = None


class CritiqueEvaluation(BaseModel):
    """Structured critic assessment for one current answer version."""

    iteration: int
    no_critical_issues: bool
    score: float
    severity: Literal["critical", "major", "minor", "none"]
    issues: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    format_valid: bool
    raw_output: str
    error: str | None = None


class StopDecision(BaseModel):
    """Deterministic, fully traceable stop-controller result."""

    iteration: int
    should_stop: bool
    stop_reason: Literal[
        "quality_sufficient",
        "quality_score_threshold_reached",
        "minor_issues_only",
        "max_iterations_reached",
        "continue_revision",
    ]
    stopped_by_quality: bool
    stopped_by_max_iterations: bool
    quality_threshold: float
    current_score: float
    current_version_index: int


def document_ids(input_data: ExperimentInput) -> list[str]:
    return [document.document_id for document in input_data.documents]


def get_reflection_settings(config: ExperimentConfig) -> ReflectionSettings:
    """Resolve a small reflection limit without changing ExperimentConfig."""

    configured_cap = max(int(config.max_agent_iterations), 1)
    requested = config.metadata.get(
        "reflection_max_iterations",
        config.metadata.get("max_iterations", min(DEFAULT_MAX_ITERATIONS, configured_cap)),
    )
    try:
        max_iterations = int(requested)
    except (TypeError, ValueError) as exc:
        raise ValueError("reflection_max_iterations must be an integer.") from exc
    if max_iterations < 1:
        raise ValueError("reflection_max_iterations must be at least 1.")
    max_iterations = min(max_iterations, configured_cap)

    try:
        quality_threshold = float(
            config.metadata.get("reflection_quality_threshold", DEFAULT_QUALITY_THRESHOLD)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("reflection_quality_threshold must be numeric.") from exc
    if not 0.0 <= quality_threshold <= 1.0:
        raise ValueError("reflection_quality_threshold must be between 0.0 and 1.0.")
    return ReflectionSettings(
        max_iterations=max_iterations,
        quality_threshold=quality_threshold,
    )


def render_reflection_prompt(
    input_data: ExperimentInput,
    component: str,
    *,
    iteration: int,
    settings: ReflectionSettings,
    current_version: dict[str, Any] | None = None,
    critique: dict[str, Any] | None = None,
) -> str:
    """Render the canonical generator, critic, or reviser prompt."""

    if component not in REFLECTION_COMPONENTS:
        raise ValueError(f"Unknown ARCH_09 component: {component}")
    if component == GENERATOR and iteration != 0:
        raise ValueError("Generator must use reflection iteration 0.")
    if component in {CRITIC, REVISER} and (iteration < 1 or current_version is None):
        raise ValueError(f"{component} requires a current version and iteration >= 1.")
    if component == REVISER and critique is None:
        raise ValueError("Reviser requires the current critique.")

    document_blocks = "\n".join(
        f"[{document.document_id}] {document.content}" for document in input_data.documents
    ) or "No documents provided."
    metadata_json = json.dumps(input_data.metadata, ensure_ascii=False, sort_keys=True)
    common = (
        "You are executing ARCH_09_REFLECTION_CRITIC_LOOP for a benchmark.\n"
        f"Reflection component: {component}\n"
        f"Reflection iteration: {iteration}\n"
        f"Maximum reflection iterations: {settings.max_iterations}\n"
        f"Quality threshold: {settings.quality_threshold:.2f}\n"
        "This architecture improves one answer sequentially. Do not debate independent proposals, judge, "
        "route, hand off, supervise, fan out, or start any hidden loop.\n\n"
        f"Task type: {input_data.task_type}\n"
        f"Question: {input_data.query}\n"
        f"Input metadata: {metadata_json}\n\n"
        f"Documents:\n{document_blocks}\n\n"
    )

    if component == GENERATOR:
        return common + (
            "Create version 0: one concise initial answer grounded in the common input. State evidence, "
            "confidence, and limitations. Do not critique or revise yourself.\n\n"
            "Return exactly these labeled fields:\n"
            "ANSWER=initial answer\n"
            "EVIDENCE=item 1 | item 2 | none\n"
            "CONFIDENCE=number from 0.0 to 1.0\n"
            "LIMITATIONS=item 1 | item 2 | none\n"
        )

    current_json = json.dumps(current_version, ensure_ascii=False, sort_keys=True)
    if component == CRITIC:
        return common + (
            "Evaluate only the current answer below. Detect factual or logical errors, ambiguity, "
            "contradictions, evidence gaps, overconfidence, and format defects. Suggest concrete improvements. "
            "Do not rewrite the answer and do not decide whether the workflow stops.\n\n"
            "Return exactly these labeled fields:\n"
            "NO_CRITICAL_ISSUES=true|false\n"
            "SCORE=number from 0.0 to 1.0\n"
            "SEVERITY=critical|major|minor|none\n"
            "ISSUES=item 1 | item 2 | none\n"
            "IMPROVEMENTS=item 1 | item 2 | none\n"
            "EVIDENCE_GAPS=item 1 | item 2 | none\n"
            "FORMAT_VALID=true|false\n\n"
            f"Current version JSON:\n{current_json}\n"
        )

    critique_json = json.dumps(critique, ensure_ascii=False, sort_keys=True)
    return common + (
        "Revise the single current answer using the supplied critique. Apply concrete corrections while "
        "preserving useful content and the common structured format. Do not critique again, decide whether "
        "to stop, consult another agent, or create alternative competing proposals.\n\n"
        "Return exactly these labeled fields:\n"
        "ANSWER=revised answer\n"
        "EVIDENCE=item 1 | item 2 | none\n"
        "CONFIDENCE=number from 0.0 to 1.0\n"
        "LIMITATIONS=item 1 | item 2 | none\n"
        "CHANGES_APPLIED=item 1 | item 2 | none\n\n"
        f"Current version JSON:\n{current_json}\n\n"
        f"Critique JSON:\n{critique_json}\n"
    )


def detect_reflection_component(prompt: str) -> tuple[str, int] | None:
    """Detect the ARCH_09 component and iteration in one canonical prompt."""

    if "ARCH_09_REFLECTION_CRITIC_LOOP" not in prompt:
        return None
    component = None
    iteration = None
    for line in prompt.splitlines():
        if line.startswith("Reflection component:"):
            candidate = line.removeprefix("Reflection component:").strip().lower()
            component = candidate if candidate in REFLECTION_COMPONENTS else None
        elif line.startswith("Reflection iteration:"):
            try:
                iteration = int(line.removeprefix("Reflection iteration:").strip())
            except ValueError:
                return None
    if component is None or iteration is None:
        return None
    return component, iteration


def _labeled_value(output: str, label: str) -> str | None:
    prefix = f"{label}="
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped.removeprefix(prefix).strip()
    return None


def _split_items(value: str | None) -> list[str]:
    if value is None or value.strip().lower() in {"", "none", "n/a", "null"}:
        return []
    return [item.strip() for item in value.split("|") if item.strip()]


def _parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"true", "1", "yes", "si", "sí"}


def _parse_score(value: str | None) -> float:
    try:
        return min(max(float(value or "0"), 0.0), 1.0)
    except ValueError:
        return 0.0


def parse_reflection_version(
    output: str,
    *,
    version_index: int,
    iteration: int,
    created_by: Literal["generator", "reviser"],
    error: str | None = None,
) -> ReflectionVersion:
    """Normalize one initial or revised answer version."""

    answer = _labeled_value(output, "ANSWER")
    if not answer and "Final Answer:" in output:
        answer = output.split("Final Answer:", maxsplit=1)[1].strip()
    return ReflectionVersion(
        version_index=version_index,
        iteration=iteration,
        created_by=created_by,
        answer=answer or output.strip(),
        evidence=_split_items(_labeled_value(output, "EVIDENCE")),
        confidence=_parse_score(_labeled_value(output, "CONFIDENCE")),
        limitations=_split_items(_labeled_value(output, "LIMITATIONS")),
        changes_applied=_split_items(_labeled_value(output, "CHANGES_APPLIED")),
        raw_output=output.strip(),
        error=error,
    )


def parse_critique(
    output: str,
    *,
    iteration: int,
    error: str | None = None,
) -> CritiqueEvaluation:
    """Normalize one critic response with conservative fallbacks."""

    raw_severity = (_labeled_value(output, "SEVERITY") or "critical").lower()
    severity = raw_severity if raw_severity in {"critical", "major", "minor", "none"} else "critical"
    return CritiqueEvaluation(
        iteration=iteration,
        no_critical_issues=_parse_bool(_labeled_value(output, "NO_CRITICAL_ISSUES")),
        score=_parse_score(_labeled_value(output, "SCORE")),
        severity=severity,
        issues=_split_items(_labeled_value(output, "ISSUES")),
        improvements=_split_items(_labeled_value(output, "IMPROVEMENTS")),
        evidence_gaps=_split_items(_labeled_value(output, "EVIDENCE_GAPS")),
        format_valid=_parse_bool(_labeled_value(output, "FORMAT_VALID")),
        raw_output=output.strip(),
        error=error,
    )


def evaluate_stop(
    critique: CritiqueEvaluation,
    *,
    current_version_index: int,
    settings: ReflectionSettings,
) -> StopDecision:
    """Apply the deterministic stop policy in a stable priority order."""

    if critique.no_critical_issues:
        reason = "quality_sufficient"
        should_stop = True
        stopped_by_quality = True
        stopped_by_max = False
    elif critique.score >= settings.quality_threshold:
        reason = "quality_score_threshold_reached"
        should_stop = True
        stopped_by_quality = True
        stopped_by_max = False
    elif critique.severity in {"minor", "none"}:
        reason = "minor_issues_only"
        should_stop = True
        stopped_by_quality = True
        stopped_by_max = False
    elif critique.iteration >= settings.max_iterations:
        reason = "max_iterations_reached"
        should_stop = True
        stopped_by_quality = False
        stopped_by_max = True
    else:
        reason = "continue_revision"
        should_stop = False
        stopped_by_quality = False
        stopped_by_max = False
    return StopDecision(
        iteration=critique.iteration,
        should_stop=should_stop,
        stop_reason=reason,
        stopped_by_quality=stopped_by_quality,
        stopped_by_max_iterations=stopped_by_max,
        quality_threshold=settings.quality_threshold,
        current_score=critique.score,
        current_version_index=current_version_index,
    )


def build_deterministic_reflection_output(
    input_data: ExperimentInput,
    component: str,
    iteration: int,
) -> str:
    """Build reproducible local outputs with one observable quality improvement."""

    sources = document_ids(input_data)
    source_text = ", ".join(sources) if sources else "no documents"
    first_fragment = (
        input_data.documents[0].content.strip().replace("\n", " ")[:180]
        if input_data.documents
        else "No documentary evidence was supplied."
    )
    if component == GENERATOR:
        return (
            f"ANSWER=Initial answer to '{input_data.query}' based on the supplied benchmark context.\n"
            f"EVIDENCE=Source ids: {source_text}\n"
            "CONFIDENCE=0.58\n"
            "LIMITATIONS=The initial version does not yet connect every conclusion to explicit evidence"
        )
    if component == CRITIC and iteration == 1:
        return (
            "NO_CRITICAL_ISSUES=false\n"
            "SCORE=0.62\n"
            "SEVERITY=major\n"
            "ISSUES=The conclusion is under-qualified | Evidence linkage is too generic\n"
            "IMPROVEMENTS=Tie the answer to source ids | State uncertainty and scope explicitly\n"
            f"EVIDENCE_GAPS=The fragment '{first_fragment}' is not explicitly connected to the conclusion\n"
            "FORMAT_VALID=true"
        )
    if component == CRITIC:
        return (
            "NO_CRITICAL_ISSUES=true\n"
            "SCORE=0.93\n"
            "SEVERITY=none\n"
            "ISSUES=none\n"
            "IMPROVEMENTS=Only optional stylistic shortening remains\n"
            "EVIDENCE_GAPS=none\n"
            "FORMAT_VALID=true"
        )
    if component == REVISER:
        return (
            f"ANSWER=Revised answer to '{input_data.query}' grounded in {source_text}, with explicit scope and "
            "uncertainty where the supplied evidence is incomplete.\n"
            f"EVIDENCE={first_fragment} | Source ids: {source_text}\n"
            "CONFIDENCE=0.88\n"
            "LIMITATIONS=Conclusions remain bounded to the supplied benchmark documents\n"
            "CHANGES_APPLIED=Linked claims to source ids | Qualified uncertainty | Preserved the structured format"
        )
    raise ValueError(f"Unknown ARCH_09 component: {component}")


def reflection_step_name(component: str, iteration: int) -> str:
    if component == GENERATOR:
        return GENERATOR
    return f"{component}_{iteration:03d}"


def make_reflection_step(
    *,
    step_id: int,
    component: str,
    iteration: int,
    actor: str,
    output: dict[str, Any],
    started_at: datetime,
    finished_at: datetime,
    framework_primitive: str,
    max_iterations: int,
    prompt: str | None = None,
    llm_call_ids: list[str] | None = None,
    current_version: dict[str, Any] | None = None,
    critique: dict[str, Any] | None = None,
    stop_decision: dict[str, Any] | None = None,
    depends_on: list[str] | None = None,
    error: str | None = None,
) -> AgentStep:
    """Create one canonical generator, critic, stop, or reviser trace entry."""

    step_types = {
        GENERATOR: "reflection_generator_llm_call",
        CRITIC: "reflection_critic_llm_call",
        STOP_CONTROLLER: "reflection_stop_controller",
        REVISER: "reflection_reviser_llm_call",
    }
    if component not in step_types:
        raise ValueError(f"Unknown reflection trace component: {component}")
    input_payload: dict[str, Any] = {
        "component": component,
        "iteration": iteration,
        "depends_on": depends_on or [],
    }
    if prompt is not None:
        input_payload["prompt"] = prompt
    if current_version is not None:
        input_payload["current_version"] = current_version
    if critique is not None:
        input_payload["critique"] = critique
    if stop_decision is not None:
        input_payload["stop_decision"] = stop_decision
    return AgentStep(
        step_id=step_id,
        name=reflection_step_name(component, iteration),
        step_type=step_types[component],
        actor=actor,
        input_data=input_payload,
        output_data=output,
        llm_call_ids=llm_call_ids or [],
        started_at=started_at,
        finished_at=finished_at,
        error=error,
        metadata={
            "architecture": "ARCH_09_REFLECTION_CRITIC_LOOP",
            "component": component,
            "reflection_iteration": iteration,
            "max_iterations": max_iterations,
            "framework_primitive": framework_primitive,
            "deterministic_controller": component == STOP_CONTROLLER,
            "latency_ms": max((finished_at - started_at).total_seconds() * 1000, 0.0),
        },
    )


def summarize_versions(versions: list[ReflectionVersion]) -> list[dict[str, Any]]:
    """Return compact, stable summaries for longitudinal quality analysis."""

    return [
        {
            "version_index": version.version_index,
            "iteration": version.iteration,
            "created_by": version.created_by,
            "answer_preview": version.answer[:240],
            "confidence": version.confidence,
            "evidence_count": len(version.evidence),
            "limitation_count": len(version.limitations),
            "changes_applied_count": len(version.changes_applied),
            "error": version.error,
        }
        for version in versions
    ]


def build_reflection_execution_metadata(
    *,
    versions: list[ReflectionVersion],
    critiques: list[CritiqueEvaluation],
    stop_decisions: list[StopDecision],
    steps: list[AgentStep],
    llm_calls: list[LLMCallMetrics],
    settings: ReflectionSettings,
    framework_primitive: str,
) -> dict[str, Any]:
    """Aggregate loop, component, iteration, latency, token, and error metrics."""

    calls_by_id = {call.call_id: call for call in llm_calls}

    def step_metrics(step: AgentStep) -> dict[str, Any]:
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

    generator_step = next(step for step in steps if step.metadata.get("component") == GENERATOR)
    critic_steps = [step for step in steps if step.metadata.get("component") == CRITIC]
    reviser_steps = [step for step in steps if step.metadata.get("component") == REVISER]
    stop_steps = [step for step in steps if step.metadata.get("component") == STOP_CONTROLLER]
    iteration_metrics: dict[str, Any] = {}
    for iteration in range(1, len(critiques) + 1):
        critic_step = next(
            step for step in critic_steps if step.metadata["reflection_iteration"] == iteration
        )
        reviser_step = next(
            (step for step in reviser_steps if step.metadata["reflection_iteration"] == iteration),
            None,
        )
        stop_step = next(
            step for step in stop_steps if step.metadata["reflection_iteration"] == iteration
        )
        metrics = {
            "critic": step_metrics(critic_step),
            "stop_controller": step_metrics(stop_step),
            "reviser": step_metrics(reviser_step) if reviser_step else None,
            "critic_score": critiques[iteration - 1].score,
            "critic_severity": critiques[iteration - 1].severity,
            "stop_reason": stop_decisions[iteration - 1].stop_reason,
        }
        iteration_metrics[f"iteration_{iteration:03d}"] = metrics

    final_stop = stop_decisions[-1]
    scores = [critique.score for critique in critiques]
    return {
        "max_iterations": settings.max_iterations,
        "iterations_executed": len(critiques),
        "revision_count": len(reviser_steps),
        "version_count": len(versions),
        "stop_reason": final_stop.stop_reason,
        "stopped_by_quality": final_stop.stopped_by_quality,
        "stopped_by_max_iterations": final_stop.stopped_by_max_iterations,
        "final_version_index": versions[-1].version_index,
        "quality_threshold": settings.quality_threshold,
        "critic_scores": scores,
        "critic_score_delta": scores[-1] - scores[0] if len(scores) > 1 else 0.0,
        "framework_primitive": framework_primitive,
        "parallelism_used": False,
        "generator_metrics": step_metrics(generator_step),
        "critic_metrics": {step.name: step_metrics(step) for step in critic_steps},
        "reviser_metrics": {step.name: step_metrics(step) for step in reviser_steps},
        "stop_controller_metrics": {step.name: step_metrics(step) for step in stop_steps},
        "iteration_metrics": iteration_metrics,
        "llm_call_count_by_component": {
            GENERATOR: sum(len(step.llm_call_ids) for step in [generator_step]),
            CRITIC: sum(len(step.llm_call_ids) for step in critic_steps),
            REVISER: sum(len(step.llm_call_ids) for step in reviser_steps),
            STOP_CONTROLLER: 0,
        },
        "component_errors": {
            step.name: step.error for step in steps if step.error is not None
        },
    }


def build_reflection_structured_output(
    *,
    input_data: ExperimentInput,
    config: ExperimentConfig,
    versions: list[ReflectionVersion],
    critiques: list[CritiqueEvaluation],
    stop_decisions: list[StopDecision],
    steps: list[AgentStep],
    llm_calls: list[LLMCallMetrics],
    settings: ReflectionSettings,
    framework_execution: str,
    framework_primitive: str,
) -> tuple[str, dict[str, Any]]:
    """Build the common final result payload for every ARCH_09 runner."""

    if not versions or not critiques or not stop_decisions:
        raise ValueError("Reflection output requires versions, critiques, and stop decisions.")
    execution = build_reflection_execution_metadata(
        versions=versions,
        critiques=critiques,
        stop_decisions=stop_decisions,
        steps=steps,
        llm_calls=llm_calls,
        settings=settings,
        framework_primitive=framework_primitive,
    )
    final_version = versions[-1]
    structured_output = {
        "answer": final_version.answer,
        "mode": f"{config.model_provider}_reflection_critic_loop",
        "initial_version": versions[0].model_dump(),
        "final_version": final_version.model_dump(),
        "version_history": [version.model_dump() for version in versions],
        "version_summaries": summarize_versions(versions),
        "critique_history": [critique.model_dump() for critique in critiques],
        "stop_decisions": [decision.model_dump() for decision in stop_decisions],
        "iterations_executed": execution["iterations_executed"],
        "max_iterations": settings.max_iterations,
        "revision_count": execution["revision_count"],
        "number_of_versions": execution["version_count"],
        "stop_reason": execution["stop_reason"],
        "stopped_by_quality": execution["stopped_by_quality"],
        "stopped_by_max_iterations": execution["stopped_by_max_iterations"],
        "final_version_index": execution["final_version_index"],
        "quality_threshold": settings.quality_threshold,
        "document_ids": document_ids(input_data),
        "reflection_execution": execution,
        "framework_execution": framework_execution,
    }
    return final_version.answer, structured_output
