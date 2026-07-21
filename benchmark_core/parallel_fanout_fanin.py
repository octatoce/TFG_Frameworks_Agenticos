"""Shared semantics and trace helpers for ARCH_06_PARALLEL_FANOUT_FANIN.

This module intentionally contains no framework orchestration.  Each framework
owns its fan-out/fan-in graph; the helpers below only keep prompts, parsing,
deterministic baseline responses, trace shape, and branch metrics comparable.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from benchmark_core.schemas import AgentStep, ExperimentConfig, ExperimentInput, LLMCallMetrics


PARALLEL_BRANCHES = (
    "factual_analysis_branch",
    "technical_reasoning_branch",
    "risk_constraints_branch",
    "alternative_solution_branch",
)
AGGREGATOR = "aggregator"
PARALLEL_COMPONENTS = (*PARALLEL_BRANCHES, AGGREGATOR)


class BranchAnalysis(BaseModel):
    """Normalized partial output produced independently by one fan-out branch."""

    branch_name: str
    analysis: str
    key_points: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)
    raw_output: str
    error: str | None = None


class AggregatedAnalysis(BaseModel):
    """Normalized fan-in output produced by the only aggregator."""

    answer: str
    resolved_contradictions: list[str] = Field(default_factory=list)
    integrated_risks: list[str] = Field(default_factory=list)
    alternatives_considered: list[str] = Field(default_factory=list)
    raw_output: str


def document_ids(input_data: ExperimentInput) -> list[str]:
    return [document.document_id for document in input_data.documents]


def render_parallel_fanout_fanin_prompt(
    input_data: ExperimentInput,
    component: str,
    partial_outputs: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Render the canonical ARCH_06 prompt for a branch or the aggregator."""

    if component not in PARALLEL_COMPONENTS:
        raise ValueError(f"Unknown ARCH_06 component: {component}")

    document_blocks = "\n".join(
        f"[{document.document_id}] {document.content}" for document in input_data.documents
    ) or "No documents provided."
    metadata_json = json.dumps(input_data.metadata, ensure_ascii=False, sort_keys=True)

    common = (
        "You are executing ARCH_06_PARALLEL_FANOUT_FANIN for a benchmark.\n"
        f"Fan-out component: {component}\n"
        "Use only the common benchmark input and tools. Keep the output concise and traceable.\n\n"
        f"Task type: {input_data.task_type}\n"
        f"Question: {input_data.query}\n"
        f"Input metadata: {metadata_json}\n\n"
        f"Documents:\n{document_blocks}\n\n"
    )

    if component == AGGREGATOR:
        missing = [branch for branch in PARALLEL_BRANCHES if branch not in (partial_outputs or {})]
        if missing:
            raise ValueError(f"Aggregator requires all four branch outputs; missing: {missing}")
        partial_json = json.dumps(partial_outputs, ensure_ascii=False, sort_keys=True)
        return common + (
            "You are the only fan-in point. Use all four independent partial outputs below. "
            "Deduplicate claims, resolve contradictions, integrate risks, and retain useful alternatives. "
            "Do not start another round, delegate, supervise, or hand off.\n\n"
            "Return exactly these labeled fields:\n"
            "FINAL_ANSWER=concise final answer\n"
            "RESOLVED_CONTRADICTIONS=item 1 | item 2 | none\n"
            "INTEGRATED_RISKS=item 1 | item 2 | none\n"
            "ALTERNATIVES_CONSIDERED=item 1 | item 2 | none\n\n"
            f"Partial outputs JSON:\n{partial_json}\n"
        )

    responsibilities = {
        "factual_analysis_branch": (
            "Extract objective facts, evidence, internal citations, and relevant fragments. "
            "Do not produce the complete final answer."
        ),
        "technical_reasoning_branch": (
            "Reason technically and propose a preliminary solution or decision directly from the common input. "
            "Do not depend on another branch and do not aggregate."
        ),
        "risk_constraints_branch": (
            "Identify risks, limitations, contradictions, uncertainty, weak assumptions, and edge conditions. "
            "Do not act as final judge."
        ),
        "alternative_solution_branch": (
            "Provide a genuinely different interpretation, solution, or complementary approach. "
            "Do not repeat the technical branch and do not aggregate."
        ),
    }
    return common + (
        f"Independent responsibility: {responsibilities[component]}\n"
        "This branch has no access to other branch outputs and must not wait for them.\n\n"
        "Return exactly these labeled fields:\n"
        "ANALYSIS=the branch-specific analysis\n"
        "KEY_POINTS=item 1 | item 2 | none\n"
        "EVIDENCE=item 1 | item 2 | none\n"
        "RISKS=item 1 | item 2 | none\n"
        "ALTERNATIVES=item 1 | item 2 | none\n"
    )


def detect_parallel_component(prompt: str) -> str | None:
    """Return the ARCH_06 component embedded in a canonical prompt."""

    if "ARCH_06_PARALLEL_FANOUT_FANIN" not in prompt:
        return None
    prefix = "Fan-out component:"
    for line in prompt.splitlines():
        if line.startswith(prefix):
            component = line.removeprefix(prefix).strip().lower()
            return component if component in PARALLEL_COMPONENTS else None
    return None


def _split_items(value: str | None) -> list[str]:
    if value is None or value.strip().lower() in {"", "none", "n/a"}:
        return []
    return [item.strip() for item in value.split("|") if item.strip()]


def _labeled_value(output: str, label: str) -> str | None:
    prefix = f"{label}="
    for line in output.splitlines():
        if line.strip().startswith(prefix):
            return line.strip().removeprefix(prefix).strip()
    return None


def parse_branch_analysis(component: str, output: str, error: str | None = None) -> BranchAnalysis:
    """Normalize a branch response without requiring a schema change."""

    analysis = _labeled_value(output, "ANALYSIS") or output.strip()
    return BranchAnalysis(
        branch_name=component,
        analysis=analysis,
        key_points=_split_items(_labeled_value(output, "KEY_POINTS")),
        evidence=_split_items(_labeled_value(output, "EVIDENCE")),
        risks=_split_items(_labeled_value(output, "RISKS")),
        alternatives=_split_items(_labeled_value(output, "ALTERNATIVES")),
        raw_output=output.strip(),
        error=error,
    )


def parse_aggregated_analysis(output: str) -> AggregatedAnalysis:
    """Normalize the aggregator response and retain its complete raw output."""

    answer = _labeled_value(output, "FINAL_ANSWER")
    if not answer and "Final Answer:" in output:
        answer = output.split("Final Answer:", maxsplit=1)[1].strip()
    return AggregatedAnalysis(
        answer=answer or output.strip(),
        resolved_contradictions=_split_items(_labeled_value(output, "RESOLVED_CONTRADICTIONS")),
        integrated_risks=_split_items(_labeled_value(output, "INTEGRATED_RISKS")),
        alternatives_considered=_split_items(_labeled_value(output, "ALTERNATIVES_CONSIDERED")),
        raw_output=output.strip(),
    )


def build_deterministic_parallel_output(
    input_data: ExperimentInput,
    component: str,
) -> str:
    """Build reproducible local outputs for the five ARCH_06 components."""

    sources = document_ids(input_data)
    source_text = ", ".join(sources) if sources else "no documents"
    first_fragment = (
        input_data.documents[0].content.strip().replace("\n", " ")[:180]
        if input_data.documents
        else "No documentary evidence was supplied."
    )

    if component == "factual_analysis_branch":
        return (
            f"ANALYSIS=Objective information was extracted from {source_text}.\n"
            f"KEY_POINTS=The case asks: {input_data.query} | Available sources: {source_text}\n"
            f"EVIDENCE={first_fragment} | Internal source ids: {source_text}\n"
            "RISKS=Evidence is limited to the supplied benchmark documents\n"
            "ALTERNATIVES=none"
        )
    if component == "technical_reasoning_branch":
        return (
            "ANALYSIS=The technically appropriate preliminary solution is to answer directly from the common input "
            "while preserving traceability to the supplied sources.\n"
            "KEY_POINTS=Use the query constraints | Keep the conclusion grounded and concise\n"
            f"EVIDENCE=Direct inspection of source ids: {source_text}\n"
            "RISKS=The preliminary decision may need qualification when evidence is incomplete\n"
            "ALTERNATIVES=none"
        )
    if component == "risk_constraints_branch":
        return (
            "ANALYSIS=The answer may be constrained by incomplete context, ambiguous requirements, and synthetic data.\n"
            "KEY_POINTS=State uncertainty | Avoid unsupported generalization\n"
            f"EVIDENCE=Only these sources are available: {source_text}\n"
            "RISKS=Missing evidence | Ambiguous interpretation | Overconfidence\n"
            "ALTERNATIVES=Qualify the answer when the documents do not resolve the query"
        )
    if component == "alternative_solution_branch":
        return (
            "ANALYSIS=An alternative is to present a bounded interpretation plus a complementary option instead of "
            "a single unqualified conclusion.\n"
            "KEY_POINTS=Offer a second interpretation | Separate facts from assumptions\n"
            f"EVIDENCE=Alternative is evaluated against the same sources: {source_text}\n"
            "RISKS=Additional options can reduce concision\n"
            "ALTERNATIVES=Bounded primary answer | Complementary interpretation"
        )
    if component == AGGREGATOR:
        return (
            "FINAL_ANSWER=The four independent analyses support a concise answer grounded in the supplied input, "
            "qualified by evidence limits and accompanied by a reasonable alternative where ambiguity remains.\n"
            "RESOLVED_CONTRADICTIONS=No material contradiction in the deterministic branch outputs\n"
            "INTEGRATED_RISKS=Incomplete context | Avoid unsupported generalization\n"
            "ALTERNATIVES_CONSIDERED=Bounded primary answer | Complementary interpretation"
        )
    raise ValueError(f"Unknown ARCH_06 component: {component}")


def make_parallel_step(
    *,
    step_id: int,
    component: str,
    actor: str,
    prompt: str,
    output: dict[str, Any],
    llm_call_id: str | None,
    started_at: datetime,
    finished_at: datetime,
    framework_primitive: str,
    parallelism_used: bool,
    partial_outputs: dict[str, dict[str, Any]] | None = None,
    error: str | None = None,
) -> AgentStep:
    """Create a comparable branch or aggregator trace entry."""

    is_aggregator = component == AGGREGATOR
    input_payload: dict[str, Any] = {
        "prompt": prompt,
        "component": component,
        "depends_on": list(PARALLEL_BRANCHES) if is_aggregator else [],
    }
    if is_aggregator:
        input_payload["partial_outputs"] = partial_outputs or {}
    return AgentStep(
        step_id=step_id,
        name=component,
        step_type="parallel_aggregator_llm_call" if is_aggregator else "parallel_branch_llm_call",
        actor=actor,
        input_data=input_payload,
        output_data=output,
        llm_call_ids=[llm_call_id] if llm_call_id else [],
        started_at=started_at,
        finished_at=finished_at,
        error=error,
        metadata={
            "architecture": "ARCH_06_PARALLEL_FANOUT_FANIN",
            "parallel_group": "fanout_branches" if not is_aggregator else "fanin",
            "framework_primitive": framework_primitive,
            "parallelism_used": parallelism_used,
            "latency_ms": max((finished_at - started_at).total_seconds() * 1000, 0.0),
        },
    )


def build_parallel_execution_metadata(
    *,
    steps: list[AgentStep],
    llm_calls: list[LLMCallMetrics],
    parallelism_used: bool,
    fallback_sequential: bool,
    framework_primitive: str,
) -> dict[str, Any]:
    """Aggregate per-branch latency, calls, tokens, and errors from canonical traces."""

    calls_by_id = {call.call_id: call for call in llm_calls}

    def metrics_for(component: str) -> dict[str, Any]:
        step = next((item for item in steps if item.name == component), None)
        calls = [calls_by_id[call_id] for call_id in (step.llm_call_ids if step else []) if call_id in calls_by_id]
        return {
            "latency_ms": (
                max((step.finished_at - step.started_at).total_seconds() * 1000, 0.0)
                if step and step.started_at and step.finished_at
                else 0.0
            ),
            "llm_call_count": len(calls),
            "input_tokens": sum(call.token_usage.input_tokens for call in calls),
            "output_tokens": sum(call.token_usage.output_tokens for call in calls),
            "error": step.error if step else "missing_step",
        }

    branch_metrics = {branch: metrics_for(branch) for branch in PARALLEL_BRANCHES}
    branches_completed = [branch for branch, values in branch_metrics.items() if values["error"] is None]
    branches_failed = [branch for branch in PARALLEL_BRANCHES if branch not in branches_completed]
    return {
        "parallelism_used": parallelism_used,
        "fallback_sequential": fallback_sequential,
        "framework_primitive": framework_primitive,
        "branches_completed": branches_completed,
        "branches_failed": branches_failed,
        "branch_metrics": branch_metrics,
        "aggregator_metrics": metrics_for(AGGREGATOR),
    }


def build_parallel_structured_output(
    *,
    input_data: ExperimentInput,
    config: ExperimentConfig,
    partial_outputs: dict[str, dict[str, Any]],
    aggregator_output: str,
    steps: list[AgentStep],
    llm_calls: list[LLMCallMetrics],
    framework_execution: str,
    framework_primitive: str,
    parallelism_used: bool,
    fallback_sequential: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Build the common final output shape for every ARCH_06 implementation."""

    aggregated = parse_aggregated_analysis(aggregator_output)
    execution = build_parallel_execution_metadata(
        steps=steps,
        llm_calls=llm_calls,
        parallelism_used=parallelism_used,
        fallback_sequential=fallback_sequential,
        framework_primitive=framework_primitive,
    )
    structured_output = {
        "answer": aggregated.answer,
        "mode": f"{config.model_provider}_parallel_fanout_fanin",
        "partial_outputs": partial_outputs,
        "aggregator": aggregated.model_dump(),
        "branches_completed": execution["branches_completed"],
        "branches_failed": execution["branches_failed"],
        "parallelism_used": parallelism_used,
        "fallback_sequential": fallback_sequential,
        "parallel_execution": execution,
        "document_ids": document_ids(input_data),
        "framework_execution": framework_execution,
    }
    return aggregated.answer, structured_output
