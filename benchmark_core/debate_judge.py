"""Shared semantics and trace helpers for ARCH_08_DEBATE_JUDGE.

Framework-specific orchestration deliberately stays in each implementation.
This module only centralizes prompts, typed partial outputs, deterministic local
responses, trace shape, and additive metrics so all five runners remain
comparable.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from benchmark_core.schemas import AgentStep, ExperimentConfig, ExperimentInput, LLMCallMetrics


DEBATERS = ("debater_a", "debater_b", "debater_c")
DEBATE_ROUND = "debate_round"
JUDGE = "judge"
DEBATE_COMPONENTS = (*DEBATERS, DEBATE_ROUND, JUDGE)

DEBATER_PERSPECTIVES = {
    "debater_a": "direct evidence-grounded solution",
    "debater_b": "alternative interpretation or solution",
    "debater_c": "critical, conservative, and pragmatic perspective",
}


class DebateProposal(BaseModel):
    """Normalized structured proposal produced by one independent debater."""

    debater_name: str
    perspective: str
    proposal: str
    arguments: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    raw_output: str
    error: str | None = None


class CrossCritique(BaseModel):
    """One explicit cross-review item emitted during the single debate round."""

    target_debater: str
    strength: str
    weakness: str
    disagreement: str


class DebateRoundOutput(BaseModel):
    """Normalized result of the only allowed cross-critique round."""

    round_number: int = 1
    critiques: list[CrossCritique] = Field(default_factory=list)
    consensus_points: list[str] = Field(default_factory=list)
    disagreements: list[str] = Field(default_factory=list)
    strongest_points: list[str] = Field(default_factory=list)
    weakest_points: list[str] = Field(default_factory=list)
    raw_output: str


class JudgeDecision(BaseModel):
    """Typed final decision owned exclusively by the judge."""

    answer: str
    decision_mode: Literal["select", "combine", "reject"]
    selected_proposals: list[str] = Field(default_factory=list)
    rejected_proposals: list[str] = Field(default_factory=list)
    rationale: str
    confidence: float = 0.0
    unresolved_issues: list[str] = Field(default_factory=list)
    raw_output: str


def document_ids(input_data: ExperimentInput) -> list[str]:
    return [document.document_id for document in input_data.documents]


def render_debate_judge_prompt(
    input_data: ExperimentInput,
    component: str,
    *,
    proposals: dict[str, dict[str, Any]] | None = None,
    debate: dict[str, Any] | None = None,
) -> str:
    """Render one canonical prompt for a debater, debate round, or judge."""

    if component not in DEBATE_COMPONENTS:
        raise ValueError(f"Unknown ARCH_08 component: {component}")

    document_blocks = "\n".join(
        f"[{document.document_id}] {document.content}" for document in input_data.documents
    ) or "No documents provided."
    metadata_json = json.dumps(input_data.metadata, ensure_ascii=False, sort_keys=True)
    common = (
        "You are executing ARCH_08_DEBATE_JUDGE for a benchmark.\n"
        f"Debate component: {component}\n"
        "There is exactly one cross-critique round and exactly one final judge. "
        "Do not route, hand off, supervise, plan implicitly, or start an improvement loop.\n\n"
        f"Task type: {input_data.task_type}\n"
        f"Question: {input_data.query}\n"
        f"Input metadata: {metadata_json}\n\n"
        f"Documents:\n{document_blocks}\n\n"
    )

    if component in DEBATERS:
        diversity_rules = {
            "debater_a": (
                "Propose the strongest direct solution grounded in the supplied case and evidence."
            ),
            "debater_b": (
                "Propose a genuinely alternative interpretation or solution. Do not merely echo the most "
                "obvious direct answer."
            ),
            "debater_c": (
                "Take a critical, conservative, and pragmatic perspective. Stress uncertainty, constraints, "
                "and failure modes while still proposing an actionable answer."
            ),
        }
        return common + (
            f"Independent debater responsibility: {diversity_rules[component]}\n"
            "You cannot see any other proposal. Do not critique a proposal that has not yet been produced "
            "and do not make the final decision.\n\n"
            "Return exactly these labeled fields:\n"
            "PROPOSAL=concise proposed answer or interpretation\n"
            "ARGUMENTS=item 1 | item 2 | none\n"
            "EVIDENCE=item 1 | item 2 | none\n"
            "ASSUMPTIONS=item 1 | item 2 | none\n"
            "RISKS=item 1 | item 2 | none\n"
        )

    missing_proposals = [name for name in DEBATERS if name not in (proposals or {})]
    if missing_proposals:
        raise ValueError(f"{component} requires all three proposals; missing: {missing_proposals}")
    proposals_json = json.dumps(proposals, ensure_ascii=False, sort_keys=True)

    if component == DEBATE_ROUND:
        return common + (
            "Conduct the single explicit cross-critique round over all three proposals. Compare them against "
            "one another and the common evidence. For each proposal, state one strength, one weakness, and a "
            "specific disagreement or tension. Do not rewrite a proposal and do not decide the final answer.\n\n"
            "Return exactly these labeled fields:\n"
            "CRITIQUE_DEBATER_A=STRENGTH: ... ; WEAKNESS: ... ; DISAGREEMENT: ...\n"
            "CRITIQUE_DEBATER_B=STRENGTH: ... ; WEAKNESS: ... ; DISAGREEMENT: ...\n"
            "CRITIQUE_DEBATER_C=STRENGTH: ... ; WEAKNESS: ... ; DISAGREEMENT: ...\n"
            "CONSENSUS_POINTS=item 1 | item 2 | none\n"
            "DISAGREEMENTS=item 1 | item 2 | none\n"
            "STRONGEST_POINTS=item 1 | item 2 | none\n"
            "WEAKEST_POINTS=item 1 | item 2 | none\n\n"
            f"Proposals JSON:\n{proposals_json}\n"
        )

    if debate is None:
        raise ValueError("Judge requires the explicit debate round output.")
    debate_json = json.dumps(debate, ensure_ascii=False, sort_keys=True)
    return common + (
        "You are the only component authorized to decide the final answer. Evaluate the original proposals "
        "and the explicit cross-critique round. Select one proposal, combine multiple proposals, or reject all "
        "three. Resolve material disagreements and justify the decision briefly. Do not call another agent or "
        "start a new debate/revision round.\n\n"
        "Return exactly these labeled fields:\n"
        "FINAL_ANSWER=concise final answer\n"
        "DECISION_MODE=select|combine|reject\n"
        "SELECTED_PROPOSALS=debater_a | debater_b | debater_c | none\n"
        "REJECTED_PROPOSALS=debater_a | debater_b | debater_c | none\n"
        "RATIONALE=brief justification grounded in arguments and critiques\n"
        "CONFIDENCE=number from 0.0 to 1.0\n"
        "UNRESOLVED_ISSUES=item 1 | item 2 | none\n\n"
        f"Proposals JSON:\n{proposals_json}\n\n"
        f"Debate round JSON:\n{debate_json}\n"
    )


def detect_debate_component(prompt: str) -> str | None:
    """Return the ARCH_08 component embedded in a canonical prompt."""

    if "ARCH_08_DEBATE_JUDGE" not in prompt:
        return None
    prefix = "Debate component:"
    for line in prompt.splitlines():
        if line.startswith(prefix):
            component = line.removeprefix(prefix).strip().lower()
            return component if component in DEBATE_COMPONENTS else None
    return None


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


def parse_debate_proposal(
    component: str,
    output: str,
    *,
    error: str | None = None,
) -> DebateProposal:
    """Normalize one debater response while retaining the raw output."""

    if component not in DEBATERS:
        raise ValueError(f"Unknown debater: {component}")
    return DebateProposal(
        debater_name=component,
        perspective=DEBATER_PERSPECTIVES[component],
        proposal=_labeled_value(output, "PROPOSAL") or output.strip(),
        arguments=_split_items(_labeled_value(output, "ARGUMENTS")),
        evidence=_split_items(_labeled_value(output, "EVIDENCE")),
        assumptions=_split_items(_labeled_value(output, "ASSUMPTIONS")),
        risks=_split_items(_labeled_value(output, "RISKS")),
        raw_output=output.strip(),
        error=error,
    )


def _parse_critique(output: str, debater: str) -> CrossCritique:
    raw = _labeled_value(output, f"CRITIQUE_{debater.upper()}") or ""
    values: dict[str, str] = {}
    for field in ("STRENGTH", "WEAKNESS", "DISAGREEMENT"):
        marker = f"{field}:"
        start = raw.upper().find(marker)
        if start < 0:
            continue
        start += len(marker)
        following = [
            position
            for other in ("STRENGTH", "WEAKNESS", "DISAGREEMENT")
            if other != field
            and (position := raw.upper().find(f"; {other}:", start)) >= 0
        ]
        end = min(following) if following else len(raw)
        values[field] = raw[start:end].strip(" ;")
    fallback = raw or "No parseable cross-critique was returned."
    return CrossCritique(
        target_debater=debater,
        strength=values.get("STRENGTH", fallback),
        weakness=values.get("WEAKNESS", fallback),
        disagreement=values.get("DISAGREEMENT", fallback),
    )


def parse_debate_round(output: str) -> DebateRoundOutput:
    """Normalize the one-round cross-critique response."""

    return DebateRoundOutput(
        critiques=[_parse_critique(output, debater) for debater in DEBATERS],
        consensus_points=_split_items(_labeled_value(output, "CONSENSUS_POINTS")),
        disagreements=_split_items(_labeled_value(output, "DISAGREEMENTS")),
        strongest_points=_split_items(_labeled_value(output, "STRONGEST_POINTS")),
        weakest_points=_split_items(_labeled_value(output, "WEAKEST_POINTS")),
        raw_output=output.strip(),
    )


def parse_judge_decision(output: str) -> JudgeDecision:
    """Normalize the judge response and constrain selection to known proposals."""

    selected = [item for item in _split_items(_labeled_value(output, "SELECTED_PROPOSALS")) if item in DEBATERS]
    rejected = [item for item in _split_items(_labeled_value(output, "REJECTED_PROPOSALS")) if item in DEBATERS]
    raw_mode = (_labeled_value(output, "DECISION_MODE") or "").lower()
    if raw_mode not in {"select", "combine", "reject"}:
        raw_mode = "combine" if len(selected) > 1 else "select" if selected else "reject"
    if raw_mode == "reject":
        selected = []
    elif not selected:
        selected = ["debater_a"]
    if raw_mode == "select":
        selected = selected[:1]
    elif raw_mode == "combine" and len(selected) < 2:
        selected = list(DEBATERS[:2])

    confidence_text = _labeled_value(output, "CONFIDENCE") or "0"
    try:
        confidence = min(max(float(confidence_text), 0.0), 1.0)
    except ValueError:
        confidence = 0.0
    answer = _labeled_value(output, "FINAL_ANSWER")
    if not answer and "Final Answer:" in output:
        answer = output.split("Final Answer:", maxsplit=1)[1].strip()
    return JudgeDecision(
        answer=answer or output.strip(),
        decision_mode=raw_mode,
        selected_proposals=selected,
        rejected_proposals=rejected,
        rationale=_labeled_value(output, "RATIONALE") or "No parseable rationale was returned.",
        confidence=confidence,
        unresolved_issues=_split_items(_labeled_value(output, "UNRESOLVED_ISSUES")),
        raw_output=output.strip(),
    )


def build_deterministic_debate_output(
    input_data: ExperimentInput,
    component: str,
) -> str:
    """Build reproducible local outputs for all five ARCH_08 calls."""

    sources = document_ids(input_data)
    source_text = ", ".join(sources) if sources else "no documents"
    first_fragment = (
        input_data.documents[0].content.strip().replace("\n", " ")[:180]
        if input_data.documents
        else "No documentary evidence was supplied."
    )
    if component == "debater_a":
        return (
            "PROPOSAL=Answer the query directly from the supplied evidence and state the evidence boundary.\n"
            f"ARGUMENTS=The common input directly addresses the case | Traceability is preserved through {source_text}\n"
            f"EVIDENCE={first_fragment} | Source ids: {source_text}\n"
            "ASSUMPTIONS=The supplied documents are the complete benchmark evidence\n"
            "RISKS=The direct interpretation may miss a plausible alternative"
        )
    if component == "debater_b":
        return (
            "PROPOSAL=Present a bounded alternative interpretation alongside the primary answer when ambiguity remains.\n"
            "ARGUMENTS=The wording can support more than one reasonable reading | Explicit alternatives improve robustness\n"
            f"EVIDENCE=The alternative is checked against the same source ids: {source_text}\n"
            "ASSUMPTIONS=Ambiguity is material enough to disclose\n"
            "RISKS=Multiple interpretations can reduce concision"
        )
    if component == "debater_c":
        return (
            "PROPOSAL=Use a conservative answer that separates confirmed facts, assumptions, and unresolved limitations.\n"
            "ARGUMENTS=Uncertainty should constrain confidence | Pragmatic caveats prevent unsupported claims\n"
            f"EVIDENCE=Only these benchmark sources are available: {source_text}\n"
            "ASSUMPTIONS=Missing context cannot be inferred safely\n"
            "RISKS=Excessive caution may make the answer less decisive"
        )
    if component == DEBATE_ROUND:
        return (
            "CRITIQUE_DEBATER_A=STRENGTH: direct and evidence-grounded ; WEAKNESS: may understate ambiguity ; "
            "DISAGREEMENT: debater_b and debater_c require stronger qualification\n"
            "CRITIQUE_DEBATER_B=STRENGTH: exposes a reasonable alternative ; WEAKNESS: adds verbosity ; "
            "DISAGREEMENT: debater_a favors a single direct answer\n"
            "CRITIQUE_DEBATER_C=STRENGTH: controls uncertainty and unsupported claims ; WEAKNESS: may be overly cautious ; "
            "DISAGREEMENT: debater_a assigns more confidence to the direct interpretation\n"
            "CONSENSUS_POINTS=Use the supplied evidence | State material limitations\n"
            "DISAGREEMENTS=Whether to include an alternative interpretation | How conservative confidence should be\n"
            "STRONGEST_POINTS=debater_a traceability | debater_b ambiguity handling | debater_c risk control\n"
            "WEAKEST_POINTS=debater_a may overcommit | debater_b may dilute focus | debater_c may undercommit"
        )
    if component == JUDGE:
        return (
            "FINAL_ANSWER=Give the direct evidence-grounded answer, qualify its evidence limits, and mention the "
            "alternative only when the ambiguity materially changes the conclusion.\n"
            "DECISION_MODE=combine\n"
            "SELECTED_PROPOSALS=debater_a | debater_b | debater_c\n"
            "REJECTED_PROPOSALS=none\n"
            "RATIONALE=The direct proposal is clearest, while the alternative and conservative critiques add robustness.\n"
            "CONFIDENCE=0.82\n"
            "UNRESOLVED_ISSUES=The supplied benchmark evidence may not eliminate every interpretation"
        )
    raise ValueError(f"Unknown ARCH_08 component: {component}")


def make_debate_step(
    *,
    step_id: int,
    component: str,
    actor: str,
    prompt: str,
    output: dict[str, Any],
    llm_call_ids: list[str],
    started_at: datetime,
    finished_at: datetime,
    framework_primitive: str,
    proposals: dict[str, dict[str, Any]] | None = None,
    debate: dict[str, Any] | None = None,
    parallel_proposals: bool = True,
    error: str | None = None,
) -> AgentStep:
    """Create one canonical ARCH_08 trace entry."""

    if component in DEBATERS:
        step_type = "debate_proposal_llm_call"
        depends_on: list[str] = []
        phase = "proposal"
    elif component == DEBATE_ROUND:
        step_type = "debate_round_llm_call"
        depends_on = list(DEBATERS)
        phase = "cross_critique"
    elif component == JUDGE:
        step_type = "debate_judge_llm_call"
        depends_on = [*DEBATERS, DEBATE_ROUND]
        phase = "judgment"
    else:
        raise ValueError(f"Unknown ARCH_08 component: {component}")

    input_payload: dict[str, Any] = {
        "prompt": prompt,
        "component": component,
        "depends_on": depends_on,
    }
    if proposals is not None:
        input_payload["proposals"] = proposals
    if debate is not None:
        input_payload["debate_round"] = debate
    return AgentStep(
        step_id=step_id,
        name=component,
        step_type=step_type,
        actor=actor,
        input_data=input_payload,
        output_data=output,
        llm_call_ids=llm_call_ids,
        started_at=started_at,
        finished_at=finished_at,
        error=error,
        metadata={
            "architecture": "ARCH_08_DEBATE_JUDGE",
            "phase": phase,
            "debate_round_number": 1 if component == DEBATE_ROUND else None,
            "framework_primitive": framework_primitive,
            "parallel_proposals": parallel_proposals,
            "latency_ms": max((finished_at - started_at).total_seconds() * 1000, 0.0),
        },
    )


def build_debate_execution_metadata(
    *,
    steps: list[AgentStep],
    llm_calls: list[LLMCallMetrics],
    debate: DebateRoundOutput,
    decision: JudgeDecision,
    framework_primitive: str,
    parallel_proposals: bool,
) -> dict[str, Any]:
    """Aggregate per-component latency, calls, tokens, errors, and judge choice."""

    calls_by_id = {call.call_id: call for call in llm_calls}

    def metrics_for(component: str) -> dict[str, Any]:
        step = next((item for item in steps if item.name == component), None)
        calls = [
            calls_by_id[call_id]
            for call_id in (step.llm_call_ids if step else [])
            if call_id in calls_by_id
        ]
        return {
            "latency_ms": (
                max((step.finished_at - step.started_at).total_seconds() * 1000, 0.0)
                if step and step.started_at and step.finished_at
                else 0.0
            ),
            "llm_call_count": len(calls),
            "input_tokens": sum(call.token_usage.input_tokens for call in calls),
            "output_tokens": sum(call.token_usage.output_tokens for call in calls),
            "total_tokens": sum(call.token_usage.total_tokens for call in calls),
            "error": step.error if step else "missing_step",
        }

    component_metrics = {component: metrics_for(component) for component in DEBATE_COMPONENTS}
    return {
        "proposal_count": len(DEBATERS),
        "debate_round_count": 1,
        "critique_count": len(debate.critiques),
        "disagreement_count": len(debate.disagreements),
        "decision_mode": decision.decision_mode,
        "selected_proposals": decision.selected_proposals,
        "rejected_proposals": decision.rejected_proposals,
        "judge_combined_proposals": decision.decision_mode == "combine",
        "judge_selected_single_proposal": decision.decision_mode == "select",
        "parallel_proposals": parallel_proposals,
        "fallback_sequential": False,
        "framework_primitive": framework_primitive,
        "component_metrics": component_metrics,
        "component_errors": {
            component: values["error"]
            for component, values in component_metrics.items()
            if values["error"] is not None
        },
    }


def build_debate_structured_output(
    *,
    input_data: ExperimentInput,
    config: ExperimentConfig,
    proposals: dict[str, dict[str, Any]],
    debate_output: str,
    judge_output: str,
    steps: list[AgentStep],
    llm_calls: list[LLMCallMetrics],
    framework_execution: str,
    framework_primitive: str,
    parallel_proposals: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Build the final common output shape for every ARCH_08 implementation."""

    debate = parse_debate_round(debate_output)
    decision = parse_judge_decision(judge_output)
    execution = build_debate_execution_metadata(
        steps=steps,
        llm_calls=llm_calls,
        debate=debate,
        decision=decision,
        framework_primitive=framework_primitive,
        parallel_proposals=parallel_proposals,
    )
    structured_output = {
        "answer": decision.answer,
        "mode": f"{config.model_provider}_debate_judge",
        "proposals": proposals,
        "debate_round": debate.model_dump(),
        "judge": decision.model_dump(),
        "number_of_proposals": len(proposals),
        "number_of_debate_rounds": 1,
        "critique_count": execution["critique_count"],
        "disagreement_count": execution["disagreement_count"],
        "decision_mode": decision.decision_mode,
        "selected_proposals": decision.selected_proposals,
        "judge_combined_proposals": execution["judge_combined_proposals"],
        "judge_selected_single_proposal": execution["judge_selected_single_proposal"],
        "document_ids": document_ids(input_data),
        "debate_execution": execution,
        "framework_execution": framework_execution,
    }
    return decision.answer, structured_output
