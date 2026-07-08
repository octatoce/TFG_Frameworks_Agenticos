"""Pure shared semantics for ARCH_05_HANDOFF_SWARM."""

from __future__ import annotations

import os
import re
from typing import Any

from benchmark_core.schemas import ExperimentConfig, ExperimentInput


HANDOFF_AGENTS = [
    "data_specialist",
    "reasoning_specialist",
    "validation_specialist",
    "synthesis_specialist",
]

AGENT_DISPLAY_NAMES = {
    "data_specialist": "DataSpecialist",
    "reasoning_specialist": "ReasoningSpecialist",
    "validation_specialist": "ValidationSpecialist",
    "synthesis_specialist": "SynthesisSpecialist",
}

DISPLAY_TO_AGENT = {display: agent for agent, display in AGENT_DISPLAY_NAMES.items()}

VALID_HANDOFF_ACTIONS = {"handoff", "finalize"}


def get_handoff_limits(config: ExperimentConfig) -> dict[str, int]:
    """Resolve ARCH_05 limits without changing the public config schema."""

    def resolve(name: str, env_name: str, default: int) -> int:
        configured = config.metadata.get(name)
        if configured is None:
            configured = os.environ.get(env_name)
        try:
            return max(1, int(configured)) if configured is not None else default
        except (TypeError, ValueError):
            return default

    return {
        "max_handoffs": resolve("max_handoffs", "MAX_HANDOFFS", 4),
        "max_agent_invocations": resolve("max_agent_invocations", "MAX_AGENT_INVOCATIONS", 6),
        "max_consecutive_visits_per_agent": resolve(
            "max_consecutive_visits_per_agent",
            "MAX_CONSECUTIVE_VISITS_PER_AGENT",
            2,
        ),
    }


def choose_initial_handoff_agent(input_data: ExperimentInput) -> str:
    """Pick only the first active agent; this is not a route plan."""

    query_text = f"{input_data.query} {input_data.task_type}".lower()
    if "direct" in query_text and not input_data.documents:
        return "reasoning_specialist"
    if input_data.documents:
        return "data_specialist"
    return "reasoning_specialist"


def document_ids(input_data: ExperimentInput) -> list[str]:
    return [document.document_id for document in input_data.documents]


def extract_final_answer(response: str) -> str:
    marker = "Final Answer:"
    if marker not in response:
        return response.strip()
    return response.split(marker, maxsplit=1)[1].strip()


def render_handoff_swarm_prompt(
    input_data: ExperimentInput,
    active_agent: str,
    state: dict[str, Any],
) -> str:
    """Render the canonical ARCH_05 prompt for one active specialist."""

    document_blocks = "\n".join(
        f"[{document.document_id}] {document.content}" for document in input_data.documents
    )
    if not document_blocks:
        document_blocks = "No documents provided."

    allowed_targets = [
        agent
        for agent in HANDOFF_AGENTS
        if agent != active_agent
    ]
    return (
        "You are executing ARCH_05_HANDOFF_SWARM for a benchmark.\n"
        f"Active agent: {active_agent}\n"
        f"Active agent display name: {AGENT_DISPLAY_NAMES[active_agent]}\n"
        "There is no central supervisor. The active specialist must decide whether to finalize or hand off control.\n\n"
        "Use exactly this format:\n"
        "ACTION=handoff|finalize\n"
        "TARGET_AGENT=agent_id_or_none\n"
        "REASON=short reason\n"
        "TASK=task for next agent or none\n"
        "CONTEXT_SUMMARY=only necessary transferred context\n"
        "FINAL_OUTPUT=final answer text or none\n"
        "CONFIDENCE=0.0_to_1.0\n"
        "EVIDENCE=short evidence summary\n"
        "LIMITATIONS=short limitations summary\n\n"
        f"Allowed target agents: {', '.join(allowed_targets)}.\n"
        f"Task type: {input_data.task_type}\n"
        f"Question: {input_data.query}\n\n"
        f"Active agent history: {state.get('active_agent_history') or []}\n"
        f"Handoff history: {state.get('handoff_history') or []}\n"
        f"Partial results: {state.get('partial_results') or []}\n"
        f"Context transferred: {state.get('context_transferred') or 'None'}\n"
        f"Number of handoffs: {state.get('number_of_handoffs', 0)}\n"
        f"Number of agent invocations: {state.get('number_of_agent_invocations', 0)}\n"
        f"Warnings: {state.get('warnings') or []}\n\n"
        f"Documents:\n{document_blocks}\n"
    )


def parse_handoff_decision(output: str) -> dict[str, Any]:
    """Parse an ARCH_05 handoff/finalize decision."""

    def parse_line(name: str) -> str | None:
        match = re.search(rf"^{name}\s*=\s*(.*)$", output, flags=re.MULTILINE)
        return match.group(1).strip() if match else None

    action = (parse_line("ACTION") or "").lower()
    if action not in VALID_HANDOFF_ACTIONS:
        lowered = output.lower()
        if "handoff" in lowered or "transfer" in lowered:
            action = "handoff"
        elif "final" in lowered:
            action = "finalize"

    target_agent = parse_line("TARGET_AGENT") or None
    if target_agent in {"none", "None", ""}:
        target_agent = None

    confidence_text = parse_line("CONFIDENCE") or "0.0"
    try:
        confidence = max(0.0, min(float(confidence_text), 1.0))
    except ValueError:
        confidence = 0.0

    return {
        "action": action,
        "target_agent": target_agent,
        "reason": parse_line("REASON") or "No reason provided.",
        "task": parse_line("TASK") or "none",
        "context_summary": parse_line("CONTEXT_SUMMARY") or "none",
        "final_output": parse_line("FINAL_OUTPUT") or "none",
        "confidence": confidence,
        "evidence": parse_line("EVIDENCE") or "none",
        "limitations": parse_line("LIMITATIONS") or "none",
        "raw_decision": output,
    }


def is_valid_handoff_decision(decision: dict[str, Any]) -> bool:
    if decision["action"] == "finalize":
        return bool(decision.get("final_output")) and decision["final_output"] != "none"
    if decision["action"] == "handoff":
        return decision.get("target_agent") in HANDOFF_AGENTS
    return False


def build_fallback_answer(input_data: ExperimentInput, partial_results: list[dict[str, Any]]) -> str:
    source_text = ", ".join(document_ids(input_data)) if input_data.documents else "no-documents"
    if partial_results:
        latest = str(partial_results[-1].get("decision", {}).get("final_output", ""))
        if latest and latest != "none":
            return extract_final_answer(latest)
    return (
        f"Fallback ARCH_05 answer for case {input_data.case_id}. "
        f"The swarm finalized with available context. Sources used: {source_text}."
    )


def _prompt_has_agent(prompt: str, agent: str) -> bool:
    history_prefixes = (
        "Active agent history:",
        "Handoff history:",
        "Partial results:",
    )
    for line in prompt.splitlines():
        if line.startswith(history_prefixes) and (
            agent in line or AGENT_DISPLAY_NAMES[agent] in line
        ):
            return True
    return False


def _needs_validation(input_data: ExperimentInput) -> bool:
    query_text = f"{input_data.query} {input_data.task_type}".lower()
    return any(
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


def build_deterministic_handoff_output(
    input_data: ExperimentInput,
    active_agent: str,
    prompt: str,
) -> str:
    """Build a stable local ARCH_05 decision for tests without external APIs."""

    source_text = ", ".join(document_ids(input_data)) if input_data.documents else "no-documents"
    evidence = (
        input_data.documents[0].content.strip().replace("\n", " ")[:240]
        if input_data.documents
        else "No document context was provided."
    )
    query_text = input_data.query.lower()

    if "invalid" in query_text:
        return (
            "ACTION=handoff\n"
            "TARGET_AGENT=unknown_specialist\n"
            "REASON=Intentional invalid deterministic decision for fallback coverage.\n"
            "TASK=none\n"
            "CONTEXT_SUMMARY=invalid target requested\n"
            "FINAL_OUTPUT=none\n"
            "CONFIDENCE=0.1\n"
            "EVIDENCE=none\n"
            "LIMITATIONS=invalid decision"
        )

    if "cycle" in query_text:
        target = "reasoning_specialist" if active_agent == "data_specialist" else "data_specialist"
        return (
            "ACTION=handoff\n"
            f"TARGET_AGENT={target}\n"
            "REASON=Intentional deterministic cycle for limit coverage.\n"
            "TASK=Continue the cyclic test handoff.\n"
            "CONTEXT_SUMMARY=cycle coverage context\n"
            "FINAL_OUTPUT=none\n"
            "CONFIDENCE=0.4\n"
            f"EVIDENCE=sources={source_text}\n"
            "LIMITATIONS=cycle test"
        )

    if "return" in query_text:
        if active_agent == "data_specialist":
            if _prompt_has_agent(prompt, "reasoning_specialist"):
                return (
                    "ACTION=handoff\n"
                    "TARGET_AGENT=synthesis_specialist\n"
                    "REASON=Returned context is now sufficient for synthesis.\n"
                    "TASK=Synthesize the returned context into a final answer.\n"
                    "CONTEXT_SUMMARY=DataSpecialist received control back and approved synthesis.\n"
                    "FINAL_OUTPUT=none\n"
                    "CONFIDENCE=0.79\n"
                    f"EVIDENCE=sources={source_text}\n"
                    "LIMITATIONS=Return-path deterministic test."
                )
            return (
                "ACTION=handoff\n"
                "TARGET_AGENT=reasoning_specialist\n"
                "REASON=Evidence was extracted and reasoning should inspect it.\n"
                "TASK=Analyze evidence and return control if more data context is required.\n"
                f"CONTEXT_SUMMARY=DataSpecialist found sources={source_text}; evidence={evidence}\n"
                "FINAL_OUTPUT=none\n"
                "CONFIDENCE=0.77\n"
                f"EVIDENCE={evidence}\n"
                "LIMITATIONS=Return-path deterministic test."
            )
        if active_agent == "reasoning_specialist":
            return (
                "ACTION=handoff\n"
                "TARGET_AGENT=data_specialist\n"
                "REASON=ReasoningSpecialist needs the DataSpecialist to refine evidence.\n"
                "TASK=Refine evidence before synthesis.\n"
                "CONTEXT_SUMMARY=Reasoning requested a return to data context.\n"
                "FINAL_OUTPUT=none\n"
                "CONFIDENCE=0.71\n"
                f"EVIDENCE=sources={source_text}\n"
                "LIMITATIONS=Return-path deterministic test."
            )

    if active_agent == "data_specialist":
        return (
            "ACTION=handoff\n"
            "TARGET_AGENT=reasoning_specialist\n"
            "REASON=Evidence was extracted and reasoning should continue.\n"
            "TASK=Analyze the evidence and decide whether validation or synthesis is needed.\n"
            f"CONTEXT_SUMMARY=DataSpecialist found sources={source_text}; evidence={evidence}\n"
            "FINAL_OUTPUT=none\n"
            "CONFIDENCE=0.78\n"
            f"EVIDENCE={evidence}\n"
            "LIMITATIONS=Evidence is limited to provided documents."
        )

    if active_agent == "reasoning_specialist":
        if "direct" in query_text and not input_data.documents:
            return (
                "ACTION=finalize\n"
                "TARGET_AGENT=none\n"
                "REASON=ReasoningSpecialist can answer directly without additional context.\n"
                "TASK=none\n"
                "CONTEXT_SUMMARY=no handoff required\n"
                f"FINAL_OUTPUT=Final Answer: Direct ARCH_05 answer for {input_data.case_id}. Sources used: {source_text}.\n"
                "CONFIDENCE=0.72\n"
                "EVIDENCE=No documents were provided.\n"
                "LIMITATIONS=Direct answer has no documentary evidence."
            )
        if _needs_validation(input_data) and not _prompt_has_agent(prompt, "validation_specialist"):
            return (
                "ACTION=handoff\n"
                "TARGET_AGENT=validation_specialist\n"
                "REASON=The task benefits from risk and consistency validation.\n"
                "TASK=Validate the reasoning and identify contradictions or missing evidence.\n"
                "CONTEXT_SUMMARY=Reasoning completed; validation requested.\n"
                "FINAL_OUTPUT=none\n"
                "CONFIDENCE=0.74\n"
                f"EVIDENCE=sources={source_text}\n"
                "LIMITATIONS=Needs validation before final answer."
            )
        if not _prompt_has_agent(prompt, "synthesis_specialist"):
            return (
                "ACTION=handoff\n"
                "TARGET_AGENT=synthesis_specialist\n"
                "REASON=Reasoning is sufficient and synthesis should produce the final answer.\n"
                "TASK=Synthesize the final answer from evidence and reasoning.\n"
                "CONTEXT_SUMMARY=ReasoningSpecialist approved the evidence path.\n"
                "FINAL_OUTPUT=none\n"
                "CONFIDENCE=0.82\n"
                f"EVIDENCE=sources={source_text}\n"
                "LIMITATIONS=No independent validation was requested."
            )

    if active_agent == "validation_specialist":
        return (
            "ACTION=handoff\n"
            "TARGET_AGENT=synthesis_specialist\n"
            "REASON=Validation found no blocking contradiction; synthesis can finalize.\n"
            "TASK=Produce the final answer with limitations.\n"
            "CONTEXT_SUMMARY=Validation confidence high; no contradiction detected.\n"
            "FINAL_OUTPUT=none\n"
            "CONFIDENCE=0.86\n"
            f"EVIDENCE=sources={source_text}\n"
            "LIMITATIONS=Validation is based only on provided benchmark context."
        )

    if active_agent == "synthesis_specialist":
        return (
            "ACTION=finalize\n"
            "TARGET_AGENT=none\n"
            "REASON=SynthesisSpecialist has enough context to finalize.\n"
            "TASK=none\n"
            "CONTEXT_SUMMARY=Final answer produced by active specialist.\n"
            "FINAL_OUTPUT=Final Answer: The handoff swarm transferred control between specialists and finalized "
            f"a concise answer grounded in the available context. Sources used: {source_text}.\n"
            "CONFIDENCE=0.84\n"
            f"EVIDENCE=sources={source_text}; excerpt={evidence}\n"
            "LIMITATIONS=Deterministic local run for benchmark instrumentation."
        )

    return (
        "ACTION=finalize\n"
        "TARGET_AGENT=none\n"
        "REASON=Unknown agent fallback.\n"
        "TASK=none\n"
        "CONTEXT_SUMMARY=unknown agent\n"
        f"FINAL_OUTPUT=Final Answer: {build_fallback_answer(input_data, [])}\n"
        "CONFIDENCE=0.2\n"
        "EVIDENCE=none\n"
        "LIMITATIONS=unknown agent fallback"
    )
