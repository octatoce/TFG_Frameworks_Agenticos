"""Shared ARCH_04 supervisor/workers orchestration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from benchmark_core.llm_wrapper import (
    LLMCallRecord,
    SUPERVISOR_WORKERS,
    parse_supervisor_action,
    parse_supervisor_plan,
    render_supervisor_workers_prompt,
)
from benchmark_core.schemas import AgentStep, ExperimentConfig, ExperimentInput, LLMCallMetrics
from benchmark_core.tracing import utc_now


PromptRunner = Callable[[str, int, str, str | None], LLMCallRecord]


@dataclass
class SupervisorWorkersResult:
    final_answer: str
    structured_output: dict[str, Any]
    steps: list[AgentStep]
    llm_calls: list[LLMCallMetrics]


def get_max_supervisor_iterations(config: ExperimentConfig) -> int:
    """Resolve the bounded ARCH_04 iteration limit without changing the public schema."""

    configured = config.metadata.get("max_supervisor_iterations")
    if configured is None:
        configured = os.environ.get("MAX_SUPERVISOR_ITERATIONS")
    try:
        return max(1, int(configured)) if configured is not None else 3
    except (TypeError, ValueError):
        return 3


def extract_final_answer(response: str) -> str:
    marker = "Final Answer:"
    if marker not in response:
        return response.strip()
    return response.split(marker, maxsplit=1)[1].strip()


def document_ids(input_data: ExperimentInput) -> list[str]:
    return [document.document_id for document in input_data.documents]


def run_supervisor_workers_loop(
    *,
    input_data: ExperimentInput,
    config: ExperimentConfig,
    framework_execution: str,
    actor_prefix: str,
    run_prompt: PromptRunner,
) -> SupervisorWorkersResult:
    """Execute ARCH_04 as a bounded centralized supervisor loop."""

    max_iterations = get_max_supervisor_iterations(config)
    state: dict[str, Any] = {
        "input_data": input_data,
        "plan": None,
        "worker_outputs": [],
        "iterations": 0,
        "max_supervisor_iterations": max_iterations,
        "workers_executed": [],
        "revisions_requested": 0,
        "accepted_worker_outputs": [],
        "rejected_worker_outputs": [],
        "warnings": [],
        "stop_reason": None,
    }
    steps: list[AgentStep] = []
    llm_calls: list[LLMCallMetrics] = []

    def next_step_id() -> int:
        return len(steps) + 1

    def record_step(
        *,
        name: str,
        step_type: str,
        prompt: str,
        output_data: dict[str, Any],
        call_record: LLMCallRecord,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        step_started_at = utc_now()
        llm_calls.append(call_record.metrics)
        steps.append(
            AgentStep(
                step_id=len(steps) + 1,
                name=name,
                step_type=step_type,
                actor=f"{actor_prefix}.{name}",
                input_data={"prompt": prompt},
                output_data=output_data,
                llm_call_ids=[call_record.metrics.call_id],
                started_at=step_started_at,
                finished_at=utc_now(),
                metadata={
                    "architecture": "ARCH_04_SUPERVISOR_WORKERS",
                    **(metadata or {}),
                },
            )
        )

    plan_prompt = render_supervisor_workers_prompt(input_data, "supervisor_plan", state)
    plan_record = run_prompt(plan_prompt, next_step_id(), "supervisor_plan", None)
    plan = parse_supervisor_plan(plan_record.response.strip())
    state["plan"] = plan
    workers_to_run = list(plan["workers_to_run"])
    workers_not_used = [worker for worker in SUPERVISOR_WORKERS if worker not in workers_to_run]
    record_step(
        name="supervisor_plan",
        step_type="supervisor_plan_llm_call",
        prompt=plan_prompt,
        output_data={"supervisor_plan": plan},
        call_record=plan_record,
        metadata={"supervisor_role": "planner"},
    )

    while state["iterations"] < max_iterations:
        decision_prompt = render_supervisor_workers_prompt(input_data, "supervisor_decision", state)
        decision_record = run_prompt(decision_prompt, next_step_id(), "supervisor_decision", None)
        decision = parse_supervisor_action(decision_record.response.strip())

        reviewed_output = state["worker_outputs"][-1] if state["worker_outputs"] else None
        if reviewed_output is not None:
            reviewed_worker = reviewed_output["worker_name"]
            if decision["accepted"] and reviewed_worker not in state["accepted_worker_outputs"]:
                state["accepted_worker_outputs"].append(reviewed_worker)
            elif decision["needs_revision"] and reviewed_worker not in state["rejected_worker_outputs"]:
                state["rejected_worker_outputs"].append(reviewed_worker)

        record_step(
            name="supervisor_decision",
            step_type="supervisor_review_llm_call",
            prompt=decision_prompt,
            output_data={
                "decision": decision,
                "reviewed_worker_output": reviewed_output,
                "accepted_worker_outputs": list(state["accepted_worker_outputs"]),
                "rejected_worker_outputs": list(state["rejected_worker_outputs"]),
            },
            call_record=decision_record,
            metadata={
                "supervisor_role": "reviewer",
                "supervisor_iteration": state["iterations"],
            },
        )

        action = str(decision["action"])
        if action == "finalize":
            state["stop_reason"] = decision["stop_reason"] if decision["stop_reason"] != "none" else "supervisor_finalized"
            break

        worker_name = decision["worker_name"]
        if worker_name not in SUPERVISOR_WORKERS:
            state["warnings"].append("Invalid supervisor action. Finalizing with available outputs.")
            state["stop_reason"] = "invalid_supervisor_action"
            break

        is_revision = action == "request_revision"
        if is_revision:
            state["revisions_requested"] += 1
        elif action != "run_worker":
            state["warnings"].append(f"Unsupported supervisor action '{action}'. Finalizing with available outputs.")
            state["stop_reason"] = "unsupported_supervisor_action"
            break

        worker_prompt = render_supervisor_workers_prompt(
            input_data,
            "worker",
            state,
            worker_name=worker_name,
            task=str(decision["task"]),
            revision_instructions=str(decision["revision_instructions"]) if is_revision else None,
        )
        worker_record = run_prompt(worker_prompt, next_step_id(), "worker", str(worker_name))
        worker_output = {
            "worker_name": worker_name,
            "task": decision["task"],
            "output": worker_record.response.strip(),
            "revision": is_revision,
            "iteration": state["iterations"],
        }
        state["worker_outputs"].append(worker_output)
        state["workers_executed"].append(worker_name)
        record_step(
            name=str(worker_name),
            step_type="worker_llm_call",
            prompt=worker_prompt,
            output_data={"worker_output": worker_output},
            call_record=worker_record,
            metadata={
                "worker_role": worker_name,
                "supervisor_iteration": state["iterations"],
                "revision": is_revision,
            },
        )
        state["iterations"] += 1
    else:
        state["stop_reason"] = "max_supervisor_iterations_reached"

    if state["stop_reason"] is None:
        state["stop_reason"] = "supervisor_finalized"

    finalize_prompt = render_supervisor_workers_prompt(input_data, "supervisor_finalize", state)
    finalize_record = run_prompt(finalize_prompt, next_step_id(), "supervisor_finalize", None)
    final_answer = extract_final_answer(finalize_record.response.strip())
    record_step(
        name="supervisor_finalize",
        step_type="supervisor_finalize_llm_call",
        prompt=finalize_prompt,
        output_data={
            "final_output": finalize_record.response.strip(),
            "stop_reason": state["stop_reason"],
            "warnings": list(state["warnings"]),
        },
        call_record=finalize_record,
        metadata={"supervisor_role": "finalizer"},
    )

    structured_output = {
        "answer": final_answer,
        "mode": f"{config.model_provider}_supervisor_workers",
        "supervisor_plan": plan,
        "worker_outputs": list(state["worker_outputs"]),
        "workers_executed": list(state["workers_executed"]),
        "number_of_workers_executed": len(state["workers_executed"]),
        "workers_used": sorted(set(state["workers_executed"]), key=SUPERVISOR_WORKERS.index),
        "workers_not_used": workers_not_used,
        "supervisor_iterations": state["iterations"],
        "max_supervisor_iterations": max_iterations,
        "revisions_requested": state["revisions_requested"],
        "accepted_worker_outputs": list(state["accepted_worker_outputs"]),
        "rejected_worker_outputs": list(state["rejected_worker_outputs"]),
        "stop_reason": state["stop_reason"],
        "warnings": list(state["warnings"]),
        "document_ids": document_ids(input_data),
        "framework_execution": framework_execution,
    }
    return SupervisorWorkersResult(
        final_answer=final_answer,
        structured_output=structured_output,
        steps=steps,
        llm_calls=llm_calls,
    )
