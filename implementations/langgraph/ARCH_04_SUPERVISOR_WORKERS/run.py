"""LangGraph implementation for ARCH_04_SUPERVISOR_WORKERS."""

from __future__ import annotations

from typing import TypedDict

from benchmark_core.llm_wrapper import (
    SUPERVISOR_WORKERS,
    parse_supervisor_action,
    parse_supervisor_plan,
    render_supervisor_workers_prompt,
)
from benchmark_core.schemas import AgentStep, ExperimentConfig, ExperimentInput
from benchmark_core.tracing import utc_now
from implementations.langgraph.utils_langgraph import (
    LangGraphRunContext,
    LangGraphRunOutput,
    complete_llm_step,
    invoke_with_resource_monitor,
    langgraph_architecture_runner,
    next_step_id,
)
from implementations.supervisor_workers_common import (
    document_ids,
    extract_final_answer,
    get_max_supervisor_iterations,
)


class SupervisorState(TypedDict, total=False):
    plan: dict[str, object] | None
    worker_outputs: list[dict[str, object]]
    iterations: int
    max_supervisor_iterations: int
    workers_executed: list[str]
    revisions_requested: int
    accepted_worker_outputs: list[str]
    rejected_worker_outputs: list[str]
    next_action: dict[str, object] | None
    stop_reason: str | None
    warnings: list[str]
    steps: list[object]
    llm_calls: list[object]
    final_output: str | None


@langgraph_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: LangGraphRunContext,
) -> LangGraphRunOutput:
    """Execute a bounded centralized supervisor graph through LangGraph."""

    from langgraph.graph import END, StateGraph

    def append_step(
        state: SupervisorState,
        *,
        name: str,
        step_type: str,
        prompt: str,
        output_data: dict[str, object],
        call_record: object,
        metadata: dict[str, object] | None = None,
    ) -> SupervisorState:
        step_started_at = utc_now()
        steps = [
            *state.get("steps", []),
            AgentStep(
                step_id=len(state.get("steps", [])) + 1,
                name=name,
                step_type=step_type,
                actor=f"langgraph.{name}_node",
                input_data={"prompt": prompt},
                output_data=output_data,
                llm_call_ids=[call_record.metrics.call_id],
                started_at=step_started_at,
                finished_at=utc_now(),
                metadata={
                    "architecture": "ARCH_04_SUPERVISOR_WORKERS",
                    **(metadata or {}),
                },
            ),
        ]
        return {
            **state,
            "steps": steps,
            "llm_calls": [*state.get("llm_calls", []), call_record.metrics],
        }

    def supervisor_plan_node(state: SupervisorState) -> SupervisorState:
        step_id = next_step_id(state)
        prompt = render_supervisor_workers_prompt(input_data, "supervisor_plan", dict(state))
        call_record = complete_llm_step(
            llm=context.llm,
            input_data=input_data,
            config=config,
            prompt=prompt,
            step_id=step_id,
        )
        plan = parse_supervisor_plan(call_record.response.strip())
        updated = {
            **state,
            "plan": plan,
        }
        return append_step(
            updated,
            name="supervisor_plan",
            step_type="supervisor_plan_llm_call",
            prompt=prompt,
            output_data={"supervisor_plan": plan},
            call_record=call_record,
            metadata={"supervisor_role": "planner"},
        )

    def supervisor_decision_node(state: SupervisorState) -> SupervisorState:
        if state["iterations"] >= state["max_supervisor_iterations"]:
            return {
                **state,
                "next_action": {"action": "finalize", "stop_reason": "max_supervisor_iterations_reached"},
                "stop_reason": "max_supervisor_iterations_reached",
            }

        step_id = next_step_id(state)
        prompt = render_supervisor_workers_prompt(input_data, "supervisor_decision", dict(state))
        call_record = complete_llm_step(
            llm=context.llm,
            input_data=input_data,
            config=config,
            prompt=prompt,
            step_id=step_id,
        )
        decision = parse_supervisor_action(call_record.response.strip())
        accepted_outputs = list(state["accepted_worker_outputs"])
        rejected_outputs = list(state["rejected_worker_outputs"])
        reviewed_output = state["worker_outputs"][-1] if state["worker_outputs"] else None
        if reviewed_output is not None:
            reviewed_worker = reviewed_output["worker_name"]
            if decision["accepted"] and reviewed_worker not in accepted_outputs:
                accepted_outputs.append(reviewed_worker)
            elif decision["needs_revision"] and reviewed_worker not in rejected_outputs:
                rejected_outputs.append(reviewed_worker)

        updated = {
            **state,
            "next_action": decision,
            "accepted_worker_outputs": accepted_outputs,
            "rejected_worker_outputs": rejected_outputs,
            "stop_reason": (
                decision["stop_reason"]
                if decision["action"] == "finalize" and decision["stop_reason"] != "none"
                else state.get("stop_reason")
            ),
        }
        return append_step(
            updated,
            name="supervisor_decision",
            step_type="supervisor_review_llm_call",
            prompt=prompt,
            output_data={
                "decision": decision,
                "reviewed_worker_output": reviewed_output,
                "accepted_worker_outputs": accepted_outputs,
                "rejected_worker_outputs": rejected_outputs,
            },
            call_record=call_record,
            metadata={
                "supervisor_role": "reviewer",
                "supervisor_iteration": state["iterations"],
            },
        )

    def worker_node(state: SupervisorState, worker_name: str) -> SupervisorState:
        decision = state["next_action"] or {}
        is_revision = decision.get("action") == "request_revision"
        revisions_requested = state["revisions_requested"] + (1 if is_revision else 0)
        prompt = render_supervisor_workers_prompt(
            input_data,
            "worker",
            {
                **state,
                "revisions_requested": revisions_requested,
            },
            worker_name=worker_name,
            task=str(decision.get("task") or "none"),
            revision_instructions=str(decision.get("revision_instructions") or "none") if is_revision else None,
        )
        step_id = next_step_id(state)
        call_record = complete_llm_step(
            llm=context.llm,
            input_data=input_data,
            config=config,
            prompt=prompt,
            step_id=step_id,
        )
        worker_output = {
            "worker_name": worker_name,
            "task": decision.get("task") or "none",
            "output": call_record.response.strip(),
            "revision": is_revision,
            "iteration": state["iterations"],
        }
        updated = {
            **state,
            "worker_outputs": [*state["worker_outputs"], worker_output],
            "workers_executed": [*state["workers_executed"], worker_name],
            "revisions_requested": revisions_requested,
            "iterations": state["iterations"] + 1,
        }
        return append_step(
            updated,
            name=worker_name,
            step_type="worker_llm_call",
            prompt=prompt,
            output_data={"worker_output": worker_output},
            call_record=call_record,
            metadata={
                "worker_role": worker_name,
                "supervisor_iteration": state["iterations"],
                "revision": is_revision,
            },
        )

    def data_worker_node(state: SupervisorState) -> SupervisorState:
        return worker_node(state, "data_worker")

    def reasoning_worker_node(state: SupervisorState) -> SupervisorState:
        return worker_node(state, "reasoning_worker")

    def validation_worker_node(state: SupervisorState) -> SupervisorState:
        return worker_node(state, "validation_worker")

    def synthesis_worker_node(state: SupervisorState) -> SupervisorState:
        return worker_node(state, "synthesis_worker")

    def supervisor_finalize_node(state: SupervisorState) -> SupervisorState:
        stop_reason = state.get("stop_reason") or "supervisor_finalized"
        prompt = render_supervisor_workers_prompt(
            input_data,
            "supervisor_finalize",
            {**state, "stop_reason": stop_reason},
        )
        step_id = next_step_id(state)
        call_record = complete_llm_step(
            llm=context.llm,
            input_data=input_data,
            config=config,
            prompt=prompt,
            step_id=step_id,
        )
        updated = {
            **state,
            "final_output": call_record.response.strip(),
            "stop_reason": stop_reason,
        }
        return append_step(
            updated,
            name="supervisor_finalize",
            step_type="supervisor_finalize_llm_call",
            prompt=prompt,
            output_data={
                "final_output": call_record.response.strip(),
                "stop_reason": stop_reason,
                "warnings": list(state["warnings"]),
            },
            call_record=call_record,
            metadata={"supervisor_role": "finalizer"},
        )

    def route_after_decision(state: SupervisorState) -> str:
        action = state.get("next_action") or {}
        if action.get("action") == "finalize":
            return "supervisor_finalize"
        worker_name = action.get("worker_name")
        if worker_name in SUPERVISOR_WORKERS:
            return str(worker_name)
        state["warnings"].append("Invalid supervisor action. Finalizing with available outputs.")
        state["stop_reason"] = "invalid_supervisor_action"
        return "supervisor_finalize"

    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor_plan", supervisor_plan_node)
    graph.add_node("supervisor_decision", supervisor_decision_node)
    graph.add_node("data_worker", data_worker_node)
    graph.add_node("reasoning_worker", reasoning_worker_node)
    graph.add_node("validation_worker", validation_worker_node)
    graph.add_node("synthesis_worker", synthesis_worker_node)
    graph.add_node("supervisor_finalize", supervisor_finalize_node)
    graph.set_entry_point("supervisor_plan")
    graph.add_edge("supervisor_plan", "supervisor_decision")
    graph.add_conditional_edges(
        "supervisor_decision",
        route_after_decision,
        {
            "data_worker": "data_worker",
            "reasoning_worker": "reasoning_worker",
            "validation_worker": "validation_worker",
            "synthesis_worker": "synthesis_worker",
            "supervisor_finalize": "supervisor_finalize",
        },
    )
    for worker in SUPERVISOR_WORKERS:
        graph.add_edge(worker, "supervisor_decision")
    graph.add_edge("supervisor_finalize", END)

    initial_state: SupervisorState = {
        "plan": None,
        "worker_outputs": [],
        "iterations": 0,
        "max_supervisor_iterations": get_max_supervisor_iterations(config),
        "workers_executed": [],
        "revisions_requested": 0,
        "accepted_worker_outputs": [],
        "rejected_worker_outputs": [],
        "next_action": None,
        "stop_reason": None,
        "warnings": [],
        "steps": [],
        "llm_calls": [],
        "final_output": None,
    }
    state, resource_usage = invoke_with_resource_monitor(graph.compile(), initial_state)

    plan = state["plan"] or {}
    workers_to_run = list(plan.get("workers_to_run", []))
    workers_not_used = [worker for worker in SUPERVISOR_WORKERS if worker not in workers_to_run]
    final_answer = extract_final_answer(str(state["final_output"] or ""))
    structured_output = {
        "answer": final_answer,
        "mode": f"{config.model_provider}_supervisor_workers",
        "supervisor_plan": plan,
        "worker_outputs": state["worker_outputs"],
        "workers_executed": state["workers_executed"],
        "number_of_workers_executed": len(state["workers_executed"]),
        "workers_used": sorted(set(state["workers_executed"]), key=SUPERVISOR_WORKERS.index),
        "workers_not_used": workers_not_used,
        "supervisor_iterations": state["iterations"],
        "max_supervisor_iterations": state["max_supervisor_iterations"],
        "revisions_requested": state["revisions_requested"],
        "accepted_worker_outputs": state["accepted_worker_outputs"],
        "rejected_worker_outputs": state["rejected_worker_outputs"],
        "stop_reason": state["stop_reason"] or "supervisor_finalized",
        "warnings": state["warnings"],
        "document_ids": document_ids(input_data),
        "framework_execution": "langgraph_supervisor_workers_state_graph",
    }
    return LangGraphRunOutput(
        final_answer=final_answer,
        structured_output=structured_output,
        steps=state["steps"],
        llm_calls=state["llm_calls"],
        resource_usage=resource_usage,
    )
