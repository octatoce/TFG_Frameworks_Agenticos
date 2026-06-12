"""LangGraph implementation for ARCH_03_SUPERVISOR_WORKERS."""

from __future__ import annotations

from typing import TypedDict

from benchmark_core.llm_wrapper import parse_worker_selection, render_supervisor_workers_prompt
from benchmark_core.schemas import AgentStep, ExperimentConfig, ExperimentInput
from benchmark_core.tracing import utc_now
from implementations.langgraph.utils_langgraph import (
    LangGraphRunContext,
    LangGraphRunOutput,
    complete_llm_step,
    document_ids,
    extract_final_answer,
    invoke_with_resource_monitor,
    langgraph_architecture_runner,
    next_step_id,
)


WORKERS = ["data_worker", "reasoning_worker", "validation_worker"]


class SupervisorState(TypedDict, total=False):
    """State passed through the LangGraph supervisor/workers graph."""

    query: str
    documents: list[object]
    selected_workers: list[str]
    skipped_workers: list[str]
    supervisor_plan: object
    evidence: object
    preliminary_decision: object
    validation_report: object
    final_output: object
    steps: list[object]
    llm_calls: list[object]
    errors: list[object]


@langgraph_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: LangGraphRunContext,
) -> LangGraphRunOutput:
    """Execute a supervisor-workers graph through LangGraph."""

    from langgraph.graph import END, StateGraph

    def run_phase(state: SupervisorState, phase: str, output_key: str | None) -> SupervisorState:
        step_id = next_step_id(state)
        step_started_at = utc_now()
        prompt = render_supervisor_workers_prompt(input_data, phase=phase, state=dict(state))
        call_record = complete_llm_step(
            llm=context.llm,
            input_data=input_data,
            config=config,
            prompt=prompt,
            step_id=step_id,
        )
        phase_output = call_record.response.strip()
        update: SupervisorState = {}

        if phase == "supervisor_planning":
            selected_workers, skipped_workers = parse_worker_selection(phase_output, WORKERS)
            update["selected_workers"] = selected_workers
            update["skipped_workers"] = skipped_workers
        if output_key is not None:
            update[output_key] = phase_output

        step = AgentStep(
            step_id=step_id,
            name=phase,
            step_type="supervisor_worker_llm_call",
            actor=f"langgraph.{phase}_node",
            input_data={
                "phase": phase,
                "prompt": prompt,
            },
            output_data={
                "phase_output": phase_output,
                "selected_workers": update.get("selected_workers", state.get("selected_workers", [])),
                "skipped_workers": update.get("skipped_workers", state.get("skipped_workers", [])),
            },
            llm_call_ids=[call_record.metrics.call_id],
            started_at=step_started_at,
            finished_at=utc_now(),
            metadata={
                "graph_node": phase,
                "worker_role": phase if phase in WORKERS else None,
                "pipeline_order": step_id,
            },
        )
        update["steps"] = [*state.get("steps", []), step]
        update["llm_calls"] = [*state.get("llm_calls", []), call_record.metrics]
        return update

    def next_after(state: SupervisorState, current_worker: str | None = None) -> str:
        selected_workers = state.get("selected_workers", [])
        start_index = 0
        if current_worker is not None:
            start_index = WORKERS.index(current_worker) + 1

        for worker in WORKERS[start_index:]:
            if worker in selected_workers:
                return worker
        return "supervisor_synthesis"

    graph = StateGraph(SupervisorState)
    graph.add_node(
        "supervisor_planning",
        lambda state: run_phase(state, "supervisor_planning", "supervisor_plan"),
    )
    graph.add_node("data_worker", lambda state: run_phase(state, "data_worker", "evidence"))
    graph.add_node(
        "reasoning_worker",
        lambda state: run_phase(state, "reasoning_worker", "preliminary_decision"),
    )
    graph.add_node(
        "validation_worker",
        lambda state: run_phase(state, "validation_worker", "validation_report"),
    )
    graph.add_node(
        "supervisor_synthesis",
        lambda state: run_phase(state, "supervisor_synthesis", "final_output"),
    )
    graph.set_entry_point("supervisor_planning")
    graph.add_conditional_edges(
        "supervisor_planning",
        lambda state: next_after(state),
        {
            "data_worker": "data_worker",
            "reasoning_worker": "reasoning_worker",
            "validation_worker": "validation_worker",
            "supervisor_synthesis": "supervisor_synthesis",
        },
    )
    graph.add_conditional_edges(
        "data_worker",
        lambda state: next_after(state, "data_worker"),
        {
            "reasoning_worker": "reasoning_worker",
            "validation_worker": "validation_worker",
            "supervisor_synthesis": "supervisor_synthesis",
        },
    )
    graph.add_conditional_edges(
        "reasoning_worker",
        lambda state: next_after(state, "reasoning_worker"),
        {
            "validation_worker": "validation_worker",
            "supervisor_synthesis": "supervisor_synthesis",
        },
    )
    graph.add_edge("validation_worker", "supervisor_synthesis")
    graph.add_edge("supervisor_synthesis", END)

    initial_state: SupervisorState = {
        "query": input_data.query,
        "documents": list(input_data.documents),
        "selected_workers": [],
        "skipped_workers": [],
        "supervisor_plan": None,
        "evidence": [],
        "preliminary_decision": None,
        "validation_report": None,
        "final_output": None,
        "steps": [],
        "llm_calls": [],
        "errors": [],
    }
    state, resource_usage = invoke_with_resource_monitor(graph.compile(), initial_state)

    final_output = str(state["final_output"])
    final_answer = extract_final_answer(final_output)
    structured_output = {
        "answer": final_answer,
        "mode": f"{config.model_provider}_supervisor_workers",
        "selected_workers": state["selected_workers"],
        "skipped_workers": state["skipped_workers"],
        "evidence": state["evidence"],
        "preliminary_decision": state["preliminary_decision"],
        "validation_report": state["validation_report"],
        "document_ids": document_ids(input_data),
        "framework_execution": "langgraph_supervisor_workers_graph",
    }

    return LangGraphRunOutput(
        final_answer=final_answer,
        structured_output=structured_output,
        steps=state["steps"],
        llm_calls=state["llm_calls"],
        resource_usage=resource_usage,
    )
