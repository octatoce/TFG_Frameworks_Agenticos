"""LangGraph implementation for ARCH_03_ROUTER_SPECIALISTS."""

from __future__ import annotations

from typing import TypedDict

from benchmark_core.llm_wrapper import parse_specialist_selection, render_router_specialists_prompt
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


SPECIALISTS = ["data_specialist", "reasoning_specialist", "validation_specialist"]


class RouterState(TypedDict, total=False):
    """State passed through the LangGraph router/specialists graph."""

    query: str
    documents: list[object]
    selected_specialists: list[str]
    skipped_specialists: list[str]
    router_plan: object
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
    """Execute a router-specialists graph through LangGraph."""

    from langgraph.graph import END, StateGraph

    def run_phase(state: RouterState, phase: str, output_key: str | None) -> RouterState:
        step_id = next_step_id(state)
        step_started_at = utc_now()
        prompt = render_router_specialists_prompt(input_data, phase=phase, state=dict(state))
        call_record = complete_llm_step(
            llm=context.llm,
            input_data=input_data,
            config=config,
            prompt=prompt,
            step_id=step_id,
        )
        phase_output = call_record.response.strip()
        update: RouterState = {}

        if phase == "router_routing":
            selected_specialists, skipped_specialists = parse_specialist_selection(phase_output, SPECIALISTS)
            update["selected_specialists"] = selected_specialists
            update["skipped_specialists"] = skipped_specialists
        if output_key is not None:
            update[output_key] = phase_output

        step = AgentStep(
            step_id=step_id,
            name=phase,
            step_type="router_specialist_llm_call",
            actor=f"langgraph.{phase}_node",
            input_data={
                "phase": phase,
                "prompt": prompt,
            },
            output_data={
                "phase_output": phase_output,
                "selected_specialists": update.get("selected_specialists", state.get("selected_specialists", [])),
                "skipped_specialists": update.get("skipped_specialists", state.get("skipped_specialists", [])),
            },
            llm_call_ids=[call_record.metrics.call_id],
            started_at=step_started_at,
            finished_at=utc_now(),
            metadata={
                "graph_node": phase,
                "specialist_role": phase if phase in SPECIALISTS else None,
                "pipeline_order": step_id,
            },
        )
        update["steps"] = [*state.get("steps", []), step]
        update["llm_calls"] = [*state.get("llm_calls", []), call_record.metrics]
        return update

    def next_after(state: RouterState, current_specialist: str | None = None) -> str:
        selected_specialists = state.get("selected_specialists", [])
        start_index = 0
        if current_specialist is not None:
            start_index = SPECIALISTS.index(current_specialist) + 1

        for specialist in SPECIALISTS[start_index:]:
            if specialist in selected_specialists:
                return specialist
        return "router_synthesis"

    graph = StateGraph(RouterState)
    graph.add_node(
        "router_routing",
        lambda state: run_phase(state, "router_routing", "router_plan"),
    )
    graph.add_node("data_specialist", lambda state: run_phase(state, "data_specialist", "evidence"))
    graph.add_node(
        "reasoning_specialist",
        lambda state: run_phase(state, "reasoning_specialist", "preliminary_decision"),
    )
    graph.add_node(
        "validation_specialist",
        lambda state: run_phase(state, "validation_specialist", "validation_report"),
    )
    graph.add_node(
        "router_synthesis",
        lambda state: run_phase(state, "router_synthesis", "final_output"),
    )
    graph.set_entry_point("router_routing")
    graph.add_conditional_edges(
        "router_routing",
        lambda state: next_after(state),
        {
            "data_specialist": "data_specialist",
            "reasoning_specialist": "reasoning_specialist",
            "validation_specialist": "validation_specialist",
            "router_synthesis": "router_synthesis",
        },
    )
    graph.add_conditional_edges(
        "data_specialist",
        lambda state: next_after(state, "data_specialist"),
        {
            "reasoning_specialist": "reasoning_specialist",
            "validation_specialist": "validation_specialist",
            "router_synthesis": "router_synthesis",
        },
    )
    graph.add_conditional_edges(
        "reasoning_specialist",
        lambda state: next_after(state, "reasoning_specialist"),
        {
            "validation_specialist": "validation_specialist",
            "router_synthesis": "router_synthesis",
        },
    )
    graph.add_edge("validation_specialist", "router_synthesis")
    graph.add_edge("router_synthesis", END)

    initial_state: RouterState = {
        "query": input_data.query,
        "documents": list(input_data.documents),
        "selected_specialists": [],
        "skipped_specialists": [],
        "router_plan": None,
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
        "mode": f"{config.model_provider}_router_specialists",
        "selected_specialists": state["selected_specialists"],
        "skipped_specialists": state["skipped_specialists"],
        "evidence": state["evidence"],
        "preliminary_decision": state["preliminary_decision"],
        "validation_report": state["validation_report"],
        "document_ids": document_ids(input_data),
        "framework_execution": "langgraph_router_specialists_graph",
    }

    return LangGraphRunOutput(
        final_answer=final_answer,
        structured_output=structured_output,
        steps=state["steps"],
        llm_calls=state["llm_calls"],
        resource_usage=resource_usage,
    )

