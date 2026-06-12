"""LangGraph implementation for ARCH_02_SEQUENTIAL_PIPELINE."""

from __future__ import annotations

from typing import TypedDict

from benchmark_core.llm_wrapper import render_sequential_pipeline_prompt
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


class PipelineState(TypedDict, total=False):
    """State passed through the linear LangGraph pipeline."""

    query: str
    documents: list[object]
    plan: object
    evidence: object
    analysis: object
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
    """Execute a four-phase sequential pipeline through LangGraph."""

    from langgraph.graph import END, StateGraph

    def run_phase(state: PipelineState, phase: str, output_key: str) -> PipelineState:
        step_id = next_step_id(state)
        step_started_at = utc_now()
        prompt = render_sequential_pipeline_prompt(input_data, phase=phase, state=dict(state))
        call_record = complete_llm_step(
            llm=context.llm,
            input_data=input_data,
            config=config,
            prompt=prompt,
            step_id=step_id,
        )
        phase_output = call_record.response.strip()
        step = AgentStep(
            step_id=step_id,
            name=phase,
            step_type="pipeline_phase_llm_call",
            actor=f"langgraph.{phase}_node",
            input_data={
                "phase": phase,
                "prompt": prompt,
            },
            output_data={
                "phase_output": phase_output,
            },
            llm_call_ids=[call_record.metrics.call_id],
            started_at=step_started_at,
            finished_at=utc_now(),
            metadata={
                "pipeline_order": step_id,
                "graph_node": phase,
            },
        )

        return {
            output_key: phase_output,
            "steps": [*state.get("steps", []), step],
            "llm_calls": [*state.get("llm_calls", []), call_record.metrics],
        }

    graph = StateGraph(PipelineState)
    graph.add_node("planner", lambda state: run_phase(state, "planner", "plan"))
    graph.add_node("retriever", lambda state: run_phase(state, "retriever", "evidence"))
    graph.add_node("analyst", lambda state: run_phase(state, "analyst", "analysis"))
    graph.add_node("writer", lambda state: run_phase(state, "writer", "final_output"))
    graph.set_entry_point("planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "analyst")
    graph.add_edge("analyst", "writer")
    graph.add_edge("writer", END)

    initial_state: PipelineState = {
        "query": input_data.query,
        "documents": list(input_data.documents),
        "plan": None,
        "evidence": [],
        "analysis": None,
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
        "mode": f"{config.model_provider}_sequential_pipeline",
        "plan": state["plan"],
        "evidence": state["evidence"],
        "analysis": state["analysis"],
        "document_ids": document_ids(input_data),
        "framework_execution": "langgraph_linear_state_graph",
    }

    return LangGraphRunOutput(
        final_answer=final_answer,
        structured_output=structured_output,
        steps=state["steps"],
        llm_calls=state["llm_calls"],
        resource_usage=resource_usage,
    )
