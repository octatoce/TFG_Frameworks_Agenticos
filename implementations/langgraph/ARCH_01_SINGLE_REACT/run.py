"""LangGraph implementation for ARCH_01_SINGLE_REACT."""

from __future__ import annotations

from typing import TypedDict

from benchmark_core.llm_wrapper import render_single_react_prompt
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
)


class SingleReActState(TypedDict, total=False):
    """State passed through the single-node LangGraph graph."""

    final_answer: str
    structured_output: dict[str, object]
    steps: list[object]
    llm_calls: list[object]
    errors: list[object]


@langgraph_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: LangGraphRunContext,
) -> LangGraphRunOutput:
    """Execute a single ReAct-style node through LangGraph."""

    from langgraph.graph import END, StateGraph

    def react_agent(_: SingleReActState) -> SingleReActState:
        step_started_at = utc_now()
        prompt = render_single_react_prompt(input_data)
        call_record = complete_llm_step(
            llm=context.llm,
            input_data=input_data,
            config=config,
            prompt=prompt,
            step_id=1,
        )
        final_answer = extract_final_answer(call_record.response)
        structured_output = {
            "answer": final_answer,
            "mode": f"{config.model_provider}_react",
            "document_ids": document_ids(input_data),
            "framework_execution": "langgraph_state_graph",
        }
        step = AgentStep(
            step_id=1,
            name="single_react_agent",
            step_type="agent_llm_call",
            actor="langgraph.react_agent_node",
            input_data={"prompt": prompt},
            output_data=structured_output,
            llm_call_ids=[call_record.metrics.call_id],
            started_at=step_started_at,
            finished_at=utc_now(),
            metadata={"graph_nodes": ["react_agent"]},
        )
        return {
            "final_answer": final_answer,
            "structured_output": structured_output,
            "steps": [step],
            "llm_calls": [call_record.metrics],
            "errors": [],
        }

    graph = StateGraph(SingleReActState)
    graph.add_node("react_agent", react_agent)
    graph.set_entry_point("react_agent")
    graph.add_edge("react_agent", END)

    state, resource_usage = invoke_with_resource_monitor(graph.compile(), {})

    return LangGraphRunOutput(
        final_answer=state["final_answer"],
        structured_output=state["structured_output"],
        steps=state["steps"],
        llm_calls=state["llm_calls"],
        resource_usage=resource_usage,
    )
