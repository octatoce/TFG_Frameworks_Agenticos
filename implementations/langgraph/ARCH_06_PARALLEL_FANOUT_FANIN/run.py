"""LangGraph implementation for ARCH_06_PARALLEL_FANOUT_FANIN."""

from __future__ import annotations

import asyncio
from typing import TypedDict

from benchmark_core.parallel_fanout_fanin import (
    AGGREGATOR,
    PARALLEL_BRANCHES,
    build_parallel_structured_output,
    make_parallel_step,
    parse_branch_analysis,
    render_parallel_fanout_fanin_prompt,
)
from benchmark_core.resource_monitor import ResourceMonitor
from benchmark_core.schemas import AgentStep, ExperimentConfig, ExperimentInput, LLMCallMetrics
from benchmark_core.tracing import utc_now
from implementations.langgraph.utils_langgraph import (
    LangGraphRunContext,
    LangGraphRunOutput,
    complete_llm_step,
    langgraph_architecture_runner,
)


FRAMEWORK_PRIMITIVE = "StateGraph.START_fanout.list_edge_fanin"


class ParallelState(TypedDict, total=False):
    """State with isolated slots so concurrent branches never write the same key."""

    common_input: dict[str, object]
    factual_analysis_branch_output: dict[str, object]
    technical_reasoning_branch_output: dict[str, object]
    risk_constraints_branch_output: dict[str, object]
    alternative_solution_branch_output: dict[str, object]
    factual_analysis_branch_step: object
    technical_reasoning_branch_step: object
    risk_constraints_branch_step: object
    alternative_solution_branch_step: object
    factual_analysis_branch_call: object
    technical_reasoning_branch_call: object
    risk_constraints_branch_call: object
    alternative_solution_branch_call: object
    final_answer: str
    structured_output: dict[str, object]
    steps: list[object]
    llm_calls: list[object]


@langgraph_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: LangGraphRunContext,
) -> LangGraphRunOutput:
    """Execute four START fan-out branches and one list-edge fan-in aggregator."""

    from langgraph.graph import END, START, StateGraph

    def branch_node(component: str):
        def execute(_: ParallelState) -> ParallelState:
            started_at = utc_now()
            prompt = render_parallel_fanout_fanin_prompt(input_data, component)
            call_record = None
            error = None
            try:
                call_record = complete_llm_step(
                    llm=context.llm,
                    input_data=input_data,
                    config=config,
                    prompt=prompt,
                    step_id=PARALLEL_BRANCHES.index(component) + 1,
                )
                response = call_record.response.strip()
            except Exception as exc:  # pragma: no cover - integration failure path
                error = f"{type(exc).__name__}: {exc}"
                response = ""
            partial = parse_branch_analysis(component, response, error=error).model_dump()
            finished_at = utc_now()
            step = make_parallel_step(
                step_id=PARALLEL_BRANCHES.index(component) + 1,
                component=component,
                actor=f"langgraph.{component}_node",
                prompt=prompt,
                output={"partial_output": partial},
                llm_call_id=call_record.metrics.call_id if call_record else None,
                started_at=started_at,
                finished_at=finished_at,
                framework_primitive=FRAMEWORK_PRIMITIVE,
                parallelism_used=True,
                error=error,
            )
            return {
                f"{component}_output": partial,
                f"{component}_step": step,
                f"{component}_call": call_record.metrics if call_record else None,
            }

        return execute

    def aggregator_node(state: ParallelState) -> ParallelState:
        partial_outputs = {
            branch: state[f"{branch}_output"] for branch in PARALLEL_BRANCHES
        }
        prompt = render_parallel_fanout_fanin_prompt(input_data, AGGREGATOR, partial_outputs)
        started_at = utc_now()
        call_record = complete_llm_step(
            llm=context.llm,
            input_data=input_data,
            config=config,
            prompt=prompt,
            step_id=5,
        )
        finished_at = utc_now()
        aggregator_step = make_parallel_step(
            step_id=5,
            component=AGGREGATOR,
            actor="langgraph.aggregator_node",
            prompt=prompt,
            output={"aggregator_output": call_record.response.strip()},
            llm_call_id=call_record.metrics.call_id,
            started_at=started_at,
            finished_at=finished_at,
            framework_primitive=FRAMEWORK_PRIMITIVE,
            parallelism_used=True,
            partial_outputs=partial_outputs,
        )
        steps = [state[f"{branch}_step"] for branch in PARALLEL_BRANCHES] + [aggregator_step]
        llm_calls = [
            state[f"{branch}_call"]
            for branch in PARALLEL_BRANCHES
            if state.get(f"{branch}_call") is not None
        ] + [call_record.metrics]
        final_answer, structured_output = build_parallel_structured_output(
            input_data=input_data,
            config=config,
            partial_outputs=partial_outputs,
            aggregator_output=call_record.response,
            steps=steps,
            llm_calls=llm_calls,
            framework_execution="langgraph_native_parallel_state_graph",
            framework_primitive=FRAMEWORK_PRIMITIVE,
            parallelism_used=True,
        )
        return {
            "final_answer": final_answer,
            "structured_output": structured_output,
            "steps": steps,
            "llm_calls": llm_calls,
        }

    graph = StateGraph(ParallelState)
    for branch in PARALLEL_BRANCHES:
        graph.add_node(branch, branch_node(branch))
        graph.add_edge(START, branch)
    graph.add_node(AGGREGATOR, aggregator_node)
    graph.add_edge(list(PARALLEL_BRANCHES), AGGREGATOR)
    graph.add_edge(AGGREGATOR, END)

    initial_state: ParallelState = {
        "common_input": {
            "query": input_data.query,
            "documents": [document.model_dump() for document in input_data.documents],
            "metadata": input_data.metadata,
        }
    }

    async def invoke_graph():
        return await asyncio.wait_for(
            graph.compile().ainvoke(initial_state),
            timeout=float(config.timeout_seconds),
        )

    with ResourceMonitor() as monitor:
        state = asyncio.run(invoke_graph())
        resource_usage = monitor.usage

    return LangGraphRunOutput(
        final_answer=state["final_answer"],
        structured_output=state["structured_output"],
        steps=state["steps"],
        llm_calls=state["llm_calls"],
        resource_usage=resource_usage,
    )
