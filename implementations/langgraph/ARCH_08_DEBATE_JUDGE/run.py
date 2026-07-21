"""LangGraph implementation for ARCH_08_DEBATE_JUDGE."""

from __future__ import annotations

import asyncio
from typing import TypedDict

from benchmark_core.debate_judge import (
    DEBATERS,
    DEBATE_ROUND,
    JUDGE,
    build_debate_structured_output,
    make_debate_step,
    parse_debate_proposal,
    parse_debate_round,
    parse_judge_decision,
    render_debate_judge_prompt,
)
from benchmark_core.resource_monitor import ResourceMonitor
from benchmark_core.schemas import ExperimentConfig, ExperimentInput
from benchmark_core.tracing import utc_now
from implementations.langgraph.utils_langgraph import (
    LangGraphRunContext,
    LangGraphRunOutput,
    complete_llm_step,
    langgraph_architecture_runner,
)


FRAMEWORK_PRIMITIVE = "StateGraph.START_debater_fanout.debate_round.judge"


class DebateState(TypedDict, total=False):
    common_input: dict[str, object]
    debater_a_output: dict[str, object]
    debater_b_output: dict[str, object]
    debater_c_output: dict[str, object]
    debater_a_step: object
    debater_b_step: object
    debater_c_step: object
    debater_a_call: object
    debater_b_call: object
    debater_c_call: object
    debate_output: dict[str, object]
    debate_raw_output: str
    debate_step: object
    debate_call: object
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
    """Execute three proposal nodes, one explicit debate node, and one judge."""

    from langgraph.graph import END, START, StateGraph

    def debater_node(component: str):
        def execute(_: DebateState) -> DebateState:
            step_id = DEBATERS.index(component) + 1
            prompt = render_debate_judge_prompt(input_data, component)
            started_at = utc_now()
            call_record = None
            error = None
            try:
                call_record = complete_llm_step(
                    llm=context.llm,
                    input_data=input_data,
                    config=config,
                    prompt=prompt,
                    step_id=step_id,
                )
                response = call_record.response.strip()
            except Exception as exc:  # pragma: no cover - integration failure path
                error = f"{type(exc).__name__}: {exc}"
                response = ""
            proposal = parse_debate_proposal(component, response, error=error).model_dump()
            finished_at = utc_now()
            step = make_debate_step(
                step_id=step_id,
                component=component,
                actor=f"langgraph.{component}_node",
                prompt=prompt,
                output={"proposal": proposal},
                llm_call_ids=[call_record.metrics.call_id] if call_record else [],
                started_at=started_at,
                finished_at=finished_at,
                framework_primitive=FRAMEWORK_PRIMITIVE,
                error=error,
            )
            return {
                f"{component}_output": proposal,
                f"{component}_step": step,
                f"{component}_call": call_record.metrics if call_record else None,
            }

        return execute

    def debate_round_node(state: DebateState) -> DebateState:
        proposals = {debater: state[f"{debater}_output"] for debater in DEBATERS}
        prompt = render_debate_judge_prompt(
            input_data,
            DEBATE_ROUND,
            proposals=proposals,
        )
        started_at = utc_now()
        call_record = complete_llm_step(
            llm=context.llm,
            input_data=input_data,
            config=config,
            prompt=prompt,
            step_id=4,
        )
        finished_at = utc_now()
        debate_output = parse_debate_round(call_record.response).model_dump()
        return {
            "debate_output": debate_output,
            "debate_raw_output": call_record.response.strip(),
            "debate_step": make_debate_step(
                step_id=4,
                component=DEBATE_ROUND,
                actor="langgraph.debate_round_node",
                prompt=prompt,
                output={"debate_round": debate_output},
                llm_call_ids=[call_record.metrics.call_id],
                started_at=started_at,
                finished_at=finished_at,
                framework_primitive=FRAMEWORK_PRIMITIVE,
                proposals=proposals,
            ),
            "debate_call": call_record.metrics,
        }

    def judge_node(state: DebateState) -> DebateState:
        proposals = {debater: state[f"{debater}_output"] for debater in DEBATERS}
        debate_output = state["debate_output"]
        prompt = render_debate_judge_prompt(
            input_data,
            JUDGE,
            proposals=proposals,
            debate=debate_output,
        )
        started_at = utc_now()
        call_record = complete_llm_step(
            llm=context.llm,
            input_data=input_data,
            config=config,
            prompt=prompt,
            step_id=5,
        )
        finished_at = utc_now()
        decision = parse_judge_decision(call_record.response).model_dump()
        judge_step = make_debate_step(
            step_id=5,
            component=JUDGE,
            actor="langgraph.judge_node",
            prompt=prompt,
            output={"judge_decision": decision},
            llm_call_ids=[call_record.metrics.call_id],
            started_at=started_at,
            finished_at=finished_at,
            framework_primitive=FRAMEWORK_PRIMITIVE,
            proposals=proposals,
            debate=debate_output,
        )
        steps = [state[f"{debater}_step"] for debater in DEBATERS] + [
            state["debate_step"],
            judge_step,
        ]
        llm_calls = [
            state[f"{debater}_call"]
            for debater in DEBATERS
            if state.get(f"{debater}_call") is not None
        ] + [state["debate_call"], call_record.metrics]
        final_answer, structured_output = build_debate_structured_output(
            input_data=input_data,
            config=config,
            proposals=proposals,
            debate_output=state["debate_raw_output"],
            judge_output=call_record.response,
            steps=steps,
            llm_calls=llm_calls,
            framework_execution="langgraph_native_stategraph_debate_judge",
            framework_primitive=FRAMEWORK_PRIMITIVE,
        )
        return {
            "final_answer": final_answer,
            "structured_output": structured_output,
            "steps": steps,
            "llm_calls": llm_calls,
        }

    graph = StateGraph(DebateState)
    for debater in DEBATERS:
        graph.add_node(debater, debater_node(debater))
        graph.add_edge(START, debater)
    graph.add_node(DEBATE_ROUND, debate_round_node)
    graph.add_node(JUDGE, judge_node)
    graph.add_edge(list(DEBATERS), DEBATE_ROUND)
    graph.add_edge(DEBATE_ROUND, JUDGE)
    graph.add_edge(JUDGE, END)

    initial_state: DebateState = {
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
