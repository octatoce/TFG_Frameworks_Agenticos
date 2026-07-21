"""Pydantic AI + pydantic-graph implementation for ARCH_08_DEBATE_JUDGE."""

import asyncio
from datetime import datetime
from typing import Any

from pydantic import BaseModel
from pydantic_graph import GraphBuilder, StepContext
from pydantic_graph.join import reduce_list_append

from benchmark_core.debate_judge import (
    DEBATERS,
    DEBATE_ROUND,
    JUDGE,
    DebateProposal,
    DebateRoundOutput,
    JudgeDecision,
    build_debate_structured_output,
    make_debate_step,
    parse_debate_proposal,
    parse_debate_round,
    parse_judge_decision,
    render_debate_judge_prompt,
)
from benchmark_core.resource_monitor import ResourceMonitor
from benchmark_core.schemas import AgentStep, ExperimentConfig, ExperimentInput, LLMCallMetrics
from benchmark_core.tracing import utc_now
from implementations.pydantic_ai.utils_pydantic_ai import (
    PydanticAIRunContext,
    PydanticAIRunOutput,
    build_typed_agent,
    complete_agent_step,
    pydantic_ai_architecture_runner,
)


FRAMEWORK_PRIMITIVE = "GraphBuilder.broadcast_fork.Join.debate_round.judge"


class ProposalExecution(BaseModel):
    component: str
    proposal: DebateProposal
    prompt: str
    started_at: datetime
    finished_at: datetime
    call: LLMCallMetrics | None = None
    error: str | None = None


class DebateExecution(BaseModel):
    proposals: dict[str, dict[str, Any]]
    proposal_results: list[ProposalExecution]
    debate_round: DebateRoundOutput
    raw_output: str
    prompt: str
    started_at: datetime
    finished_at: datetime
    call: LLMCallMetrics


class DebateGraphOutput(BaseModel):
    final_answer: str
    structured_output: dict[str, Any]
    judge_decision: JudgeDecision
    steps: list[AgentStep]
    llm_calls: list[LLMCallMetrics]


@pydantic_ai_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: PydanticAIRunContext,
) -> PydanticAIRunOutput:
    """Execute a typed proposal fork, join, debate step, and judge step."""

    agents = {
        component: build_typed_agent(
            name=component,
            instructions=(
                f"You are the typed bounded {component} component in ARCH_08. "
                "Do not route, hand off, delegate, supervise, or add debate rounds."
            ),
            context=context,
            input_data=input_data,
            config=config,
        )
        for component in (*DEBATERS, DEBATE_ROUND, JUDGE)
    }
    builder = GraphBuilder(
        name="ARCH_08_DEBATE_JUDGE",
        state_type=None,
        deps_type=None,
        input_type=ExperimentInput,
        output_type=DebateGraphOutput,
        auto_instrument=False,
    )

    async def run_debater(component: str, common_input: ExperimentInput) -> ProposalExecution:
        step_id = DEBATERS.index(component) + 1
        prompt = render_debate_judge_prompt(common_input, component)
        started_at = utc_now()
        call_record = None
        error = None
        try:
            call_record = await asyncio.to_thread(
                complete_agent_step,
                agent=agents[component],
                prompt=prompt,
                input_data=common_input,
                config=config,
                step_id=step_id,
            )
            response = call_record.response.strip()
        except Exception as exc:  # pragma: no cover - integration failure path
            error = f"{type(exc).__name__}: {exc}"
            response = ""
        return ProposalExecution(
            component=component,
            proposal=parse_debate_proposal(component, response, error=error),
            prompt=prompt,
            started_at=started_at,
            finished_at=utc_now(),
            call=call_record.metrics if call_record else None,
            error=error,
        )

    async def debater_a(
        ctx: StepContext[None, None, ExperimentInput],
    ) -> ProposalExecution:
        return await run_debater("debater_a", ctx.inputs)

    async def debater_b(
        ctx: StepContext[None, None, ExperimentInput],
    ) -> ProposalExecution:
        return await run_debater("debater_b", ctx.inputs)

    async def debater_c(
        ctx: StepContext[None, None, ExperimentInput],
    ) -> ProposalExecution:
        return await run_debater("debater_c", ctx.inputs)

    async def debate_round(
        ctx: StepContext[None, None, list[ProposalExecution]],
    ) -> DebateExecution:
        results_by_name = {result.component: result for result in ctx.inputs}
        proposals = {
            debater: results_by_name[debater].proposal.model_dump() for debater in DEBATERS
        }
        prompt = render_debate_judge_prompt(
            input_data,
            DEBATE_ROUND,
            proposals=proposals,
        )
        started_at = utc_now()
        call_record = await asyncio.to_thread(
            complete_agent_step,
            agent=agents[DEBATE_ROUND],
            prompt=prompt,
            input_data=input_data,
            config=config,
            step_id=4,
        )
        return DebateExecution(
            proposals=proposals,
            proposal_results=ctx.inputs,
            debate_round=parse_debate_round(call_record.response),
            raw_output=call_record.response.strip(),
            prompt=prompt,
            started_at=started_at,
            finished_at=utc_now(),
            call=call_record.metrics,
        )

    async def judge(
        ctx: StepContext[None, None, DebateExecution],
    ) -> DebateGraphOutput:
        debate_result = ctx.inputs
        debate_dict = debate_result.debate_round.model_dump()
        prompt = render_debate_judge_prompt(
            input_data,
            JUDGE,
            proposals=debate_result.proposals,
            debate=debate_dict,
        )
        started_at = utc_now()
        call_record = await asyncio.to_thread(
            complete_agent_step,
            agent=agents[JUDGE],
            prompt=prompt,
            input_data=input_data,
            config=config,
            step_id=5,
        )
        finished_at = utc_now()
        decision = parse_judge_decision(call_record.response)
        results_by_name = {
            result.component: result for result in debate_result.proposal_results
        }
        steps = []
        for step_id, debater in enumerate(DEBATERS, start=1):
            result = results_by_name[debater]
            steps.append(
                make_debate_step(
                    step_id=step_id,
                    component=debater,
                    actor=f"pydantic_ai.graph.{debater}",
                    prompt=result.prompt,
                    output={"proposal": result.proposal.model_dump()},
                    llm_call_ids=[result.call.call_id] if result.call else [],
                    started_at=result.started_at,
                    finished_at=result.finished_at,
                    framework_primitive=FRAMEWORK_PRIMITIVE,
                    error=result.error,
                )
            )
        steps.append(
            make_debate_step(
                step_id=4,
                component=DEBATE_ROUND,
                actor="pydantic_ai.graph.debate_round",
                prompt=debate_result.prompt,
                output={"debate_round": debate_dict},
                llm_call_ids=[debate_result.call.call_id],
                started_at=debate_result.started_at,
                finished_at=debate_result.finished_at,
                framework_primitive=FRAMEWORK_PRIMITIVE,
                proposals=debate_result.proposals,
            )
        )
        steps.append(
            make_debate_step(
                step_id=5,
                component=JUDGE,
                actor="pydantic_ai.graph.judge",
                prompt=prompt,
                output={"judge_decision": decision.model_dump()},
                llm_call_ids=[call_record.metrics.call_id],
                started_at=started_at,
                finished_at=finished_at,
                framework_primitive=FRAMEWORK_PRIMITIVE,
                proposals=debate_result.proposals,
                debate=debate_dict,
            )
        )
        llm_calls = [
            results_by_name[debater].call
            for debater in DEBATERS
            if results_by_name[debater].call is not None
        ] + [debate_result.call, call_record.metrics]
        final_answer, structured_output = build_debate_structured_output(
            input_data=input_data,
            config=config,
            proposals=debate_result.proposals,
            debate_output=debate_result.raw_output,
            judge_output=call_record.response,
            steps=steps,
            llm_calls=llm_calls,
            framework_execution="pydantic_ai_typed_graph_debate_judge",
            framework_primitive=FRAMEWORK_PRIMITIVE,
        )
        return DebateGraphOutput(
            final_answer=final_answer,
            structured_output=structured_output,
            judge_decision=decision,
            steps=steps,
            llm_calls=llm_calls,
        )

    debater_steps = [
        builder.step(debater_a, node_id="debater_a"),
        builder.step(debater_b, node_id="debater_b"),
        builder.step(debater_c, node_id="debater_c"),
    ]
    proposal_join = builder.join(
        reduce_list_append,
        initial_factory=list,
        node_id="proposal_join",
        parent_fork_id="debater_fan_out",
    )
    debate_step = builder.step(debate_round, node_id=DEBATE_ROUND)
    judge_step = builder.step(judge, node_id=JUDGE)
    builder.add(
        builder.edge_from(builder.start_node).to(
            *debater_steps,
            fork_id="debater_fan_out",
        )
    )
    for debater_step in debater_steps:
        builder.add(builder.edge_from(debater_step).to(proposal_join))
    builder.add(builder.edge_from(proposal_join).to(debate_step))
    builder.add(builder.edge_from(debate_step).to(judge_step))
    builder.add(builder.edge_from(judge_step).to(builder.end_node))
    graph = builder.build()

    async def execute_graph():
        return await asyncio.wait_for(
            graph.run(inputs=input_data),
            timeout=float(config.timeout_seconds),
        )

    with ResourceMonitor() as monitor:
        output = asyncio.run(execute_graph())
        resource_usage = monitor.usage
    return PydanticAIRunOutput(
        final_answer=output.final_answer,
        structured_output=output.structured_output,
        steps=output.steps,
        llm_calls=output.llm_calls,
        resource_usage=resource_usage,
    )
