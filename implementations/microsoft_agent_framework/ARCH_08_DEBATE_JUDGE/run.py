"""Microsoft Agent Framework implementation for ARCH_08_DEBATE_JUDGE."""

import asyncio
from typing import Any, Never

from pydantic import BaseModel

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
from benchmark_core.schemas import AgentStep, ExperimentConfig, ExperimentInput, LLMCallMetrics
from benchmark_core.tracing import utc_now
from implementations.microsoft_agent_framework.utils_microsoft_agent_framework import (
    MicrosoftAgentFrameworkRunContext,
    MicrosoftAgentFrameworkRunOutput,
    build_agent,
    complete_agent_step,
    microsoft_agent_framework_architecture_runner,
    run_async,
    run_with_resource_monitor,
)


FRAMEWORK_PRIMITIVE = "WorkflowBuilder.fan_out.fan_in.debate_round.judge"


class ProposalExecution(BaseModel):
    component: str
    proposal: dict[str, Any]
    prompt: str
    started_at: Any
    finished_at: Any
    call: LLMCallMetrics | None = None
    error: str | None = None


class DebateExecution(BaseModel):
    proposals: dict[str, dict[str, Any]]
    proposal_results: list[ProposalExecution]
    debate_round: dict[str, Any]
    raw_output: str
    prompt: str
    started_at: Any
    finished_at: Any
    call: LLMCallMetrics


class WorkflowPayload(BaseModel):
    final_answer: str
    structured_output: dict[str, Any]
    steps: list[AgentStep]
    llm_calls: list[LLMCallMetrics]


@microsoft_agent_framework_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: MicrosoftAgentFrameworkRunContext,
) -> MicrosoftAgentFrameworkRunOutput:
    """Execute a native fan-out followed by explicit debate and judge executors."""

    def execute() -> MicrosoftAgentFrameworkRunOutput:
        from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler

        agents = {
            component: build_agent(
                name=component,
                instructions=(
                    f"You are the bounded {component} component of ARCH_08. "
                    "Use no memory, delegation, handoffs, routing, or iterative supervision."
                ),
                context=context,
                input_data=input_data,
                config=config,
            )
            for component in (*DEBATERS, DEBATE_ROUND, JUDGE)
        }

        class DebateSource(Executor):
            @handler
            async def dispatch(
                self,
                message: ExperimentInput,
                ctx: WorkflowContext[ExperimentInput],
            ) -> None:
                await ctx.send_message(message)

        class DebaterExecutor(Executor):
            def __init__(self, component: str) -> None:
                super().__init__(id=component)
                self.component = component

            @handler
            async def propose(
                self,
                _message: ExperimentInput,
                ctx: WorkflowContext[ProposalExecution],
            ) -> None:
                step_id = DEBATERS.index(self.component) + 1
                prompt = render_debate_judge_prompt(input_data, self.component)
                started_at = utc_now()
                call_record = None
                error = None
                try:
                    call_record = await asyncio.to_thread(
                        complete_agent_step,
                        agent=agents[self.component],
                        prompt=prompt,
                        input_data=input_data,
                        config=config,
                        step_id=step_id,
                    )
                    response = call_record.response.strip()
                except Exception as exc:  # pragma: no cover - integration failure path
                    error = f"{type(exc).__name__}: {exc}"
                    response = ""
                await ctx.send_message(
                    ProposalExecution(
                        component=self.component,
                        proposal=parse_debate_proposal(
                            self.component,
                            response,
                            error=error,
                        ).model_dump(),
                        prompt=prompt,
                        started_at=started_at,
                        finished_at=utc_now(),
                        call=call_record.metrics if call_record else None,
                        error=error,
                    )
                )

        class DebateRoundExecutor(Executor):
            @handler
            async def debate(
                self,
                proposal_results: list[ProposalExecution],
                ctx: WorkflowContext[DebateExecution],
            ) -> None:
                results_by_name = {result.component: result for result in proposal_results}
                proposals = {debater: results_by_name[debater].proposal for debater in DEBATERS}
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
                await ctx.send_message(
                    DebateExecution(
                        proposals=proposals,
                        proposal_results=proposal_results,
                        debate_round=parse_debate_round(call_record.response).model_dump(),
                        raw_output=call_record.response.strip(),
                        prompt=prompt,
                        started_at=started_at,
                        finished_at=utc_now(),
                        call=call_record.metrics,
                    )
                )

        class JudgeExecutor(Executor):
            @handler
            async def judge(
                self,
                debate_result: DebateExecution,
                ctx: WorkflowContext[Never, WorkflowPayload],
            ) -> None:
                prompt = render_debate_judge_prompt(
                    input_data,
                    JUDGE,
                    proposals=debate_result.proposals,
                    debate=debate_result.debate_round,
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
                decision = parse_judge_decision(call_record.response).model_dump()
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
                            actor=f"microsoft_agent_framework.{debater}_executor",
                            prompt=result.prompt,
                            output={"proposal": result.proposal},
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
                        actor="microsoft_agent_framework.debate_round_executor",
                        prompt=debate_result.prompt,
                        output={"debate_round": debate_result.debate_round},
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
                        actor="microsoft_agent_framework.judge_executor",
                        prompt=prompt,
                        output={"judge_decision": decision},
                        llm_call_ids=[call_record.metrics.call_id],
                        started_at=started_at,
                        finished_at=finished_at,
                        framework_primitive=FRAMEWORK_PRIMITIVE,
                        proposals=debate_result.proposals,
                        debate=debate_result.debate_round,
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
                    framework_execution="microsoft_native_workflow_debate_judge",
                    framework_primitive=FRAMEWORK_PRIMITIVE,
                )
                await ctx.yield_output(
                    WorkflowPayload(
                        final_answer=final_answer,
                        structured_output=structured_output,
                        steps=steps,
                        llm_calls=llm_calls,
                    )
                )

        source = DebateSource(id="common_input_source")
        debaters = [DebaterExecutor(debater) for debater in DEBATERS]
        debate_round = DebateRoundExecutor(id=DEBATE_ROUND)
        judge = JudgeExecutor(id=JUDGE)
        workflow = (
            WorkflowBuilder(
                start_executor=source,
                output_from=[judge],
                max_iterations=max(config.max_agent_iterations, 10),
                name="ARCH_08_DEBATE_JUDGE",
            )
            .add_fan_out_edges(source, debaters)
            .add_fan_in_edges(debaters, debate_round)
            .add_edge(debate_round, judge)
            .build()
        )

        async def run_workflow():
            return await asyncio.wait_for(
                workflow.run(input_data),
                timeout=float(config.timeout_seconds),
            )

        result = run_async(run_workflow())
        outputs = result.get_outputs()
        if len(outputs) != 1 or not isinstance(outputs[0], WorkflowPayload):
            raise RuntimeError("Microsoft debate workflow did not produce one WorkflowPayload.")
        payload = outputs[0]
        return MicrosoftAgentFrameworkRunOutput(
            final_answer=payload.final_answer,
            structured_output=payload.structured_output,
            steps=payload.steps,
            llm_calls=payload.llm_calls,
        )

    return run_with_resource_monitor(execute)
