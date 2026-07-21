"""LlamaIndex implementation for ARCH_08_DEBATE_JUDGE."""

import asyncio
from typing import Any

from llama_index.core.workflow import Context, Event, StartEvent, StopEvent, Workflow, step

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
from implementations.llamaindex.utils_llamaindex import (
    LlamaIndexRunContext,
    LlamaIndexRunOutput,
    build_function_agent,
    complete_agent_step,
    llamaindex_architecture_runner,
    run_async,
    run_with_resource_monitor,
)


FRAMEWORK_PRIMITIVE = "Workflow.send_event.collect_events.debate_round.judge"


class DebaterAEvent(Event):
    pass


class DebaterBEvent(Event):
    pass


class DebaterCEvent(Event):
    pass


class ProposalResultEvent(Event):
    component: str
    proposal: dict[str, Any]
    prompt: str
    started_at: Any
    finished_at: Any
    call: Any = None
    error: str | None = None


class DebateResultEvent(Event):
    proposals: dict[str, dict[str, Any]]
    proposal_results: list[ProposalResultEvent]
    debate_round: dict[str, Any]
    raw_output: str
    prompt: str
    started_at: Any
    finished_at: Any
    call: Any


class WorkflowPayload:
    def __init__(
        self,
        *,
        final_answer: str,
        structured_output: dict[str, Any],
        steps: list[AgentStep],
        llm_calls: list[LLMCallMetrics],
    ) -> None:
        self.final_answer = final_answer
        self.structured_output = structured_output
        self.steps = steps
        self.llm_calls = llm_calls


@llamaindex_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: LlamaIndexRunContext,
) -> LlamaIndexRunOutput:
    """Execute proposal events, collect them for debate, then invoke the judge."""

    agents = {
        component: build_function_agent(
            name=component,
            system_prompt=(
                f"You are the bounded {component} component of ARCH_08. "
                "Do not hand off, route, delegate, supervise, or run additional rounds."
            ),
            context=context,
            input_data=input_data,
            config=config,
        )
        for component in (*DEBATERS, DEBATE_ROUND, JUDGE)
    }

    class DebateJudgeWorkflow(Workflow):
        @step
        async def fan_out(
            self,
            ctx: Context,
            _ev: StartEvent,
        ) -> DebaterAEvent | DebaterBEvent | DebaterCEvent | None:
            ctx.send_event(DebaterAEvent())
            ctx.send_event(DebaterBEvent())
            ctx.send_event(DebaterCEvent())

        async def _run_debater(self, component: str) -> ProposalResultEvent:
            step_id = DEBATERS.index(component) + 1
            prompt = render_debate_judge_prompt(input_data, component)
            started_at = utc_now()
            call_record = None
            error = None
            try:
                call_record = await asyncio.to_thread(
                    complete_agent_step,
                    agent=agents[component],
                    prompt=prompt,
                    input_data=input_data,
                    config=config,
                    step_id=step_id,
                )
                response = call_record.response.strip()
            except Exception as exc:  # pragma: no cover - integration failure path
                error = f"{type(exc).__name__}: {exc}"
                response = ""
            return ProposalResultEvent(
                component=component,
                proposal=parse_debate_proposal(component, response, error=error).model_dump(),
                prompt=prompt,
                started_at=started_at,
                finished_at=utc_now(),
                call=call_record.metrics if call_record else None,
                error=error,
            )

        @step
        async def debater_a(self, _ev: DebaterAEvent) -> ProposalResultEvent:
            return await self._run_debater("debater_a")

        @step
        async def debater_b(self, _ev: DebaterBEvent) -> ProposalResultEvent:
            return await self._run_debater("debater_b")

        @step
        async def debater_c(self, _ev: DebaterCEvent) -> ProposalResultEvent:
            return await self._run_debater("debater_c")

        @step
        async def debate_round(
            self,
            ctx: Context,
            ev: ProposalResultEvent,
        ) -> DebateResultEvent | None:
            collected = ctx.collect_events(ev, [ProposalResultEvent] * len(DEBATERS))
            if collected is None:
                return None
            results_by_name = {result.component: result for result in collected}
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
            return DebateResultEvent(
                proposals=proposals,
                proposal_results=collected,
                debate_round=parse_debate_round(call_record.response).model_dump(),
                raw_output=call_record.response.strip(),
                prompt=prompt,
                started_at=started_at,
                finished_at=utc_now(),
                call=call_record.metrics,
            )

        @step
        async def judge(self, ev: DebateResultEvent) -> StopEvent:
            prompt = render_debate_judge_prompt(
                input_data,
                JUDGE,
                proposals=ev.proposals,
                debate=ev.debate_round,
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
            results_by_name = {result.component: result for result in ev.proposal_results}
            steps = []
            for step_id, debater in enumerate(DEBATERS, start=1):
                result = results_by_name[debater]
                steps.append(
                    make_debate_step(
                        step_id=step_id,
                        component=debater,
                        actor=f"llamaindex.workflow.{debater}",
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
                    actor="llamaindex.workflow.debate_round",
                    prompt=ev.prompt,
                    output={"debate_round": ev.debate_round},
                    llm_call_ids=[ev.call.call_id],
                    started_at=ev.started_at,
                    finished_at=ev.finished_at,
                    framework_primitive=FRAMEWORK_PRIMITIVE,
                    proposals=ev.proposals,
                )
            )
            steps.append(
                make_debate_step(
                    step_id=5,
                    component=JUDGE,
                    actor="llamaindex.workflow.judge",
                    prompt=prompt,
                    output={"judge_decision": decision},
                    llm_call_ids=[call_record.metrics.call_id],
                    started_at=started_at,
                    finished_at=finished_at,
                    framework_primitive=FRAMEWORK_PRIMITIVE,
                    proposals=ev.proposals,
                    debate=ev.debate_round,
                )
            )
            llm_calls = [
                results_by_name[debater].call
                for debater in DEBATERS
                if results_by_name[debater].call is not None
            ] + [ev.call, call_record.metrics]
            final_answer, structured_output = build_debate_structured_output(
                input_data=input_data,
                config=config,
                proposals=ev.proposals,
                debate_output=ev.raw_output,
                judge_output=call_record.response,
                steps=steps,
                llm_calls=llm_calls,
                framework_execution="llamaindex_native_event_workflow_debate_judge",
                framework_primitive=FRAMEWORK_PRIMITIVE,
            )
            return StopEvent(
                result=WorkflowPayload(
                    final_answer=final_answer,
                    structured_output=structured_output,
                    steps=steps,
                    llm_calls=llm_calls,
                )
            )

    workflow = DebateJudgeWorkflow(
        timeout=float(config.timeout_seconds),
        verbose=False,
        num_concurrent_runs=1,
    )

    def execute_workflow() -> LlamaIndexRunOutput:
        async def run_workflow():
            handler = workflow.run(common_input=input_data.model_dump(mode="json"))
            return await handler

        payload = run_async(run_workflow())
        if not isinstance(payload, WorkflowPayload):
            raise RuntimeError("LlamaIndex debate workflow did not return WorkflowPayload.")
        return LlamaIndexRunOutput(
            final_answer=payload.final_answer,
            structured_output=payload.structured_output,
            steps=payload.steps,
            llm_calls=payload.llm_calls,
        )

    return run_with_resource_monitor(execute_workflow)
