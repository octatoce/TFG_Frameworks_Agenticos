"""LlamaIndex implementation for ARCH_06_PARALLEL_FANOUT_FANIN."""

import asyncio
from typing import Any

from llama_index.core.workflow import Context, Event, StartEvent, StopEvent, Workflow, step

from benchmark_core.parallel_fanout_fanin import (
    AGGREGATOR,
    PARALLEL_BRANCHES,
    build_parallel_structured_output,
    make_parallel_step,
    parse_branch_analysis,
    render_parallel_fanout_fanin_prompt,
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


FRAMEWORK_PRIMITIVE = "Workflow.send_event.collect_events"


class FactualAnalysisEvent(Event):
    pass


class TechnicalReasoningEvent(Event):
    pass


class RiskConstraintsEvent(Event):
    pass


class AlternativeSolutionEvent(Event):
    pass


class BranchResultEvent(Event):
    component: str
    partial_output: dict[str, Any]
    prompt: str
    started_at: Any
    finished_at: Any
    call: Any = None
    error: str | None = None


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
    """Execute event fan-out and collect_events fan-in through LlamaIndex Workflow."""

    agents = {
        component: build_function_agent(
            name=component,
            system_prompt=(
                f"You are the isolated {component} of ARCH_06. "
                "Do not communicate with other branches, delegate, hand off, or iterate."
            ),
            context=context,
            input_data=input_data,
            config=config,
        )
        for component in (*PARALLEL_BRANCHES, AGGREGATOR)
    }

    class ParallelFanOutFanInWorkflow(Workflow):
        @step
        async def fan_out(
            self,
            ctx: Context,
            _ev: StartEvent,
        ) -> (
            FactualAnalysisEvent
            | TechnicalReasoningEvent
            | RiskConstraintsEvent
            | AlternativeSolutionEvent
            | None
        ):
            ctx.send_event(FactualAnalysisEvent())
            ctx.send_event(TechnicalReasoningEvent())
            ctx.send_event(RiskConstraintsEvent())
            ctx.send_event(AlternativeSolutionEvent())

        async def _run_branch(self, component: str) -> BranchResultEvent:
            step_id = PARALLEL_BRANCHES.index(component) + 1
            prompt = render_parallel_fanout_fanin_prompt(input_data, component)
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
            return BranchResultEvent(
                component=component,
                partial_output=parse_branch_analysis(component, response, error=error).model_dump(),
                prompt=prompt,
                started_at=started_at,
                finished_at=utc_now(),
                call=call_record.metrics if call_record else None,
                error=error,
            )

        @step
        async def factual_analysis_branch(
            self,
            ev: FactualAnalysisEvent,
        ) -> BranchResultEvent:
            return await self._run_branch("factual_analysis_branch")

        @step
        async def technical_reasoning_branch(
            self,
            ev: TechnicalReasoningEvent,
        ) -> BranchResultEvent:
            return await self._run_branch("technical_reasoning_branch")

        @step
        async def risk_constraints_branch(
            self,
            ev: RiskConstraintsEvent,
        ) -> BranchResultEvent:
            return await self._run_branch("risk_constraints_branch")

        @step
        async def alternative_solution_branch(
            self,
            ev: AlternativeSolutionEvent,
        ) -> BranchResultEvent:
            return await self._run_branch("alternative_solution_branch")

        @step
        async def aggregator(
            self,
            ctx: Context,
            ev: BranchResultEvent,
        ) -> StopEvent | None:
            collected = ctx.collect_events(ev, [BranchResultEvent] * len(PARALLEL_BRANCHES))
            if collected is None:
                return None
            results_by_name = {result.component: result for result in collected}
            partial_outputs = {
                branch: results_by_name[branch].partial_output for branch in PARALLEL_BRANCHES
            }
            prompt = render_parallel_fanout_fanin_prompt(
                input_data,
                AGGREGATOR,
                partial_outputs,
            )
            started_at = utc_now()
            call_record = await asyncio.to_thread(
                complete_agent_step,
                agent=agents[AGGREGATOR],
                prompt=prompt,
                input_data=input_data,
                config=config,
                step_id=5,
            )
            finished_at = utc_now()
            steps = []
            for step_id, branch in enumerate(PARALLEL_BRANCHES, start=1):
                result = results_by_name[branch]
                steps.append(
                    make_parallel_step(
                        step_id=step_id,
                        component=branch,
                        actor=f"llamaindex.workflow.{branch}",
                        prompt=result.prompt,
                        output={"partial_output": result.partial_output},
                        llm_call_id=result.call.call_id if result.call else None,
                        started_at=result.started_at,
                        finished_at=result.finished_at,
                        framework_primitive=FRAMEWORK_PRIMITIVE,
                        parallelism_used=True,
                        error=result.error,
                    )
                )
            steps.append(
                make_parallel_step(
                    step_id=5,
                    component=AGGREGATOR,
                    actor="llamaindex.workflow.aggregator",
                    prompt=prompt,
                    output={"aggregator_output": call_record.response.strip()},
                    llm_call_id=call_record.metrics.call_id,
                    started_at=started_at,
                    finished_at=finished_at,
                    framework_primitive=FRAMEWORK_PRIMITIVE,
                    parallelism_used=True,
                    partial_outputs=partial_outputs,
                )
            )
            llm_calls = [
                results_by_name[branch].call
                for branch in PARALLEL_BRANCHES
                if results_by_name[branch].call is not None
            ] + [call_record.metrics]
            final_answer, structured_output = build_parallel_structured_output(
                input_data=input_data,
                config=config,
                partial_outputs=partial_outputs,
                aggregator_output=call_record.response,
                steps=steps,
                llm_calls=llm_calls,
                framework_execution="llamaindex_native_event_workflow_fanout_fanin",
                framework_primitive=FRAMEWORK_PRIMITIVE,
                parallelism_used=True,
            )
            return StopEvent(
                result=WorkflowPayload(
                    final_answer=final_answer,
                    structured_output=structured_output,
                    steps=steps,
                    llm_calls=llm_calls,
                )
            )

    workflow = ParallelFanOutFanInWorkflow(
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
            raise RuntimeError("LlamaIndex fan-in workflow did not return WorkflowPayload.")
        return LlamaIndexRunOutput(
            final_answer=payload.final_answer,
            structured_output=payload.structured_output,
            steps=payload.steps,
            llm_calls=payload.llm_calls,
        )

    return run_with_resource_monitor(execute_workflow)
