"""Microsoft Agent Framework implementation for ARCH_06_PARALLEL_FANOUT_FANIN."""

import asyncio
from typing import Any, Never

from pydantic import BaseModel

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
from implementations.microsoft_agent_framework.utils_microsoft_agent_framework import (
    MicrosoftAgentFrameworkRunContext,
    MicrosoftAgentFrameworkRunOutput,
    build_agent,
    complete_agent_step,
    microsoft_agent_framework_architecture_runner,
    run_async,
    run_with_resource_monitor,
)


FRAMEWORK_PRIMITIVE = "WorkflowBuilder.add_fan_out_edges.add_fan_in_edges"


class BranchExecution(BaseModel):
    component: str
    partial_output: dict[str, Any]
    prompt: str
    started_at: Any
    finished_at: Any
    call: LLMCallMetrics | None
    error: str | None = None


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
    """Execute a native WorkflowBuilder fan-out/fan-in workflow."""

    def execute() -> MicrosoftAgentFrameworkRunOutput:
        from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler

        agents = {
            component: build_agent(
                name=component,
                instructions=(
                    f"You are the isolated {component} component of ARCH_06. "
                    "Use no memory, delegation, handoffs, or iterative supervision."
                ),
                context=context,
                input_data=input_data,
                config=config,
            )
            for component in (*PARALLEL_BRANCHES, AGGREGATOR)
        }

        class FanOutSource(Executor):
            @handler
            async def dispatch(
                self,
                message: ExperimentInput,
                ctx: WorkflowContext[ExperimentInput],
            ) -> None:
                await ctx.send_message(message)

        class ParallelBranchExecutor(Executor):
            def __init__(self, component: str) -> None:
                super().__init__(id=component)
                self.component = component

            @handler
            async def analyze(
                self,
                _message: ExperimentInput,
                ctx: WorkflowContext[BranchExecution],
            ) -> None:
                step_id = PARALLEL_BRANCHES.index(self.component) + 1
                prompt = render_parallel_fanout_fanin_prompt(input_data, self.component)
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
                    BranchExecution(
                        component=self.component,
                        partial_output=parse_branch_analysis(
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

        class AggregatorExecutor(Executor):
            @handler
            async def aggregate(
                self,
                branch_results: list[BranchExecution],
                ctx: WorkflowContext[Never, WorkflowPayload],
            ) -> None:
                results_by_name = {result.component: result for result in branch_results}
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
                            actor=f"microsoft_agent_framework.{branch}_executor",
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
                        actor="microsoft_agent_framework.aggregator_executor",
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
                    framework_execution="microsoft_native_workflow_fanout_fanin",
                    framework_primitive=FRAMEWORK_PRIMITIVE,
                    parallelism_used=True,
                )
                await ctx.yield_output(
                    WorkflowPayload(
                        final_answer=final_answer,
                        structured_output=structured_output,
                        steps=steps,
                        llm_calls=llm_calls,
                    )
                )

        source = FanOutSource(id="common_input_source")
        branches = [ParallelBranchExecutor(branch) for branch in PARALLEL_BRANCHES]
        aggregator = AggregatorExecutor(id=AGGREGATOR)
        workflow = (
            WorkflowBuilder(
                start_executor=source,
                output_from=[aggregator],
                max_iterations=max(config.max_agent_iterations, 10),
                name="ARCH_06_PARALLEL_FANOUT_FANIN",
            )
            .add_fan_out_edges(source, branches)
            .add_fan_in_edges(branches, aggregator)
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
            raise RuntimeError("Microsoft fan-in workflow did not produce one WorkflowPayload.")
        payload = outputs[0]
        return MicrosoftAgentFrameworkRunOutput(
            final_answer=payload.final_answer,
            structured_output=payload.structured_output,
            steps=payload.steps,
            llm_calls=payload.llm_calls,
        )

    return run_with_resource_monitor(execute)
