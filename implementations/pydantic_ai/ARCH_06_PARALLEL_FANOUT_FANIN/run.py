"""Pydantic AI + pydantic-graph implementation for ARCH_06."""

import asyncio
from datetime import datetime
from typing import Any

from pydantic import BaseModel
from pydantic_graph import GraphBuilder, StepContext
from pydantic_graph.join import reduce_list_append

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
from implementations.pydantic_ai.utils_pydantic_ai import (
    PydanticAIRunContext,
    PydanticAIRunOutput,
    build_typed_agent,
    complete_agent_step,
    pydantic_ai_architecture_runner,
)


FRAMEWORK_PRIMITIVE = "GraphBuilder.broadcast_fork.Join"


class BranchExecution(BaseModel):
    component: str
    partial_output: dict[str, Any]
    prompt: str
    started_at: datetime
    finished_at: datetime
    call: LLMCallMetrics | None = None
    error: str | None = None


class ParallelGraphOutput(BaseModel):
    final_answer: str
    structured_output: dict[str, Any]
    steps: list[AgentStep]
    llm_calls: list[LLMCallMetrics]


@pydantic_ai_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: PydanticAIRunContext,
) -> PydanticAIRunOutput:
    """Execute a typed GraphBuilder broadcast fork, Join, and aggregator."""

    agents = {
        component: build_typed_agent(
            name=component,
            instructions=(
                f"You are the typed isolated {component} in ARCH_06. "
                "Do not access sibling results, delegate, supervise, hand off, or iterate."
            ),
            context=context,
            input_data=input_data,
            config=config,
        )
        for component in (*PARALLEL_BRANCHES, AGGREGATOR)
    }
    builder = GraphBuilder(
        name="ARCH_06_PARALLEL_FANOUT_FANIN",
        state_type=None,
        deps_type=None,
        input_type=ExperimentInput,
        output_type=ParallelGraphOutput,
        auto_instrument=False,
    )

    async def run_branch(component: str, common_input: ExperimentInput) -> BranchExecution:
        step_id = PARALLEL_BRANCHES.index(component) + 1
        prompt = render_parallel_fanout_fanin_prompt(common_input, component)
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
        return BranchExecution(
            component=component,
            partial_output=parse_branch_analysis(component, response, error=error).model_dump(),
            prompt=prompt,
            started_at=started_at,
            finished_at=utc_now(),
            call=call_record.metrics if call_record else None,
            error=error,
        )

    async def factual_analysis_branch(
        ctx: StepContext[None, None, ExperimentInput],
    ) -> BranchExecution:
        return await run_branch("factual_analysis_branch", ctx.inputs)

    async def technical_reasoning_branch(
        ctx: StepContext[None, None, ExperimentInput],
    ) -> BranchExecution:
        return await run_branch("technical_reasoning_branch", ctx.inputs)

    async def risk_constraints_branch(
        ctx: StepContext[None, None, ExperimentInput],
    ) -> BranchExecution:
        return await run_branch("risk_constraints_branch", ctx.inputs)

    async def alternative_solution_branch(
        ctx: StepContext[None, None, ExperimentInput],
    ) -> BranchExecution:
        return await run_branch("alternative_solution_branch", ctx.inputs)

    async def aggregator(
        ctx: StepContext[None, None, list[BranchExecution]],
    ) -> ParallelGraphOutput:
        results_by_name = {result.component: result for result in ctx.inputs}
        partial_outputs = {
            branch: results_by_name[branch].partial_output for branch in PARALLEL_BRANCHES
        }
        prompt = render_parallel_fanout_fanin_prompt(input_data, AGGREGATOR, partial_outputs)
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
                    actor=f"pydantic_ai.graph.{branch}",
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
                actor="pydantic_ai.graph.aggregator",
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
            framework_execution="pydantic_ai_graphbuilder_parallel_fork_join",
            framework_primitive=FRAMEWORK_PRIMITIVE,
            parallelism_used=True,
        )
        return ParallelGraphOutput(
            final_answer=final_answer,
            structured_output=structured_output,
            steps=steps,
            llm_calls=llm_calls,
        )

    factual_step = builder.step(factual_analysis_branch, node_id="factual_analysis_branch")
    technical_step = builder.step(technical_reasoning_branch, node_id="technical_reasoning_branch")
    risk_step = builder.step(risk_constraints_branch, node_id="risk_constraints_branch")
    alternative_step = builder.step(alternative_solution_branch, node_id="alternative_solution_branch")
    fan_in = builder.join(
        reduce_list_append,
        initial_factory=list,
        node_id="fan_in_join",
        parent_fork_id="perspective_fan_out",
    )
    aggregator_step = builder.step(aggregator, node_id=AGGREGATOR)
    branch_steps = [factual_step, technical_step, risk_step, alternative_step]

    builder.add(
        builder.edge_from(builder.start_node).to(
            *branch_steps,
            fork_id="perspective_fan_out",
        )
    )
    for branch_step in branch_steps:
        builder.add(builder.edge_from(branch_step).to(fan_in))
    builder.add(builder.edge_from(fan_in).to(aggregator_step))
    builder.add(builder.edge_from(aggregator_step).to(builder.end_node))
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
