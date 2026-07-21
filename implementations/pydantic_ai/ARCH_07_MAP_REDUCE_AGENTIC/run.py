"""Pydantic AI + pydantic-graph implementation for ARCH_07."""

import asyncio
from typing import Any

from pydantic import BaseModel
from pydantic_graph import GraphBuilder, StepContext
from pydantic_graph.join import reduce_list_append

from benchmark_core.map_reduce_agentic import (
    MAPPER,
    REDUCER,
    DocumentBatch,
    build_map_reduce_structured_output,
    get_batch_size,
    make_mapper_step,
    make_partitioner_step,
    make_reducer_step,
    parse_mapper_analysis,
    partition_documents,
    render_map_reduce_prompt,
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


FRAMEWORK_PRIMITIVE = "GraphBuilder.map.Step.Join.reducer"


class MapperExecution(BaseModel):
    batch: DocumentBatch
    partial_output: dict[str, Any]
    prompt: str
    started_at: Any
    finished_at: Any
    call: LLMCallMetrics | None = None
    error: str | None = None


class MapReduceGraphOutput(BaseModel):
    final_answer: str
    structured_output: dict[str, Any]
    steps: list[AgentStep]
    llm_calls: list[LLMCallMetrics]


class MapReduceDeps:
    def __init__(self) -> None:
        self.partition_step: AgentStep | None = None


@pydantic_ai_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: PydanticAIRunContext,
) -> PydanticAIRunOutput:
    """Execute GraphBuilder map, typed mapper outputs, Join, and reducer."""

    batches = partition_documents(input_data, config)
    parallelism_used = len(batches) > 1
    mapper_agents = {
        batch.batch_id: build_typed_agent(
            name=f"document_batch_mapper_{batch.batch_index + 1}",
            instructions=(
                "Apply the same typed ARCH_07 mapper logic to only the assigned batch. "
                "Do not access sibling batches, route, hand off, or supervise."
            ),
            context=context,
            input_data=input_data,
            config=config,
        )
        for batch in batches
    }
    reducer_agent = build_typed_agent(
        name=REDUCER,
        instructions="Reduce typed mapper outputs once without rereading complete documents.",
        context=context,
        input_data=input_data,
        config=config,
    )
    deps = MapReduceDeps()
    builder = GraphBuilder(
        name="ARCH_07_MAP_REDUCE_AGENTIC",
        state_type=None,
        deps_type=MapReduceDeps,
        input_type=ExperimentInput,
        output_type=MapReduceGraphOutput,
        auto_instrument=False,
    )

    async def document_partitioner(
        ctx: StepContext[None, MapReduceDeps, ExperimentInput],
    ) -> list[DocumentBatch]:
        started_at = utc_now()
        partitioned = partition_documents(ctx.inputs, config)
        ctx.deps.partition_step = make_partitioner_step(
            input_data=ctx.inputs,
            batches=partitioned,
            batch_size=get_batch_size(config),
            started_at=started_at,
            finished_at=utc_now(),
            actor="pydantic_ai.graph.document_partitioner",
            framework_primitive=FRAMEWORK_PRIMITIVE,
        )
        return partitioned

    async def mapper(
        ctx: StepContext[None, MapReduceDeps, DocumentBatch],
    ) -> MapperExecution:
        batch = ctx.inputs
        prompt = render_map_reduce_prompt(input_data, MAPPER, batch=batch)
        started_at = utc_now()
        call_record = None
        error = None
        try:
            call_record = await asyncio.to_thread(
                complete_agent_step,
                agent=mapper_agents[batch.batch_id],
                prompt=prompt,
                input_data=input_data,
                config=config,
                step_id=batch.batch_index + 2,
            )
            response = call_record.response.strip()
        except Exception as exc:  # pragma: no cover - integration failure path
            error = f"{type(exc).__name__}: {exc}"
            response = ""
        return MapperExecution(
            batch=batch,
            partial_output=parse_mapper_analysis(batch, response, error=error).model_dump(),
            prompt=prompt,
            started_at=started_at,
            finished_at=utc_now(),
            call=call_record.metrics if call_record else None,
            error=error,
        )

    async def reducer(
        ctx: StepContext[None, MapReduceDeps, list[MapperExecution]],
    ) -> MapReduceGraphOutput:
        results = sorted(ctx.inputs, key=lambda item: item.batch.batch_index)
        partial_outputs = {
            result.batch.batch_id: result.partial_output for result in results
        }
        prompt = render_map_reduce_prompt(
            input_data,
            REDUCER,
            partial_outputs=partial_outputs,
        )
        started_at = utc_now()
        call_record = await asyncio.to_thread(
            complete_agent_step,
            agent=reducer_agent,
            prompt=prompt,
            input_data=input_data,
            config=config,
            step_id=len(batches) + 2,
        )
        finished_at = utc_now()
        mapper_steps = [
            make_mapper_step(
                batch=result.batch,
                prompt=result.prompt,
                partial_output=result.partial_output,
                llm_call_id=result.call.call_id if result.call else None,
                started_at=result.started_at,
                finished_at=result.finished_at,
                actor="pydantic_ai.graph.equivalent_mapper",
                framework_primitive=FRAMEWORK_PRIMITIVE,
                parallelism_used=parallelism_used,
                error=result.error,
            )
            for result in results
        ]
        reducer_step = make_reducer_step(
            batches=batches,
            prompt=prompt,
            reducer_output=call_record.response,
            partial_outputs=partial_outputs,
            llm_call_id=call_record.metrics.call_id,
            started_at=started_at,
            finished_at=finished_at,
            actor="pydantic_ai.graph.reducer",
            framework_primitive=FRAMEWORK_PRIMITIVE,
            parallelism_used=parallelism_used,
        )
        if ctx.deps.partition_step is None:
            raise RuntimeError("Pydantic graph partition step was not recorded.")
        steps = [ctx.deps.partition_step, *mapper_steps, reducer_step]
        llm_calls = [
            result.call for result in results if result.call is not None
        ] + [call_record.metrics]
        final_answer, structured_output = build_map_reduce_structured_output(
            input_data=input_data,
            config=config,
            batches=batches,
            partial_outputs=partial_outputs,
            reducer_output=call_record.response,
            steps=steps,
            llm_calls=llm_calls,
            framework_execution="pydantic_graphbuilder_dynamic_map_join_reduce",
            framework_primitive=FRAMEWORK_PRIMITIVE,
            parallelism_used=parallelism_used,
        )
        return MapReduceGraphOutput(
            final_answer=final_answer,
            structured_output=structured_output,
            steps=steps,
            llm_calls=llm_calls,
        )

    partition_step = builder.step(document_partitioner, node_id="document_partitioner")
    mapper_step = builder.step(mapper, node_id=MAPPER)
    mapper_join = builder.join(
        reduce_list_append,
        initial_factory=list,
        node_id="mapper_fan_in",
        parent_fork_id="document_batch_map",
    )
    reducer_step = builder.step(reducer, node_id=REDUCER)
    builder.add(builder.edge_from(builder.start_node).to(partition_step))
    builder.add(
        builder.edge_from(partition_step)
        .map(fork_id="document_batch_map", downstream_join_id=mapper_join.id)
        .to(mapper_step)
    )
    builder.add(builder.edge_from(mapper_step).to(mapper_join))
    builder.add(builder.edge_from(mapper_join).to(reducer_step))
    builder.add(builder.edge_from(reducer_step).to(builder.end_node))
    graph = builder.build()

    async def execute_graph():
        return await asyncio.wait_for(
            graph.run(inputs=input_data, deps=deps),
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
