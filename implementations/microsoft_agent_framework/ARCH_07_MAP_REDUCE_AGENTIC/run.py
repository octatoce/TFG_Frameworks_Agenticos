"""Microsoft Agent Framework implementation for ARCH_07_MAP_REDUCE_AGENTIC."""

import asyncio
from typing import Any, Never

from pydantic import BaseModel

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


FRAMEWORK_PRIMITIVE = "WorkflowBuilder.partitioner.fan_out.fan_in.reducer"


class MapperExecution(BaseModel):
    batch: DocumentBatch
    partial_output: dict[str, Any]
    prompt: str
    started_at: Any
    finished_at: Any
    call: LLMCallMetrics | None = None
    error: str | None = None


class PartitionManifest(BaseModel):
    batches: list[DocumentBatch]


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
    """Execute one workflow partitioner, equivalent mapper executors, and reducer."""

    def execute() -> MicrosoftAgentFrameworkRunOutput:
        from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler

        batches = partition_documents(input_data, config)
        parallelism_used = len(batches) > 1
        partition_step_holder: list[AgentStep] = []
        mapper_agents = {
            batch.batch_id: build_agent(
                name=f"document_batch_mapper_{batch.batch_index + 1}",
                instructions=(
                    "You are an equivalent ARCH_07 document-batch mapper. Process only the assigned "
                    "batch with the canonical mapper prompt. Do not route, delegate, hand off, or supervise."
                ),
                context=context,
                input_data=input_data,
                config=config,
            )
            for batch in batches
        }
        reducer_agent = build_agent(
            name=REDUCER,
            instructions="Reduce mapper outputs once without rereading complete original documents.",
            context=context,
            input_data=input_data,
            config=config,
        )

        class DocumentPartitionerExecutor(Executor):
            @handler
            async def dispatch(
                self,
                message: ExperimentInput,
                ctx: WorkflowContext[PartitionManifest],
            ) -> None:
                started_at = utc_now()
                partitioned = partition_documents(message, config)
                if [batch.document_ids for batch in partitioned] != [
                    batch.document_ids for batch in batches
                ]:
                    raise RuntimeError("Microsoft workflow partition changed after graph construction.")
                partition_step_holder.append(
                    make_partitioner_step(
                        input_data=message,
                        batches=partitioned,
                        batch_size=get_batch_size(config),
                        started_at=started_at,
                        finished_at=utc_now(),
                        actor="microsoft_agent_framework.document_partitioner_executor",
                        framework_primitive=FRAMEWORK_PRIMITIVE,
                    )
                )
                await ctx.send_message(PartitionManifest(batches=partitioned))

        class MapperExecutor(Executor):
            def __init__(self, batch: DocumentBatch) -> None:
                super().__init__(id=f"mapper_{batch.batch_index + 1:03d}")
                self.batch = batch

            @handler
            async def map_batch(
                self,
                manifest: PartitionManifest,
                ctx: WorkflowContext[MapperExecution],
            ) -> None:
                batch = manifest.batches[self.batch.batch_index]
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
                await ctx.send_message(
                    MapperExecution(
                        batch=batch,
                        partial_output=parse_mapper_analysis(
                            batch,
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

        class ReducerExecutor(Executor):
            @handler
            async def reduce_batches(
                self,
                mapper_results: list[MapperExecution],
                ctx: WorkflowContext[Never, WorkflowPayload],
            ) -> None:
                results = sorted(mapper_results, key=lambda item: item.batch.batch_index)
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
                        actor="microsoft_agent_framework.mapper_executor",
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
                    actor="microsoft_agent_framework.reducer_executor",
                    framework_primitive=FRAMEWORK_PRIMITIVE,
                    parallelism_used=parallelism_used,
                )
                if not partition_step_holder:
                    raise RuntimeError("Microsoft partitioner step was not recorded.")
                steps = [partition_step_holder[0], *mapper_steps, reducer_step]
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
                    framework_execution="microsoft_workflow_dynamic_mapper_fanout_fanin",
                    framework_primitive=FRAMEWORK_PRIMITIVE,
                    parallelism_used=parallelism_used,
                )
                await ctx.yield_output(
                    WorkflowPayload(
                        final_answer=final_answer,
                        structured_output=structured_output,
                        steps=steps,
                        llm_calls=llm_calls,
                    )
                )

        partitioner = DocumentPartitionerExecutor(id="document_partitioner")
        mappers = [MapperExecutor(batch) for batch in batches]
        reducer = ReducerExecutor(id=REDUCER)
        workflow = (
            WorkflowBuilder(
                start_executor=partitioner,
                output_from=[reducer],
                max_iterations=max(config.max_agent_iterations, len(batches) + 3),
                name="ARCH_07_MAP_REDUCE_AGENTIC",
            )
            .add_fan_out_edges(partitioner, mappers)
            .add_fan_in_edges(mappers, reducer)
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
            raise RuntimeError("Microsoft Map-Reduce workflow did not produce one payload.")
        payload = outputs[0]
        return MicrosoftAgentFrameworkRunOutput(
            final_answer=payload.final_answer,
            structured_output=payload.structured_output,
            steps=payload.steps,
            llm_calls=payload.llm_calls,
        )

    return run_with_resource_monitor(execute)
