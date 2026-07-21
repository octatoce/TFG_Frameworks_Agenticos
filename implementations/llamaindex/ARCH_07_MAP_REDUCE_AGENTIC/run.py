"""LlamaIndex implementation for ARCH_07_MAP_REDUCE_AGENTIC."""

import asyncio
from typing import Any

from llama_index.core.workflow import Context, Event, StartEvent, StopEvent, Workflow, step

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
from implementations.llamaindex.utils_llamaindex import (
    LlamaIndexRunContext,
    LlamaIndexRunOutput,
    build_function_agent,
    complete_agent_step,
    llamaindex_architecture_runner,
    run_async,
    run_with_resource_monitor,
)


FRAMEWORK_PRIMITIVE = "Workflow.dynamic_events.num_workers.collect_events"


class MapperEvent(Event):
    batch: dict[str, Any]


class MapperResultEvent(Event):
    batch: dict[str, Any]
    partial_output: dict[str, Any]
    prompt: str
    started_at: Any
    finished_at: Any
    call: Any = None
    error: str | None = None


class WorkflowPayload:
    def __init__(self, final_answer, structured_output, steps, llm_calls) -> None:
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
    """Execute dynamic mapper events and collect them at one reducer step."""

    batches = partition_documents(input_data, config)
    parallelism_used = len(batches) > 1
    mapper_agents = {
        batch.batch_id: build_function_agent(
            name=f"document_batch_mapper_{batch.batch_index + 1}",
            system_prompt=(
                "Apply the canonical ARCH_07 mapper logic only to the assigned batch. "
                "Do not access sibling batches, route, hand off, or supervise."
            ),
            context=context,
            input_data=input_data,
            config=config,
        )
        for batch in batches
    }
    reducer_agent = build_function_agent(
        name=REDUCER,
        system_prompt="Reduce all mapper outputs once without rereading the original documents.",
        context=context,
        input_data=input_data,
        config=config,
    )
    partition_step_holder: list[AgentStep] = []

    class DocumentMapReduceWorkflow(Workflow):
        @step
        async def document_partitioner(
            self,
            ctx: Context,
            _ev: StartEvent,
        ) -> MapperEvent | None:
            started_at = utc_now()
            partitioned = partition_documents(input_data, config)
            partition_step_holder.append(
                make_partitioner_step(
                    input_data=input_data,
                    batches=partitioned,
                    batch_size=get_batch_size(config),
                    started_at=started_at,
                    finished_at=utc_now(),
                    actor="llamaindex.workflow.document_partitioner",
                    framework_primitive=FRAMEWORK_PRIMITIVE,
                )
            )
            for batch in partitioned:
                ctx.send_event(MapperEvent(batch=batch.model_dump(mode="json")))
            return None

        @step(num_workers=max(len(batches), 1))
        async def mapper(self, ev: MapperEvent) -> MapperResultEvent:
            batch = DocumentBatch.model_validate(ev.batch)
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
            return MapperResultEvent(
                batch=batch.model_dump(mode="json"),
                partial_output=parse_mapper_analysis(batch, response, error=error).model_dump(),
                prompt=prompt,
                started_at=started_at,
                finished_at=utc_now(),
                call=call_record.metrics if call_record else None,
                error=error,
            )

        @step
        async def reducer(
            self,
            ctx: Context,
            ev: MapperResultEvent,
        ) -> StopEvent | None:
            collected = ctx.collect_events(ev, [MapperResultEvent] * len(batches))
            if collected is None:
                return None
            results = sorted(
                collected,
                key=lambda item: int(item.batch["batch_index"]),
            )
            partial_outputs = {
                str(result.batch["batch_id"]): result.partial_output for result in results
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
            mapper_steps = []
            for result in results:
                batch = DocumentBatch.model_validate(result.batch)
                mapper_steps.append(
                    make_mapper_step(
                        batch=batch,
                        prompt=result.prompt,
                        partial_output=result.partial_output,
                        llm_call_id=result.call.call_id if result.call else None,
                        started_at=result.started_at,
                        finished_at=result.finished_at,
                        actor="llamaindex.workflow.equivalent_mapper",
                        framework_primitive=FRAMEWORK_PRIMITIVE,
                        parallelism_used=parallelism_used,
                        error=result.error,
                    )
                )
            reducer_step = make_reducer_step(
                batches=batches,
                prompt=prompt,
                reducer_output=call_record.response,
                partial_outputs=partial_outputs,
                llm_call_id=call_record.metrics.call_id,
                started_at=started_at,
                finished_at=finished_at,
                actor="llamaindex.workflow.reducer",
                framework_primitive=FRAMEWORK_PRIMITIVE,
                parallelism_used=parallelism_used,
            )
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
                framework_execution="llamaindex_dynamic_mapper_event_map_reduce",
                framework_primitive=FRAMEWORK_PRIMITIVE,
                parallelism_used=parallelism_used,
            )
            return StopEvent(
                result=WorkflowPayload(final_answer, structured_output, steps, llm_calls)
            )

    workflow = DocumentMapReduceWorkflow(
        timeout=float(config.timeout_seconds),
        verbose=False,
        num_concurrent_runs=1,
    )

    def execute_workflow() -> LlamaIndexRunOutput:
        async def run_workflow():
            return await workflow.run(input_snapshot=input_data.model_dump(mode="json"))

        payload = run_async(run_workflow())
        if not isinstance(payload, WorkflowPayload):
            raise RuntimeError("LlamaIndex Map-Reduce workflow did not return its payload.")
        return LlamaIndexRunOutput(
            final_answer=payload.final_answer,
            structured_output=payload.structured_output,
            steps=payload.steps,
            llm_calls=payload.llm_calls,
        )

    return run_with_resource_monitor(execute_workflow)
