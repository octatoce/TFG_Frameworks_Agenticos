"""LangGraph implementation for ARCH_07_MAP_REDUCE_AGENTIC."""

from __future__ import annotations

import asyncio
import operator
from typing import Annotated, Any, TypedDict

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
from implementations.langgraph.utils_langgraph import (
    LangGraphRunContext,
    LangGraphRunOutput,
    complete_llm_step,
    langgraph_architecture_runner,
)


FRAMEWORK_PRIMITIVE = "StateGraph.Send.dynamic_map.reducer"


class MapperExecution:
    def __init__(
        self,
        *,
        batch,
        partial_output,
        prompt,
        started_at,
        finished_at,
        call=None,
        error=None,
    ) -> None:
        self.batch = batch
        self.partial_output = partial_output
        self.prompt = prompt
        self.started_at = started_at
        self.finished_at = finished_at
        self.call = call
        self.error = error


MapReduceState = TypedDict(
    "MapReduceState",
    {
        "batches": list[object],
        "partition_step": object,
        "mapper_results": Annotated[list[object], operator.add],
        "final_answer": str,
        "structured_output": dict[str, object],
        "steps": list[object],
        "llm_calls": list[object],
    },
    total=False,
)
MapperState = TypedDict("MapperState", {"batch": object})


@langgraph_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: LangGraphRunContext,
) -> LangGraphRunOutput:
    """Execute a dynamic Send map and one reducer in StateGraph."""

    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Send

    batches = partition_documents(input_data, config)
    parallelism_used = len(batches) > 1

    def document_partitioner(_: MapReduceState) -> MapReduceState:
        started_at = utc_now()
        partitioned = partition_documents(input_data, config)
        finished_at = utc_now()
        return {
            "batches": partitioned,
            "partition_step": make_partitioner_step(
                input_data=input_data,
                batches=partitioned,
                batch_size=get_batch_size(config),
                started_at=started_at,
                finished_at=finished_at,
                actor="langgraph.document_partitioner_node",
                framework_primitive=FRAMEWORK_PRIMITIVE,
            ),
        }

    def dispatch_batches(state: MapReduceState) -> list[object]:
        return [Send(MAPPER, {"batch": batch}) for batch in state["batches"]]

    def mapper(state: MapperState) -> MapReduceState:
        batch = state["batch"]
        prompt = render_map_reduce_prompt(input_data, MAPPER, batch=batch)
        started_at = utc_now()
        call_record = None
        error = None
        try:
            call_record = complete_llm_step(
                llm=context.llm,
                input_data=input_data,
                config=config,
                prompt=prompt,
                step_id=batch.batch_index + 2,
            )
            response = call_record.response.strip()
        except Exception as exc:  # pragma: no cover - integration failure path
            error = f"{type(exc).__name__}: {exc}"
            response = ""
        return {
            "mapper_results": [
                MapperExecution(
                    batch=batch,
                    partial_output=parse_mapper_analysis(batch, response, error=error).model_dump(),
                    prompt=prompt,
                    started_at=started_at,
                    finished_at=utc_now(),
                    call=call_record.metrics if call_record else None,
                    error=error,
                )
            ]
        }

    def reducer(state: MapReduceState) -> MapReduceState:
        results = sorted(state["mapper_results"], key=lambda item: item.batch.batch_index)
        partial_outputs = {
            result.batch.batch_id: result.partial_output for result in results
        }
        prompt = render_map_reduce_prompt(
            input_data,
            REDUCER,
            partial_outputs=partial_outputs,
        )
        started_at = utc_now()
        call_record = complete_llm_step(
            llm=context.llm,
            input_data=input_data,
            config=config,
            prompt=prompt,
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
                actor="langgraph.mapper_node",
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
            actor="langgraph.reducer_node",
            framework_primitive=FRAMEWORK_PRIMITIVE,
            parallelism_used=parallelism_used,
        )
        steps = [state["partition_step"], *mapper_steps, reducer_step]
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
            framework_execution="langgraph_stategraph_dynamic_send_map_reduce",
            framework_primitive=FRAMEWORK_PRIMITIVE,
            parallelism_used=parallelism_used,
        )
        return {
            "final_answer": final_answer,
            "structured_output": structured_output,
            "steps": steps,
            "llm_calls": llm_calls,
        }

    graph = StateGraph(MapReduceState)
    graph.add_node("document_partitioner", document_partitioner)
    graph.add_node(MAPPER, mapper)
    graph.add_node(REDUCER, reducer)
    graph.add_edge(START, "document_partitioner")
    graph.add_conditional_edges("document_partitioner", dispatch_batches, [MAPPER])
    graph.add_edge(MAPPER, REDUCER)
    graph.add_edge(REDUCER, END)

    async def invoke_graph():
        return await asyncio.wait_for(
            graph.compile().ainvoke({"mapper_results": []}),
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
