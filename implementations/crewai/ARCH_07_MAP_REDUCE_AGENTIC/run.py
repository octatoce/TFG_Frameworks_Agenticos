"""CrewAI implementation for ARCH_07_MAP_REDUCE_AGENTIC."""

from __future__ import annotations

from datetime import datetime

from benchmark_core.map_reduce_agentic import (
    MAPPER,
    REDUCER,
    build_map_reduce_structured_output,
    detect_map_reduce_batch_id,
    detect_map_reduce_component,
    get_batch_size,
    make_mapper_step,
    make_partitioner_step,
    make_reducer_step,
    parse_mapper_analysis,
    partition_documents,
    render_map_reduce_prompt,
)
from benchmark_core.resource_monitor import ResourceMonitor
from benchmark_core.schemas import AgentStep, ExperimentConfig, ExperimentInput
from benchmark_core.tracing import utc_now
from implementations.crewai.utils_crewai import (
    CrewAIRunContext,
    CrewAIRunOutput,
    create_agent,
    create_sequential_crew,
    create_task,
    crewai_architecture_runner,
)


FRAMEWORK_PRIMITIVE = "CrewAI.Task.async_execution.context_map_reduce"
MAPPER_DEFINITION = {
    "role": "Equivalent Document Batch Mapper",
    "goal": "Apply the same extraction logic to exactly one assigned document batch.",
    "backstory": "A stateless mapper in a controlled document-volume Map-Reduce benchmark.",
}
REDUCER_DEFINITION = {
    "role": "Document Map-Reduce Reducer",
    "goal": "Synthesize all mapper outputs once without rereading the complete documents.",
    "backstory": "A one-pass reducer that deduplicates evidence and resolves contradictions.",
}


@crewai_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: CrewAIRunContext,
) -> CrewAIRunOutput:
    """Execute equivalent async mapper tasks followed by one context reducer."""

    partition_started_at = utc_now()
    batches = partition_documents(input_data, config)
    partition_finished_at = utc_now()
    parallelism_used = len(batches) > 1
    partition_step = make_partitioner_step(
        input_data=input_data,
        batches=batches,
        batch_size=get_batch_size(config),
        started_at=partition_started_at,
        finished_at=partition_finished_at,
        actor="crewai.document_partitioner",
        framework_primitive=FRAMEWORK_PRIMITIVE,
    )
    mapper_agents = [
        create_agent(
            **MAPPER_DEFINITION,
            crewai_llm=context.crewai_llm,
            config=config,
            allow_delegation=False,
        )
        for _batch in batches
    ]
    reducer_agent = create_agent(
        **REDUCER_DEFINITION,
        crewai_llm=context.crewai_llm,
        config=config,
        allow_delegation=False,
    )
    mapper_tasks = [
        create_task(
            description=render_map_reduce_prompt(input_data, MAPPER, batch=batch),
            expected_output="The four labeled mapper fields for this batch only.",
            agent=agent,
            config=config,
            context=[],
            async_execution=True,
        )
        for batch, agent in zip(batches, mapper_agents, strict=True)
    ]
    context_placeholders = {
        batch.batch_id: {"source": f"CrewAI context from mapper for {batch.batch_id}"}
        for batch in batches
    }
    reducer_task = create_task(
        description=render_map_reduce_prompt(
            input_data,
            REDUCER,
            partial_outputs=context_placeholders,
        ),
        expected_output="The four labeled reducer fields and one final answer.",
        agent=reducer_agent,
        config=config,
        context=mapper_tasks,
        async_execution=False,
    )
    crew = create_sequential_crew(
        agents=[*mapper_agents, reducer_agent],
        tasks=[*mapper_tasks, reducer_task],
    )

    kickoff_started = utc_now()
    with ResourceMonitor() as monitor:
        crew.kickoff()
        resource_usage = monitor.usage
    kickoff_finished = utc_now()

    partial_outputs = {
        batch.batch_id: parse_mapper_analysis(
            batch,
            str(task.output).strip() if task.output is not None else "",
        ).model_dump()
        for batch, task in zip(batches, mapper_tasks, strict=True)
    }
    reducer_output = str(reducer_task.output).strip() if reducer_task.output is not None else ""
    mapper_records = {batch.batch_id: [] for batch in batches}
    reducer_records = []
    for record in context.crewai_llm.call_records:
        component = detect_map_reduce_component(record.prompt)
        if component == MAPPER:
            batch_id = detect_map_reduce_batch_id(record.prompt)
            if batch_id in mapper_records:
                mapper_records[batch_id].append(record)
        elif component == REDUCER:
            reducer_records.append(record)

    def record_time(record, key: str, fallback):
        value = record.metrics.metadata.get(key) if record else None
        return datetime.fromisoformat(value) if isinstance(value, str) else fallback

    mapper_steps: list[AgentStep] = []
    mapper_finished_times = []
    for batch, task in zip(batches, mapper_tasks, strict=True):
        records = mapper_records[batch.batch_id]
        for record in records:
            record.metrics.step_id = batch.batch_index + 2
        started_at = min(
            (
                record_time(record, "crewai_call_started_at", kickoff_started)
                for record in records
            ),
            default=kickoff_started,
        )
        finished_at = max(
            (
                record_time(record, "crewai_call_finished_at", kickoff_finished)
                for record in records
            ),
            default=kickoff_finished,
        )
        mapper_finished_times.append(finished_at)
        step = make_mapper_step(
            batch=batch,
            prompt=task.description,
            partial_output=partial_outputs[batch.batch_id],
            llm_call_id=records[0].metrics.call_id if records else None,
            started_at=started_at,
            finished_at=finished_at,
            actor="crewai.equivalent_mapper_agent_task",
            framework_primitive=FRAMEWORK_PRIMITIVE,
            parallelism_used=parallelism_used,
        )
        if len(records) > 1:
            step.llm_call_ids = [record.metrics.call_id for record in records]
        mapper_steps.append(step)

    reducer_step_id = len(batches) + 2
    for record in reducer_records:
        record.metrics.step_id = reducer_step_id
    last_mapper_finished = max(mapper_finished_times, default=kickoff_started)
    reducer_step = make_reducer_step(
        batches=batches,
        prompt=reducer_task.description,
        reducer_output=reducer_output,
        partial_outputs=partial_outputs,
        llm_call_id=reducer_records[0].metrics.call_id,
        started_at=min(
            (
                record_time(record, "crewai_call_started_at", last_mapper_finished)
                for record in reducer_records
            ),
            default=last_mapper_finished,
        ),
        finished_at=max(
            (
                record_time(record, "crewai_call_finished_at", kickoff_finished)
                for record in reducer_records
            ),
            default=kickoff_finished,
        ),
        actor="crewai.reducer_agent_task",
        framework_primitive=FRAMEWORK_PRIMITIVE,
        parallelism_used=parallelism_used,
    )
    steps = [partition_step, *mapper_steps, reducer_step]
    llm_calls = [
        record.metrics
        for batch in batches
        for record in mapper_records[batch.batch_id]
    ] + [record.metrics for record in reducer_records]
    final_answer, structured_output = build_map_reduce_structured_output(
        input_data=input_data,
        config=config,
        batches=batches,
        partial_outputs=partial_outputs,
        reducer_output=reducer_output,
        steps=steps,
        llm_calls=llm_calls,
        framework_execution="crewai_async_mapper_tasks_context_reducer",
        framework_primitive=FRAMEWORK_PRIMITIVE,
        parallelism_used=parallelism_used,
    )
    return CrewAIRunOutput(
        final_answer=final_answer,
        structured_output=structured_output,
        steps=steps,
        resource_usage=resource_usage,
    )
