"""CrewAI implementation for ARCH_06_PARALLEL_FANOUT_FANIN."""

from __future__ import annotations

from datetime import datetime

from benchmark_core.parallel_fanout_fanin import (
    AGGREGATOR,
    PARALLEL_BRANCHES,
    build_parallel_structured_output,
    detect_parallel_component,
    make_parallel_step,
    parse_branch_analysis,
    render_parallel_fanout_fanin_prompt,
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


FRAMEWORK_PRIMITIVE = "Task.async_execution.context_fanin"

AGENT_DEFINITIONS = {
    "factual_analysis_branch": {
        "role": "Independent Factual Analysis Specialist",
        "goal": "Extract objective facts and internal evidence without drafting the complete answer.",
        "backstory": "A precise evidence analyst working independently in a controlled fan-out benchmark.",
    },
    "technical_reasoning_branch": {
        "role": "Independent Technical Reasoning Specialist",
        "goal": "Produce a technically reasoned preliminary solution directly from the common input.",
        "backstory": "A technical decision analyst isolated from the other fan-out perspectives.",
    },
    "risk_constraints_branch": {
        "role": "Independent Risk and Constraints Specialist",
        "goal": "Identify risks, limitations, uncertainty, contradictions, and edge conditions.",
        "backstory": "A cautious risk analyst who reports constraints without becoming the final judge.",
    },
    "alternative_solution_branch": {
        "role": "Independent Alternative Solution Specialist",
        "goal": "Develop a genuinely different interpretation or complementary solution.",
        "backstory": "A creative but disciplined specialist responsible for diversity of approach.",
    },
    "aggregator": {
        "role": "Parallel Results Aggregator",
        "goal": "Fuse exactly four independent outputs into one deduplicated and qualified final answer.",
        "backstory": "A one-pass synthesis specialist with no authority to delegate, supervise, or iterate.",
    },
}


@crewai_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: CrewAIRunContext,
) -> CrewAIRunOutput:
    """Execute four native async CrewAI tasks followed by one context fan-in task."""

    agents = {
        component: create_agent(
            **definition,
            crewai_llm=context.crewai_llm,
            config=config,
            allow_delegation=False,
        )
        for component, definition in AGENT_DEFINITIONS.items()
    }
    branch_tasks = []
    for branch in PARALLEL_BRANCHES:
        branch_tasks.append(
            create_task(
                description=render_parallel_fanout_fanin_prompt(input_data, branch),
                expected_output=(
                    "The five labeled branch fields only; no complete final answer and no use of other branches."
                ),
                agent=agents[branch],
                config=config,
                context=[],
                async_execution=True,
            )
        )

    # CrewAI injects the actual four Task outputs through context after all
    # async futures complete.  The keyed placeholders keep the canonical
    # aggregator prompt explicit before kickoff.
    context_placeholders = {
        branch: {"source": f"CrewAI context from {branch}"} for branch in PARALLEL_BRANCHES
    }
    aggregator_task = create_task(
        description=render_parallel_fanout_fanin_prompt(
            input_data,
            AGGREGATOR,
            context_placeholders,
        ),
        expected_output="The four labeled aggregator fields and one final answer.",
        agent=agents[AGGREGATOR],
        config=config,
        context=branch_tasks,
        async_execution=False,
    )
    crew = create_sequential_crew(
        agents=list(agents.values()),
        tasks=[*branch_tasks, aggregator_task],
    )

    kickoff_started = utc_now()
    with ResourceMonitor() as monitor:
        crew.kickoff()
        resource_usage = monitor.usage
    kickoff_finished = utc_now()

    partial_outputs = {
        branch: parse_branch_analysis(
            branch,
            str(task.output).strip() if task.output is not None else "",
        ).model_dump()
        for branch, task in zip(PARALLEL_BRANCHES, branch_tasks, strict=True)
    }
    aggregator_output = (
        str(aggregator_task.output).strip() if aggregator_task.output is not None else ""
    )

    records_by_component = {component: [] for component in (*PARALLEL_BRANCHES, AGGREGATOR)}
    for record in context.crewai_llm.call_records:
        component = detect_parallel_component(record.prompt)
        if component in records_by_component:
            records_by_component[component].append(record)

    def record_time(record, key: str, fallback):
        value = record.metrics.metadata.get(key) if record else None
        return datetime.fromisoformat(value) if isinstance(value, str) else fallback

    branch_finished_times = {
        branch: max(
            (
                record_time(record, "crewai_call_finished_at", kickoff_finished)
                for record in records_by_component[branch]
            ),
            default=kickoff_finished,
        )
        for branch in PARALLEL_BRANCHES
    }
    last_branch_finished = max(branch_finished_times.values(), default=kickoff_started)
    steps: list[AgentStep] = []
    for step_id, branch in enumerate(PARALLEL_BRANCHES, start=1):
        records = records_by_component[branch]
        for record in records:
            record.metrics.step_id = step_id
        step = make_parallel_step(
            step_id=step_id,
            component=branch,
            actor=f"crewai.{branch}_agent_task",
            prompt=branch_tasks[step_id - 1].description,
            output={"partial_output": partial_outputs[branch]},
            llm_call_id=records[0].metrics.call_id if records else None,
            started_at=min(
                (
                    record_time(record, "crewai_call_started_at", kickoff_started)
                    for record in records
                ),
                default=kickoff_started,
            ),
            finished_at=branch_finished_times[branch],
            framework_primitive=FRAMEWORK_PRIMITIVE,
            parallelism_used=True,
        )
        if len(records) > 1:
            step.llm_call_ids = [record.metrics.call_id for record in records]
        steps.append(step)

    aggregator_records = records_by_component[AGGREGATOR]
    for record in aggregator_records:
        record.metrics.step_id = 5
    aggregator_step = make_parallel_step(
        step_id=5,
        component=AGGREGATOR,
        actor="crewai.aggregator_agent_task",
        prompt=aggregator_task.description,
        output={"aggregator_output": aggregator_output},
        llm_call_id=aggregator_records[0].metrics.call_id if aggregator_records else None,
        started_at=min(
            (
                record_time(record, "crewai_call_started_at", last_branch_finished)
                for record in aggregator_records
            ),
            default=last_branch_finished,
        ),
        finished_at=max(
            (
                record_time(record, "crewai_call_finished_at", kickoff_finished)
                for record in aggregator_records
            ),
            default=kickoff_finished,
        ),
        framework_primitive=FRAMEWORK_PRIMITIVE,
        parallelism_used=True,
        partial_outputs=partial_outputs,
    )
    if len(aggregator_records) > 1:
        aggregator_step.llm_call_ids = [record.metrics.call_id for record in aggregator_records]
    steps.append(aggregator_step)

    llm_calls = [
        record.metrics
        for component in (*PARALLEL_BRANCHES, AGGREGATOR)
        for record in records_by_component[component]
    ]
    final_answer, structured_output = build_parallel_structured_output(
        input_data=input_data,
        config=config,
        partial_outputs=partial_outputs,
        aggregator_output=aggregator_output,
        steps=steps,
        llm_calls=llm_calls,
        framework_execution="crewai_native_async_tasks_context_fanin",
        framework_primitive=FRAMEWORK_PRIMITIVE,
        parallelism_used=True,
    )
    return CrewAIRunOutput(
        final_answer=final_answer,
        structured_output=structured_output,
        steps=steps,
        resource_usage=resource_usage,
    )
