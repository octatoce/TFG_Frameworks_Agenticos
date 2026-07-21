"""CrewAI implementation for ARCH_08_DEBATE_JUDGE."""

from __future__ import annotations

from datetime import datetime

from benchmark_core.debate_judge import (
    DEBATERS,
    DEBATE_ROUND,
    JUDGE,
    build_debate_structured_output,
    detect_debate_component,
    make_debate_step,
    parse_debate_proposal,
    parse_debate_round,
    parse_judge_decision,
    render_debate_judge_prompt,
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


FRAMEWORK_PRIMITIVE = "Task.async_debaters.context_debate.context_judge"

AGENT_DEFINITIONS = {
    "debater_a": {
        "role": "Evidence-Grounded Solution Debater",
        "goal": "Produce the strongest direct proposal with traceable evidence and explicit assumptions.",
        "backstory": "A precise analyst who argues one independent position without seeing peer proposals.",
    },
    "debater_b": {
        "role": "Alternative Interpretation Debater",
        "goal": "Develop a genuinely distinct interpretation or solution that remains grounded in the case.",
        "backstory": "A disciplined contrarian who adds diversity without inventing unsupported facts.",
    },
    "debater_c": {
        "role": "Conservative Pragmatic Debater",
        "goal": "Propose an actionable but cautious answer that foregrounds constraints and failure modes.",
        "backstory": "A risk-aware practitioner who separates confirmed facts, assumptions, and uncertainty.",
    },
    "debate_round": {
        "role": "Cross-Critique Moderator",
        "goal": "Run exactly one concise cross-review of all proposals without making the final decision.",
        "backstory": "A neutral debate analyst who exposes strengths, weaknesses, and concrete disagreements.",
    },
    "judge": {
        "role": "Final Debate Judge",
        "goal": "Select, combine, or reject proposals and own the only final decision with a brief rationale.",
        "backstory": "An impartial adjudicator who weighs both initial arguments and the explicit critique round.",
    },
}


@crewai_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: CrewAIRunContext,
) -> CrewAIRunOutput:
    """Execute async proposals, one context-bound debate task, and one judge task."""

    agents = {
        component: create_agent(
            **definition,
            crewai_llm=context.crewai_llm,
            config=config,
            allow_delegation=False,
        )
        for component, definition in AGENT_DEFINITIONS.items()
    }
    proposal_tasks = [
        create_task(
            description=render_debate_judge_prompt(input_data, debater),
            expected_output=(
                "Exactly the five labeled proposal fields, representing one independent position only."
            ),
            agent=agents[debater],
            config=config,
            context=[],
            async_execution=True,
        )
        for debater in DEBATERS
    ]

    proposal_placeholders = {
        debater: {"source": f"CrewAI context from {debater}"} for debater in DEBATERS
    }
    debate_task = create_task(
        description=render_debate_judge_prompt(
            input_data,
            DEBATE_ROUND,
            proposals=proposal_placeholders,
        ),
        expected_output=(
            "Exactly three labeled cross-critiques plus consensus, disagreements, strongest points, and weakest points."
        ),
        agent=agents[DEBATE_ROUND],
        config=config,
        context=proposal_tasks,
        async_execution=False,
    )
    judge_task = create_task(
        description=render_debate_judge_prompt(
            input_data,
            JUDGE,
            proposals=proposal_placeholders,
            debate={"source": "CrewAI context from debate_round"},
        ),
        expected_output="Exactly the seven labeled judge fields and one final answer.",
        agent=agents[JUDGE],
        config=config,
        context=[*proposal_tasks, debate_task],
        async_execution=False,
    )
    crew = create_sequential_crew(
        agents=list(agents.values()),
        tasks=[*proposal_tasks, debate_task, judge_task],
    )

    kickoff_started = utc_now()
    with ResourceMonitor() as monitor:
        crew.kickoff()
        resource_usage = monitor.usage
    kickoff_finished = utc_now()

    proposals = {
        debater: parse_debate_proposal(
            debater,
            str(task.output).strip() if task.output is not None else "",
        ).model_dump()
        for debater, task in zip(DEBATERS, proposal_tasks, strict=True)
    }
    debate_raw = str(debate_task.output).strip() if debate_task.output is not None else ""
    judge_raw = str(judge_task.output).strip() if judge_task.output is not None else ""
    debate_output = parse_debate_round(debate_raw).model_dump()
    judge_decision = parse_judge_decision(judge_raw).model_dump()

    records_by_component = {component: [] for component in (*DEBATERS, DEBATE_ROUND, JUDGE)}
    for record in context.crewai_llm.call_records:
        component = detect_debate_component(record.prompt)
        if component in records_by_component:
            records_by_component[component].append(record)

    def record_time(record, key: str, fallback):
        value = record.metrics.metadata.get(key) if record else None
        return datetime.fromisoformat(value) if isinstance(value, str) else fallback

    proposal_finished_times = {
        debater: max(
            (
                record_time(record, "crewai_call_finished_at", kickoff_finished)
                for record in records_by_component[debater]
            ),
            default=kickoff_finished,
        )
        for debater in DEBATERS
    }
    last_proposal_finished = max(proposal_finished_times.values(), default=kickoff_started)
    steps: list[AgentStep] = []
    for step_id, debater in enumerate(DEBATERS, start=1):
        records = records_by_component[debater]
        for record in records:
            record.metrics.step_id = step_id
        steps.append(
            make_debate_step(
                step_id=step_id,
                component=debater,
                actor=f"crewai.{debater}_agent_task",
                prompt=proposal_tasks[step_id - 1].description,
                output={"proposal": proposals[debater]},
                llm_call_ids=[record.metrics.call_id for record in records],
                started_at=min(
                    (
                        record_time(record, "crewai_call_started_at", kickoff_started)
                        for record in records
                    ),
                    default=kickoff_started,
                ),
                finished_at=proposal_finished_times[debater],
                framework_primitive=FRAMEWORK_PRIMITIVE,
            )
        )

    debate_records = records_by_component[DEBATE_ROUND]
    for record in debate_records:
        record.metrics.step_id = 4
    debate_finished = max(
        (
            record_time(record, "crewai_call_finished_at", kickoff_finished)
            for record in debate_records
        ),
        default=kickoff_finished,
    )
    steps.append(
        make_debate_step(
            step_id=4,
            component=DEBATE_ROUND,
            actor="crewai.debate_round_agent_task",
            prompt=debate_task.description,
            output={"debate_round": debate_output},
            llm_call_ids=[record.metrics.call_id for record in debate_records],
            started_at=min(
                (
                    record_time(record, "crewai_call_started_at", last_proposal_finished)
                    for record in debate_records
                ),
                default=last_proposal_finished,
            ),
            finished_at=debate_finished,
            framework_primitive=FRAMEWORK_PRIMITIVE,
            proposals=proposals,
        )
    )

    judge_records = records_by_component[JUDGE]
    for record in judge_records:
        record.metrics.step_id = 5
    steps.append(
        make_debate_step(
            step_id=5,
            component=JUDGE,
            actor="crewai.judge_agent_task",
            prompt=judge_task.description,
            output={"judge_decision": judge_decision},
            llm_call_ids=[record.metrics.call_id for record in judge_records],
            started_at=min(
                (
                    record_time(record, "crewai_call_started_at", debate_finished)
                    for record in judge_records
                ),
                default=debate_finished,
            ),
            finished_at=max(
                (
                    record_time(record, "crewai_call_finished_at", kickoff_finished)
                    for record in judge_records
                ),
                default=kickoff_finished,
            ),
            framework_primitive=FRAMEWORK_PRIMITIVE,
            proposals=proposals,
            debate=debate_output,
        )
    )

    llm_calls = [
        record.metrics
        for component in (*DEBATERS, DEBATE_ROUND, JUDGE)
        for record in records_by_component[component]
    ]
    final_answer, structured_output = build_debate_structured_output(
        input_data=input_data,
        config=config,
        proposals=proposals,
        debate_output=debate_raw,
        judge_output=judge_raw,
        steps=steps,
        llm_calls=llm_calls,
        framework_execution="crewai_async_tasks_single_debate_and_judge",
        framework_primitive=FRAMEWORK_PRIMITIVE,
    )
    return CrewAIRunOutput(
        final_answer=final_answer,
        structured_output=structured_output,
        steps=steps,
        resource_usage=resource_usage,
    )
