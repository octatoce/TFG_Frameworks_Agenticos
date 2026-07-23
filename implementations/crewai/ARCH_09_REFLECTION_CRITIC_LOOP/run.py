"""CrewAI implementation for ARCH_09_REFLECTION_CRITIC_LOOP."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from benchmark_core.reflection_critic_loop import (
    CRITIC,
    GENERATOR,
    REVISER,
    STOP_CONTROLLER,
    build_reflection_structured_output,
    evaluate_stop,
    get_reflection_settings,
    make_reflection_step,
    parse_critique,
    parse_reflection_version,
    reflection_step_name,
    render_reflection_prompt,
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


FRAMEWORK_PRIMITIVE = "bounded_python_for.CrewAI_Agent_Task_Crew_per_component"

AGENT_DEFINITIONS = {
    GENERATOR: {
        "role": "Initial Answer Generator",
        "goal": "Produce one evidence-grounded initial answer in the required structured format.",
        "backstory": "A concise analyst who creates version zero without self-critique or delegation.",
    },
    CRITIC: {
        "role": "Structured Answer Critic",
        "goal": "Detect material defects and propose concrete improvements without rewriting the answer.",
        "backstory": "A rigorous reviewer focused on evidence, ambiguity, confidence, and format validity.",
    },
    REVISER: {
        "role": "Controlled Answer Reviser",
        "goal": "Correct exactly one evolving answer from the latest structured critique.",
        "backstory": "A disciplined editor who preserves useful content and never starts another workflow.",
    },
}


@crewai_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: CrewAIRunContext,
) -> CrewAIRunOutput:
    """Execute bounded CrewAI tasks under a minimal explicit Python for-loop."""

    settings = get_reflection_settings(config)
    agents = {
        component: create_agent(
            **definition,
            crewai_llm=context.crewai_llm,
            config=config,
            allow_delegation=False,
            max_iter=1,
        )
        for component, definition in AGENT_DEFINITIONS.items()
    }

    def record_time(record: Any, key: str, fallback: datetime) -> datetime:
        value = record.metrics.metadata.get(key) if record else None
        return datetime.fromisoformat(value) if isinstance(value, str) else fallback

    def run_task(
        component: str,
        prompt: str,
        expected_output: str,
    ) -> tuple[str, list[Any], datetime, datetime, str | None]:
        task = create_task(
            description=prompt,
            expected_output=expected_output,
            agent=agents[component],
            config=config,
            context=[],
            async_execution=False,
        )
        crew = create_sequential_crew(agents=[agents[component]], tasks=[task])
        before = len(context.crewai_llm.call_records)
        kickoff_started = utc_now()
        error = None
        try:
            crew.kickoff()
            output = str(task.output).strip() if task.output is not None else ""
        except Exception as exc:  # pragma: no cover - integration failure path
            error = f"{type(exc).__name__}: {exc}"
            output = ""
        kickoff_finished = utc_now()
        records = list(context.crewai_llm.call_records[before:])
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
        return output, records, started_at, finished_at, error

    versions = []
    critiques = []
    stop_decisions = []
    steps: list[AgentStep] = []
    llm_calls = []

    with ResourceMonitor() as monitor:
        generator_prompt = render_reflection_prompt(
            input_data,
            GENERATOR,
            iteration=0,
            settings=settings,
        )
        output, records, started_at, finished_at, error = run_task(
            GENERATOR,
            generator_prompt,
            "Exactly ANSWER, EVIDENCE, CONFIDENCE, and LIMITATIONS for version zero.",
        )
        version = parse_reflection_version(
            output,
            version_index=0,
            iteration=0,
            created_by=GENERATOR,
            error=error,
        )
        for record in records:
            record.metrics.step_id = 1
        steps.append(
            make_reflection_step(
                step_id=1,
                component=GENERATOR,
                iteration=0,
                actor="crewai.generator_agent_task",
                prompt=generator_prompt,
                output={"version": version.model_dump()},
                llm_call_ids=[record.metrics.call_id for record in records],
                started_at=started_at,
                finished_at=finished_at,
                framework_primitive=FRAMEWORK_PRIMITIVE,
                max_iterations=settings.max_iterations,
                error=error,
            )
        )
        versions.append(version)
        llm_calls.extend(record.metrics for record in records)
        current_version = version

        for iteration in range(1, settings.max_iterations + 1):
            critic_prompt = render_reflection_prompt(
                input_data,
                CRITIC,
                iteration=iteration,
                settings=settings,
                current_version=current_version.model_dump(),
            )
            output, records, started_at, finished_at, error = run_task(
                CRITIC,
                critic_prompt,
                (
                    "Exactly NO_CRITICAL_ISSUES, SCORE, SEVERITY, ISSUES, IMPROVEMENTS, "
                    "EVIDENCE_GAPS, and FORMAT_VALID."
                ),
            )
            critique = parse_critique(output, iteration=iteration, error=error)
            critic_step_id = len(steps) + 1
            for record in records:
                record.metrics.step_id = critic_step_id
            previous_step = (
                GENERATOR
                if current_version.created_by == GENERATOR
                else reflection_step_name(REVISER, current_version.iteration)
            )
            steps.append(
                make_reflection_step(
                    step_id=critic_step_id,
                    component=CRITIC,
                    iteration=iteration,
                    actor="crewai.critic_agent_task",
                    prompt=critic_prompt,
                    current_version=current_version.model_dump(),
                    output={"critique": critique.model_dump()},
                    llm_call_ids=[record.metrics.call_id for record in records],
                    started_at=started_at,
                    finished_at=finished_at,
                    framework_primitive=FRAMEWORK_PRIMITIVE,
                    max_iterations=settings.max_iterations,
                    depends_on=[previous_step],
                    error=error,
                )
            )
            critiques.append(critique)
            llm_calls.extend(record.metrics for record in records)

            stop_started_at = utc_now()
            decision = evaluate_stop(
                critique,
                current_version_index=current_version.version_index,
                settings=settings,
            )
            stop_finished_at = utc_now()
            steps.append(
                make_reflection_step(
                    step_id=len(steps) + 1,
                    component=STOP_CONTROLLER,
                    iteration=iteration,
                    actor="crewai.deterministic_stop_controller",
                    current_version=current_version.model_dump(),
                    critique=critique.model_dump(),
                    output={"stop_decision": decision.model_dump()},
                    started_at=stop_started_at,
                    finished_at=stop_finished_at,
                    framework_primitive=FRAMEWORK_PRIMITIVE,
                    max_iterations=settings.max_iterations,
                    depends_on=[reflection_step_name(CRITIC, iteration)],
                )
            )
            stop_decisions.append(decision)
            if decision.should_stop:
                break

            reviser_prompt = render_reflection_prompt(
                input_data,
                REVISER,
                iteration=iteration,
                settings=settings,
                current_version=current_version.model_dump(),
                critique=critique.model_dump(),
            )
            output, records, started_at, finished_at, error = run_task(
                REVISER,
                reviser_prompt,
                "Exactly ANSWER, EVIDENCE, CONFIDENCE, LIMITATIONS, and CHANGES_APPLIED.",
            )
            revised_version = parse_reflection_version(
                output,
                version_index=len(versions),
                iteration=iteration,
                created_by=REVISER,
                error=error,
            )
            reviser_step_id = len(steps) + 1
            for record in records:
                record.metrics.step_id = reviser_step_id
            steps.append(
                make_reflection_step(
                    step_id=reviser_step_id,
                    component=REVISER,
                    iteration=iteration,
                    actor="crewai.reviser_agent_task",
                    prompt=reviser_prompt,
                    current_version=current_version.model_dump(),
                    critique=critique.model_dump(),
                    stop_decision=decision.model_dump(),
                    output={"version": revised_version.model_dump()},
                    llm_call_ids=[record.metrics.call_id for record in records],
                    started_at=started_at,
                    finished_at=finished_at,
                    framework_primitive=FRAMEWORK_PRIMITIVE,
                    max_iterations=settings.max_iterations,
                    depends_on=[
                        reflection_step_name(CRITIC, iteration),
                        reflection_step_name(STOP_CONTROLLER, iteration),
                    ],
                    error=error,
                )
            )
            versions.append(revised_version)
            current_version = revised_version
            llm_calls.extend(record.metrics for record in records)

        resource_usage = monitor.usage

    final_answer, structured_output = build_reflection_structured_output(
        input_data=input_data,
        config=config,
        versions=versions,
        critiques=critiques,
        stop_decisions=stop_decisions,
        steps=steps,
        llm_calls=llm_calls,
        settings=settings,
        framework_execution="crewai_bounded_external_loop_with_native_tasks",
        framework_primitive=FRAMEWORK_PRIMITIVE,
    )
    return CrewAIRunOutput(
        final_answer=final_answer,
        structured_output=structured_output,
        steps=steps,
        resource_usage=resource_usage,
    )
