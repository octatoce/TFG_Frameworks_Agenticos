"""CrewAI implementation for ARCH_10_CHECKPOINT_MEMORY_RECOVERY."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from benchmark_core.checkpoint_memory_recovery import (
    CHECKPOINT_WRITER,
    CONTINUATION_STEP,
    FAILURE_INJECTOR,
    FINALIZER,
    PLANNING_STEP,
    RECOVERY_LOADER,
    STATE_INITIALIZER,
    ControlledFailure,
    RecoveryWorkflowState,
    build_portable_checkpoint,
    build_recovery_structured_output,
    checkpoint_path,
    controlled_failure_message,
    get_recovery_settings,
    initialize_recovery_state,
    load_portable_checkpoint,
    logical_checkpoint_id,
    make_recovery_step,
    parse_continuation_result,
    parse_planning_analysis,
    render_recovery_prompt,
    seal_state_for_checkpoint,
    verify_recovered_state,
    write_portable_checkpoint,
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


FRAMEWORK_PRIMITIVE = "CrewAI.Agent.Task.Crew.portable_json_checkpoint"
CHECKPOINT_BACKEND = "benchmark_core.portable_json_checkpoint"


@crewai_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: CrewAIRunContext,
) -> CrewAIRunOutput:
    """Run two isolated CrewAI tasks around an explicit portable checkpoint."""

    settings = get_recovery_settings(config)
    agents = {
        PLANNING_STEP: create_agent(
            role="Pre-checkpoint Analyst",
            goal="Produce compact evidence-grounded state that can survive recovery.",
            backstory="A stateless analyst that completes only the pre-checkpoint phase.",
            crewai_llm=context.crewai_llm,
            config=config,
            allow_delegation=False,
            max_iter=1,
        ),
        CONTINUATION_STEP: create_agent(
            role="Post-recovery Synthesizer",
            goal="Complete the answer strictly from verified recovered state and common input.",
            backstory="A stateless synthesizer that never repeats the pre-checkpoint task.",
            crewai_llm=context.crewai_llm,
            config=config,
            allow_delegation=False,
            max_iter=1,
        ),
    }

    def record_time(record: Any, key: str, fallback: datetime) -> datetime:
        value = record.metrics.metadata.get(key) if record else None
        return datetime.fromisoformat(value) if isinstance(value, str) else fallback

    def run_task(component: str, prompt: str, expected_output: str):
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
            (record_time(record, "crewai_call_started_at", kickoff_started) for record in records),
            default=kickoff_started,
        )
        finished_at = max(
            (record_time(record, "crewai_call_finished_at", kickoff_finished) for record in records),
            default=kickoff_finished,
        )
        return output, records, started_at, finished_at, error

    steps: list[AgentStep] = []
    llm_calls = []
    checkpoint_file = None
    failure_injected = False
    recovery_attempted = False
    recovery_successful = False

    with ResourceMonitor() as monitor:
        started_at = utc_now()
        state = initialize_recovery_state(input_data)
        finished_at = utc_now()
        steps.append(
            make_recovery_step(
                step_id=1,
                component=STATE_INITIALIZER,
                actor="crewai.external_state_initializer",
                started_at=started_at,
                finished_at=finished_at,
                framework_primitive=FRAMEWORK_PRIMITIVE,
                output={"state": state.model_dump(mode="json")},
            )
        )

        planning_prompt = render_recovery_prompt(input_data, PLANNING_STEP)
        output, records, started_at, finished_at, error = run_task(
            PLANNING_STEP,
            planning_prompt,
            "Exactly ANALYSIS, EVIDENCE, and OPEN_QUESTIONS.",
        )
        planning = parse_planning_analysis(output, error=error)
        state = state.model_copy(update={"planning": planning, "current_stage": PLANNING_STEP})
        for record in records:
            record.metrics.step_id = 2
        llm_calls.extend(record.metrics for record in records)
        steps.append(
            make_recovery_step(
                step_id=2,
                component=PLANNING_STEP,
                actor="crewai.pre_checkpoint_agent_task",
                started_at=started_at,
                finished_at=finished_at,
                framework_primitive=FRAMEWORK_PRIMITIVE,
                input_payload={"prompt": planning_prompt},
                output={"planning": planning.model_dump()},
                llm_call_ids=[record.metrics.call_id for record in records],
                depends_on=[STATE_INITIALIZER],
                error=error,
            )
        )

        checkpoint_started = utc_now()
        checkpoint_id = logical_checkpoint_id(config)
        state = seal_state_for_checkpoint(
            state,
            checkpoint_id=checkpoint_id,
            created_at=checkpoint_started,
        )
        checkpoint = build_portable_checkpoint(
            framework=config.framework,
            config=config,
            state=state,
            created_at=checkpoint_started,
        )
        checkpoint_file = checkpoint_path(context.repo_root, config.framework, checkpoint_id)
        write_portable_checkpoint(checkpoint_file, checkpoint)
        checkpoint_finished = utc_now()
        steps.append(
            make_recovery_step(
                step_id=3,
                component=CHECKPOINT_WRITER,
                actor="crewai.portable_checkpoint_writer",
                started_at=checkpoint_started,
                finished_at=checkpoint_finished,
                framework_primitive=FRAMEWORK_PRIMITIVE,
                input_payload={"state_stage": PLANNING_STEP},
                output={
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_stage": PLANNING_STEP,
                    "checkpoint_path": str(checkpoint_file),
                    "state_digest": state.state_digest,
                },
                depends_on=[PLANNING_STEP],
                checkpoint_backend=CHECKPOINT_BACKEND,
                native_checkpointing=False,
            )
        )

        failure_started = utc_now()
        failure_error = None
        try:
            if settings.inject_failure:
                failure_injected = True
                raise ControlledFailure(controlled_failure_message(checkpoint_id))
        except ControlledFailure as exc:
            failure_error = str(exc)
        failure_finished = utc_now()
        steps.append(
            make_recovery_step(
                step_id=4,
                component=FAILURE_INJECTOR,
                actor="crewai.controlled_failure_injector",
                started_at=failure_started,
                finished_at=failure_finished,
                framework_primitive=FRAMEWORK_PRIMITIVE,
                input_payload={"checkpoint_id": checkpoint_id, "enabled": settings.inject_failure},
                output={"failure_injected": failure_injected, "captured": failure_error is not None},
                depends_on=[CHECKPOINT_WRITER],
                error=failure_error,
            )
        )

        recovery_started = utc_now()
        if failure_injected:
            recovery_attempted = True
            recovered_checkpoint = load_portable_checkpoint(checkpoint_file)
            recovered_state = recovered_checkpoint.state
            verify_recovered_state(recovered_state)
            recovered_state = recovered_state.model_copy(
                update={
                    "current_stage": RECOVERY_LOADER,
                    "recovered": True,
                    "recovery_reason": failure_error,
                }
            )
            recovery_successful = True
        else:
            recovered_state = state
        recovery_finished = utc_now()
        steps.append(
            make_recovery_step(
                step_id=5,
                component=RECOVERY_LOADER,
                actor="crewai.portable_checkpoint_loader",
                started_at=recovery_started,
                finished_at=recovery_finished,
                framework_primitive=FRAMEWORK_PRIMITIVE,
                input_payload={"checkpoint_path": str(checkpoint_file)},
                output={
                    "recovery_attempted": recovery_attempted,
                    "recovery_successful": recovery_successful,
                    "checkpoint_id": checkpoint_id,
                    "state_digest_verified": recovery_successful,
                },
                depends_on=[FAILURE_INJECTOR],
                checkpoint_backend=CHECKPOINT_BACKEND,
                native_checkpointing=False,
            )
        )

        continuation_prompt = render_recovery_prompt(
            input_data,
            CONTINUATION_STEP,
            planning=recovered_state.planning,
        )
        output, records, started_at, finished_at, error = run_task(
            CONTINUATION_STEP,
            continuation_prompt,
            "Exactly ANSWER, DECISION, EVIDENCE, and LIMITATIONS.",
        )
        continuation = parse_continuation_result(output, error=error)
        recovered_state = recovered_state.model_copy(
            update={"continuation": continuation, "current_stage": CONTINUATION_STEP}
        )
        for record in records:
            record.metrics.step_id = 6
        llm_calls.extend(record.metrics for record in records)
        steps.append(
            make_recovery_step(
                step_id=6,
                component=CONTINUATION_STEP,
                actor="crewai.post_recovery_agent_task",
                started_at=started_at,
                finished_at=finished_at,
                framework_primitive=FRAMEWORK_PRIMITIVE,
                input_payload={"prompt": continuation_prompt, "recovered": recovered_state.recovered},
                output={"continuation": continuation.model_dump()},
                llm_call_ids=[record.metrics.call_id for record in records],
                depends_on=[RECOVERY_LOADER],
                error=error,
            )
        )

        finalizer_started = utc_now()
        recovered_state = recovered_state.model_copy(
            update={
                "current_stage": FINALIZER,
                "result_generated_after_recovery": recovery_successful,
            }
        )
        finalizer_finished = utc_now()
        steps.append(
            make_recovery_step(
                step_id=7,
                component=FINALIZER,
                actor="crewai.deterministic_finalizer",
                started_at=finalizer_started,
                finished_at=finalizer_finished,
                framework_primitive=FRAMEWORK_PRIMITIVE,
                output={
                    "answer": continuation.answer,
                    "result_generated_after_recovery": recovery_successful,
                },
                depends_on=[CONTINUATION_STEP],
            )
        )
        resource_usage = monitor.usage

    final_answer, structured_output = build_recovery_structured_output(
        input_data=input_data,
        config=config,
        state=recovered_state,
        steps=steps,
        llm_calls=llm_calls,
        framework_execution="crewai_native_tasks_with_portable_checkpoint_recovery",
        framework_primitive=FRAMEWORK_PRIMITIVE,
        checkpoint_backend=CHECKPOINT_BACKEND,
        native_checkpointing=False,
        recovery_source=str(checkpoint_file),
        failure_injected=failure_injected,
        recovery_attempted=recovery_attempted,
        recovery_successful=recovery_successful,
    )
    return CrewAIRunOutput(
        final_answer=final_answer,
        structured_output=structured_output,
        steps=steps,
        resource_usage=resource_usage,
    )
