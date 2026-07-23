"""Microsoft Agent Framework implementation for ARCH_10."""

import asyncio
from pathlib import Path
from typing import Never

from benchmark_core.checkpoint_memory_recovery import (
    ARCHITECTURE,
    CHECKPOINT_WRITER,
    CONTINUATION_STEP,
    FAILURE_INJECTOR,
    FINALIZER,
    PLANNING_STEP,
    RECOVERY_LOADER,
    STATE_INITIALIZER,
    ControlledFailure,
    RecoveryWorkflowState,
    build_recovery_structured_output,
    checkpoint_directory,
    controlled_failure_message,
    get_recovery_settings,
    initialize_recovery_state,
    logical_checkpoint_id,
    make_recovery_step,
    parse_continuation_result,
    parse_planning_analysis,
    render_recovery_prompt,
    seal_state_for_checkpoint,
    verify_recovered_state,
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


FRAMEWORK_PRIMITIVE = "WorkflowBuilder.FileCheckpointStorage.restore_from_checkpoint"
CHECKPOINT_BACKEND = "agent_framework.FileCheckpointStorage"


@microsoft_agent_framework_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: MicrosoftAgentFrameworkRunContext,
) -> MicrosoftAgentFrameworkRunOutput:
    """Interrupt a native checkpointed workflow and restore its pending state."""

    settings = get_recovery_settings(config)
    logical_id = logical_checkpoint_id(config)

    def execute() -> MicrosoftAgentFrameworkRunOutput:
        from agent_framework import (
            Executor,
            FileCheckpointStorage,
            WorkflowBuilder,
            WorkflowContext,
            handler,
        )

        agents = {
            component: build_agent(
                name=component,
                instructions=(
                    f"You are the {component} component in ARCH_10. "
                    "Run one bounded phase without routing, handoffs, debate, or hidden retries."
                ),
                context=context,
                input_data=input_data,
                config=config,
            )
            for component in (PLANNING_STEP, CONTINUATION_STEP)
        }
        recovery_mode = False
        failure_error: str | None = None
        injected_started_at = None
        injected_finished_at = None
        selected_native_checkpoint_id: str | None = None

        class StateInitializerExecutor(Executor):
            @handler
            async def initialize(self, _message: dict, ctx: WorkflowContext[dict]) -> None:
                started_at = utc_now()
                state = initialize_recovery_state(input_data)
                finished_at = utc_now()
                step = make_recovery_step(
                    step_id=1,
                    component=STATE_INITIALIZER,
                    actor="microsoft_agent_framework.state_initializer_executor",
                    started_at=started_at,
                    finished_at=finished_at,
                    framework_primitive=FRAMEWORK_PRIMITIVE,
                    output={"state": state.model_dump(mode="json")},
                )
                await ctx.send_message(
                    {
                        "workflow_state": state.model_dump(mode="json"),
                        "steps": [step.model_dump(mode="json")],
                        "llm_calls": [],
                    }
                )

        class PlanningExecutor(Executor):
            @handler
            async def plan(self, payload: dict, ctx: WorkflowContext[dict]) -> None:
                state = RecoveryWorkflowState.model_validate(payload["workflow_state"])
                prompt = render_recovery_prompt(input_data, PLANNING_STEP)
                started_at = utc_now()
                call_record = None
                error = None
                try:
                    call_record = await asyncio.to_thread(
                        complete_agent_step,
                        agent=agents[PLANNING_STEP],
                        prompt=prompt,
                        input_data=input_data,
                        config=config,
                        step_id=2,
                    )
                    response = call_record.response.strip()
                except Exception as exc:  # pragma: no cover - integration failure path
                    error = f"{type(exc).__name__}: {exc}"
                    response = ""
                planning = parse_planning_analysis(response, error=error)
                finished_at = utc_now()
                state = state.model_copy(update={"planning": planning, "current_stage": PLANNING_STEP})
                step = make_recovery_step(
                    step_id=2,
                    component=PLANNING_STEP,
                    actor="microsoft_agent_framework.planning_executor",
                    started_at=started_at,
                    finished_at=finished_at,
                    framework_primitive=FRAMEWORK_PRIMITIVE,
                    input_payload={"prompt": prompt},
                    output={"planning": planning.model_dump()},
                    llm_call_ids=[call_record.metrics.call_id] if call_record else [],
                    depends_on=[STATE_INITIALIZER],
                    error=error,
                )
                await ctx.send_message(
                    {
                        "workflow_state": state.model_dump(mode="json"),
                        "steps": [*payload["steps"], step.model_dump(mode="json")],
                        "llm_calls": [
                            *payload["llm_calls"],
                            *([call_record.metrics.model_dump(mode="json")] if call_record else []),
                        ],
                    }
                )

        class CheckpointWriterExecutor(Executor):
            @handler
            async def checkpoint(self, payload: dict, ctx: WorkflowContext[dict]) -> None:
                state = RecoveryWorkflowState.model_validate(payload["workflow_state"])
                started_at = utc_now()
                state = seal_state_for_checkpoint(
                    state,
                    checkpoint_id=logical_id,
                    created_at=started_at,
                )
                finished_at = utc_now()
                step = make_recovery_step(
                    step_id=3,
                    component=CHECKPOINT_WRITER,
                    actor="microsoft_agent_framework.checkpoint_writer_executor",
                    started_at=started_at,
                    finished_at=finished_at,
                    framework_primitive=FRAMEWORK_PRIMITIVE,
                    output={
                        "checkpoint_id": logical_id,
                        "checkpoint_stage": PLANNING_STEP,
                        "state_digest": state.state_digest,
                        "native_checkpoint_created_after_superstep": True,
                    },
                    depends_on=[PLANNING_STEP],
                    checkpoint_backend=CHECKPOINT_BACKEND,
                    native_checkpointing=True,
                )
                await ctx.send_message(
                    {
                        "workflow_state": state.model_dump(mode="json"),
                        "steps": [*payload["steps"], step.model_dump(mode="json")],
                        "llm_calls": payload["llm_calls"],
                    }
                )

        class FailureInjectorExecutor(Executor):
            @handler
            async def inject(self, payload: dict, ctx: WorkflowContext[dict]) -> None:
                nonlocal failure_error, injected_started_at, injected_finished_at
                if settings.inject_failure and not recovery_mode:
                    injected_started_at = utc_now()
                    failure_error = controlled_failure_message(logical_id)
                    injected_finished_at = utc_now()
                    raise ControlledFailure(failure_error)
                started_at = utc_now()
                finished_at = utc_now()
                step = make_recovery_step(
                    step_id=4,
                    component=FAILURE_INJECTOR,
                    actor="microsoft_agent_framework.failure_injector_executor",
                    started_at=injected_started_at or started_at,
                    finished_at=injected_finished_at or finished_at,
                    framework_primitive=FRAMEWORK_PRIMITIVE,
                    input_payload={"checkpoint_id": logical_id, "enabled": settings.inject_failure},
                    output={
                        "failure_injected": settings.inject_failure,
                        "captured": failure_error is not None,
                        "replayed_after_checkpoint_restore": recovery_mode,
                    },
                    depends_on=[CHECKPOINT_WRITER],
                    error=failure_error,
                )
                await ctx.send_message(
                    {
                        **payload,
                        "steps": [*payload["steps"], step.model_dump(mode="json")],
                    }
                )

        class RecoveryLoaderExecutor(Executor):
            @handler
            async def recover(self, payload: dict, ctx: WorkflowContext[dict]) -> None:
                state = RecoveryWorkflowState.model_validate(payload["workflow_state"])
                started_at = utc_now()
                verify_recovered_state(state)
                state = state.model_copy(
                    update={
                        "current_stage": RECOVERY_LOADER,
                        "recovered": settings.inject_failure,
                        "recovery_reason": failure_error,
                    }
                )
                finished_at = utc_now()
                step = make_recovery_step(
                    step_id=5,
                    component=RECOVERY_LOADER,
                    actor="microsoft_agent_framework.native_recovery_loader_executor",
                    started_at=started_at,
                    finished_at=finished_at,
                    framework_primitive=FRAMEWORK_PRIMITIVE,
                    input_payload={"native_checkpoint_id": selected_native_checkpoint_id},
                    output={
                        "recovery_attempted": settings.inject_failure,
                        "recovery_successful": settings.inject_failure,
                        "checkpoint_id": logical_id,
                        "native_checkpoint_id": selected_native_checkpoint_id,
                        "state_digest_verified": True,
                    },
                    depends_on=[FAILURE_INJECTOR],
                    checkpoint_backend=CHECKPOINT_BACKEND,
                    native_checkpointing=True,
                )
                await ctx.send_message(
                    {
                        **payload,
                        "workflow_state": state.model_dump(mode="json"),
                        "steps": [*payload["steps"], step.model_dump(mode="json")],
                    }
                )

        class ContinuationExecutor(Executor):
            @handler
            async def continue_work(self, payload: dict, ctx: WorkflowContext[dict]) -> None:
                state = RecoveryWorkflowState.model_validate(payload["workflow_state"])
                if state.planning is None:
                    raise RuntimeError("Continuation requires recovered planning state.")
                prompt = render_recovery_prompt(
                    input_data,
                    CONTINUATION_STEP,
                    planning=state.planning,
                )
                started_at = utc_now()
                call_record = None
                error = None
                try:
                    call_record = await asyncio.to_thread(
                        complete_agent_step,
                        agent=agents[CONTINUATION_STEP],
                        prompt=prompt,
                        input_data=input_data,
                        config=config,
                        step_id=6,
                    )
                    response = call_record.response.strip()
                except Exception as exc:  # pragma: no cover - integration failure path
                    error = f"{type(exc).__name__}: {exc}"
                    response = ""
                continuation = parse_continuation_result(response, error=error)
                finished_at = utc_now()
                state = state.model_copy(
                    update={"continuation": continuation, "current_stage": CONTINUATION_STEP}
                )
                step = make_recovery_step(
                    step_id=6,
                    component=CONTINUATION_STEP,
                    actor="microsoft_agent_framework.continuation_executor",
                    started_at=started_at,
                    finished_at=finished_at,
                    framework_primitive=FRAMEWORK_PRIMITIVE,
                    input_payload={"prompt": prompt, "recovered": state.recovered},
                    output={"continuation": continuation.model_dump()},
                    llm_call_ids=[call_record.metrics.call_id] if call_record else [],
                    depends_on=[RECOVERY_LOADER],
                    error=error,
                )
                await ctx.send_message(
                    {
                        "workflow_state": state.model_dump(mode="json"),
                        "steps": [*payload["steps"], step.model_dump(mode="json")],
                        "llm_calls": [
                            *payload["llm_calls"],
                            *([call_record.metrics.model_dump(mode="json")] if call_record else []),
                        ],
                    }
                )

        class FinalizerExecutor(Executor):
            @handler
            async def finalize(self, payload: dict, ctx: WorkflowContext[Never, dict]) -> None:
                state = RecoveryWorkflowState.model_validate(payload["workflow_state"])
                if state.continuation is None:
                    raise RuntimeError("Finalizer requires continuation output.")
                started_at = utc_now()
                state = state.model_copy(
                    update={
                        "current_stage": FINALIZER,
                        "result_generated_after_recovery": settings.inject_failure,
                    }
                )
                finished_at = utc_now()
                step = make_recovery_step(
                    step_id=7,
                    component=FINALIZER,
                    actor="microsoft_agent_framework.finalizer_executor",
                    started_at=started_at,
                    finished_at=finished_at,
                    framework_primitive=FRAMEWORK_PRIMITIVE,
                    output={
                        "answer": state.continuation.answer,
                        "result_generated_after_recovery": settings.inject_failure,
                    },
                    depends_on=[CONTINUATION_STEP],
                )
                await ctx.yield_output(
                    {
                        "workflow_state": state.model_dump(mode="json"),
                        "steps": [*payload["steps"], step.model_dump(mode="json")],
                        "llm_calls": payload["llm_calls"],
                    }
                )

        initializer = StateInitializerExecutor(id=STATE_INITIALIZER)
        planner = PlanningExecutor(id=PLANNING_STEP)
        writer = CheckpointWriterExecutor(id=CHECKPOINT_WRITER)
        failure = FailureInjectorExecutor(id=FAILURE_INJECTOR)
        recovery = RecoveryLoaderExecutor(id=RECOVERY_LOADER)
        continuation = ContinuationExecutor(id=CONTINUATION_STEP)
        finalizer = FinalizerExecutor(id=FINALIZER)

        storage_path = (
            checkpoint_directory(context.repo_root, config.framework)
            / logical_id
            / "native"
        )
        storage = FileCheckpointStorage(storage_path)
        workflow = (
            WorkflowBuilder(
                start_executor=initializer,
                output_from=[finalizer],
                max_iterations=12,
                name=ARCHITECTURE,
                checkpoint_storage=storage,
            )
            .add_edge(initializer, planner)
            .add_edge(planner, writer)
            .add_edge(writer, failure)
            .add_edge(failure, recovery)
            .add_edge(recovery, continuation)
            .add_edge(continuation, finalizer)
            .build()
        )

        async def run_workflow():
            nonlocal recovery_mode, selected_native_checkpoint_id
            failure_injected = False
            try:
                result = await asyncio.wait_for(
                    workflow.run(input_data.model_dump(mode="json")),
                    timeout=float(config.timeout_seconds),
                )
            except Exception as exc:
                if not settings.inject_failure or "ControlledFailure" not in str(exc):
                    raise
                failure_injected = True
                checkpoint = await storage.get_latest(workflow_name=ARCHITECTURE)
                if checkpoint is None:
                    raise RuntimeError("Microsoft workflow did not persist a pre-failure checkpoint.")
                selected_native_checkpoint_id = checkpoint.checkpoint_id
                recovery_mode = True
                result = await asyncio.wait_for(
                    workflow.run(
                        checkpoint_id=selected_native_checkpoint_id,
                        checkpoint_storage=storage,
                    ),
                    timeout=float(config.timeout_seconds),
                )
            checkpoints = await storage.list_checkpoints(workflow_name=ARCHITECTURE)
            return result, failure_injected, len(checkpoints)

        result, failure_injected, native_checkpoint_count = run_async(run_workflow())
        if settings.inject_failure and not failure_injected:
            raise RuntimeError("Microsoft failure injector did not interrupt the first execution.")
        outputs = result.get_outputs()
        if len(outputs) != 1 or not isinstance(outputs[0], dict):
            raise RuntimeError("Microsoft recovery workflow did not produce one dict output.")
        payload = outputs[0]
        state = RecoveryWorkflowState.model_validate(payload["workflow_state"])
        steps = [AgentStep.model_validate(item) for item in payload["steps"]]
        llm_calls = [LLMCallMetrics.model_validate(item) for item in payload["llm_calls"]]
        final_answer, structured_output = build_recovery_structured_output(
            input_data=input_data,
            config=config,
            state=state,
            steps=steps,
            llm_calls=llm_calls,
            framework_execution="microsoft_native_file_checkpoint_restore",
            framework_primitive=FRAMEWORK_PRIMITIVE,
            checkpoint_backend=CHECKPOINT_BACKEND,
            native_checkpointing=True,
            recovery_source=str(Path(storage_path).resolve()),
            failure_injected=failure_injected,
            recovery_attempted=failure_injected,
            recovery_successful=state.recovered,
            native_checkpoint_id=selected_native_checkpoint_id,
            native_checkpoints_created=native_checkpoint_count,
        )
        return MicrosoftAgentFrameworkRunOutput(
            final_answer=final_answer,
            structured_output=structured_output,
            steps=steps,
            llm_calls=llm_calls,
        )

    return run_with_resource_monitor(execute)
