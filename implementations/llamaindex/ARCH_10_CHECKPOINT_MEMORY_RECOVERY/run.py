"""LlamaIndex implementation for ARCH_10_CHECKPOINT_MEMORY_RECOVERY."""

import asyncio
from typing import Any

from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step

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
    build_recovery_structured_output,
    checkpoint_path,
    controlled_failure_message,
    get_recovery_settings,
    initialize_recovery_state,
    logical_checkpoint_id,
    make_recovery_step,
    parse_continuation_result,
    parse_planning_analysis,
    read_json_checkpoint,
    render_recovery_prompt,
    seal_state_for_checkpoint,
    verify_recovered_state,
    write_json_checkpoint,
)
from benchmark_core.schemas import AgentStep, ExperimentConfig, ExperimentInput, LLMCallMetrics
from benchmark_core.tracing import utc_now
from implementations.llamaindex.ARCH_10_CHECKPOINT_MEMORY_RECOVERY.events import (
    CheckpointReadyEvent,
    ContinuedEvent,
    InitializedEvent,
    PlannedEvent,
    RecoveredEvent,
)
from implementations.llamaindex.utils_llamaindex import (
    LlamaIndexRunContext,
    LlamaIndexRunOutput,
    build_function_agent,
    complete_agent_step,
    llamaindex_architecture_runner,
    run_async,
    run_with_resource_monitor,
)


FRAMEWORK_PRIMITIVE = "Workflow.Context.to_dict.from_dict.two_phase_resume"
CHECKPOINT_BACKEND = "llama_index.workflow.Context JSON serialization"
CONTEXT_STATE_KEY = "arch10_recovery_payload"


class WorkflowPayload:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload


@llamaindex_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: LlamaIndexRunContext,
) -> LlamaIndexRunOutput:
    """Serialize a Workflow Context, fail, restore it, and run phase two."""

    settings = get_recovery_settings(config)
    checkpoint_id = logical_checkpoint_id(config)
    checkpoint_file = checkpoint_path(context.repo_root, config.framework, checkpoint_id)
    checkpoint_captured = asyncio.Event()
    failure_error: str | None = None
    injected_started_at = None
    injected_finished_at = None
    agents = {
        component: build_function_agent(
            name=component,
            system_prompt=(
                f"You are the {component} component in ARCH_10. "
                "Execute one bounded phase without handoffs, routing, debate, or hidden loops."
            ),
            context=context,
            input_data=input_data,
            config=config,
        )
        for component in (PLANNING_STEP, CONTINUATION_STEP)
    }

    class PreCheckpointWorkflow(Workflow):
        @step
        async def state_initializer(self, _ev: StartEvent) -> InitializedEvent:
            started_at = utc_now()
            state = initialize_recovery_state(input_data)
            finished_at = utc_now()
            trace = make_recovery_step(
                step_id=1,
                component=STATE_INITIALIZER,
                actor="llamaindex.workflow.state_initializer",
                started_at=started_at,
                finished_at=finished_at,
                framework_primitive=FRAMEWORK_PRIMITIVE,
                output={"state": state.model_dump(mode="json")},
            )
            return InitializedEvent(
                payload={
                    "workflow_state": state.model_dump(mode="json"),
                    "steps": [trace.model_dump(mode="json")],
                    "llm_calls": [],
                }
            )

        @step
        async def planning_or_analysis_step(self, ev: InitializedEvent) -> PlannedEvent:
            state = RecoveryWorkflowState.model_validate(ev.payload["workflow_state"])
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
            trace = make_recovery_step(
                step_id=2,
                component=PLANNING_STEP,
                actor="llamaindex.workflow.planning_or_analysis_step",
                started_at=started_at,
                finished_at=finished_at,
                framework_primitive=FRAMEWORK_PRIMITIVE,
                input_payload={"prompt": prompt},
                output={"planning": planning.model_dump()},
                llm_call_ids=[call_record.metrics.call_id] if call_record else [],
                depends_on=[STATE_INITIALIZER],
                error=error,
            )
            return PlannedEvent(
                payload={
                    "workflow_state": state.model_dump(mode="json"),
                    "steps": [*ev.payload["steps"], trace.model_dump(mode="json")],
                    "llm_calls": [
                        *ev.payload["llm_calls"],
                        *([call_record.metrics.model_dump(mode="json")] if call_record else []),
                    ],
                }
            )

        @step
        async def checkpoint_writer(
            self,
            ev: PlannedEvent,
            ctx: Context,
        ) -> CheckpointReadyEvent:
            state = RecoveryWorkflowState.model_validate(ev.payload["workflow_state"])
            started_at = utc_now()
            state = seal_state_for_checkpoint(
                state,
                checkpoint_id=checkpoint_id,
                created_at=started_at,
            )
            finished_at = utc_now()
            trace = make_recovery_step(
                step_id=3,
                component=CHECKPOINT_WRITER,
                actor="llamaindex.workflow.context_checkpoint_writer",
                started_at=started_at,
                finished_at=finished_at,
                framework_primitive=FRAMEWORK_PRIMITIVE,
                output={
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_stage": PLANNING_STEP,
                    "checkpoint_path": str(checkpoint_file),
                    "state_digest": state.state_digest,
                    "context_serialized_after_event": True,
                },
                depends_on=[PLANNING_STEP],
                checkpoint_backend=CHECKPOINT_BACKEND,
                native_checkpointing=True,
            )
            payload = {
                "workflow_state": state.model_dump(mode="json"),
                "steps": [*ev.payload["steps"], trace.model_dump(mode="json")],
                "llm_calls": ev.payload["llm_calls"],
            }
            await ctx.store.set(CONTEXT_STATE_KEY, payload)
            checkpoint_event = CheckpointReadyEvent(payload=payload)
            ctx.write_event_to_stream(checkpoint_event)
            return checkpoint_event

        @step
        async def failure_injector(self, _ev: CheckpointReadyEvent) -> StopEvent:
            nonlocal failure_error, injected_started_at, injected_finished_at
            await checkpoint_captured.wait()
            if settings.inject_failure:
                injected_started_at = utc_now()
                failure_error = controlled_failure_message(checkpoint_id)
                injected_finished_at = utc_now()
                raise ControlledFailure(failure_error)
            return StopEvent(result="failure injection disabled")

    pre_workflow = PreCheckpointWorkflow(
        timeout=float(config.timeout_seconds),
        verbose=False,
        num_concurrent_runs=1,
    )

    async def run_pre_checkpoint():
        handler = pre_workflow.run(common_input=input_data.model_dump(mode="json"))
        captured = False
        async for event in handler.stream_events(expose_internal=True):
            if isinstance(event, CheckpointReadyEvent):
                context_snapshot = handler.ctx.to_dict()
                write_json_checkpoint(checkpoint_file, context_snapshot)
                checkpoint_captured.set()
                captured = True
                break
        if not captured:
            raise RuntimeError("LlamaIndex did not emit the checkpoint boundary event.")
        failure_injected = False
        try:
            await handler
        except ControlledFailure:
            failure_injected = True
        if settings.inject_failure and not failure_injected:
            raise RuntimeError("LlamaIndex failure injector did not interrupt phase one.")
        return failure_injected

    failure_injected = run_async(run_pre_checkpoint())
    failure_started = injected_started_at or utc_now()
    failure_finished = injected_finished_at or failure_started

    async def restore_context_payload() -> dict[str, Any]:
        snapshot = read_json_checkpoint(checkpoint_file)
        restored_context = Context.from_dict(pre_workflow, snapshot)
        value = await restored_context.store.get(CONTEXT_STATE_KEY)
        if not isinstance(value, dict):
            raise RuntimeError("Restored LlamaIndex Context did not contain ARCH_10 state.")
        return value

    restored_payload = run_async(restore_context_payload())
    restored_payload["steps"].append(
        make_recovery_step(
            step_id=4,
            component=FAILURE_INJECTOR,
            actor="llamaindex.workflow.failure_injector",
            started_at=failure_started,
            finished_at=failure_finished,
            framework_primitive=FRAMEWORK_PRIMITIVE,
            input_payload={"checkpoint_id": checkpoint_id, "enabled": settings.inject_failure},
            output={"failure_injected": failure_injected, "captured": failure_error is not None},
            depends_on=[CHECKPOINT_WRITER],
            error=failure_error,
        ).model_dump(mode="json")
    )

    class PostRecoveryWorkflow(Workflow):
        @step
        async def recovery_loader(self, ev: StartEvent) -> RecoveredEvent:
            payload = dict(ev.recovered_payload)
            state = RecoveryWorkflowState.model_validate(payload["workflow_state"])
            started_at = utc_now()
            verify_recovered_state(state)
            state = state.model_copy(
                update={
                    "current_stage": RECOVERY_LOADER,
                    "recovered": failure_injected,
                    "recovery_reason": failure_error,
                }
            )
            finished_at = utc_now()
            trace = make_recovery_step(
                step_id=5,
                component=RECOVERY_LOADER,
                actor="llamaindex.workflow.context_recovery_loader",
                started_at=started_at,
                finished_at=finished_at,
                framework_primitive=FRAMEWORK_PRIMITIVE,
                input_payload={"checkpoint_path": str(checkpoint_file)},
                output={
                    "recovery_attempted": failure_injected,
                    "recovery_successful": failure_injected,
                    "checkpoint_id": checkpoint_id,
                    "state_digest_verified": True,
                },
                depends_on=[FAILURE_INJECTOR],
                checkpoint_backend=CHECKPOINT_BACKEND,
                native_checkpointing=True,
            )
            payload["workflow_state"] = state.model_dump(mode="json")
            payload["steps"] = [*payload["steps"], trace.model_dump(mode="json")]
            return RecoveredEvent(payload=payload)

        @step
        async def continuation_step(self, ev: RecoveredEvent) -> ContinuedEvent:
            state = RecoveryWorkflowState.model_validate(ev.payload["workflow_state"])
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
            trace = make_recovery_step(
                step_id=6,
                component=CONTINUATION_STEP,
                actor="llamaindex.workflow.continuation_step",
                started_at=started_at,
                finished_at=finished_at,
                framework_primitive=FRAMEWORK_PRIMITIVE,
                input_payload={"prompt": prompt, "recovered": state.recovered},
                output={"continuation": continuation.model_dump()},
                llm_call_ids=[call_record.metrics.call_id] if call_record else [],
                depends_on=[RECOVERY_LOADER],
                error=error,
            )
            payload = dict(ev.payload)
            payload["workflow_state"] = state.model_dump(mode="json")
            payload["steps"] = [*payload["steps"], trace.model_dump(mode="json")]
            payload["llm_calls"] = [
                *payload["llm_calls"],
                *([call_record.metrics.model_dump(mode="json")] if call_record else []),
            ]
            return ContinuedEvent(payload=payload)

        @step
        async def finalizer(self, ev: ContinuedEvent) -> StopEvent:
            state = RecoveryWorkflowState.model_validate(ev.payload["workflow_state"])
            if state.continuation is None:
                raise RuntimeError("Finalizer requires continuation output.")
            started_at = utc_now()
            state = state.model_copy(
                update={
                    "current_stage": FINALIZER,
                    "result_generated_after_recovery": failure_injected,
                }
            )
            finished_at = utc_now()
            trace = make_recovery_step(
                step_id=7,
                component=FINALIZER,
                actor="llamaindex.workflow.finalizer",
                started_at=started_at,
                finished_at=finished_at,
                framework_primitive=FRAMEWORK_PRIMITIVE,
                output={
                    "answer": state.continuation.answer,
                    "result_generated_after_recovery": failure_injected,
                },
                depends_on=[CONTINUATION_STEP],
            )
            payload = dict(ev.payload)
            payload["workflow_state"] = state.model_dump(mode="json")
            payload["steps"] = [*payload["steps"], trace.model_dump(mode="json")]
            return StopEvent(result=WorkflowPayload(payload))

    post_workflow = PostRecoveryWorkflow(
        timeout=float(config.timeout_seconds),
        verbose=False,
        num_concurrent_runs=1,
    )

    def execute_workflow() -> LlamaIndexRunOutput:
        async def run_post_recovery():
            handler = post_workflow.run(recovered_payload=restored_payload)
            return await handler

        workflow_payload = run_async(run_post_recovery())
        if not isinstance(workflow_payload, WorkflowPayload):
            raise RuntimeError("LlamaIndex recovery workflow did not return WorkflowPayload.")
        payload = workflow_payload.payload
        state = RecoveryWorkflowState.model_validate(payload["workflow_state"])
        steps = [AgentStep.model_validate(item) for item in payload["steps"]]
        llm_calls = [LLMCallMetrics.model_validate(item) for item in payload["llm_calls"]]
        final_answer, structured_output = build_recovery_structured_output(
            input_data=input_data,
            config=config,
            state=state,
            steps=steps,
            llm_calls=llm_calls,
            framework_execution="llamaindex_native_context_serialization_two_phase_recovery",
            framework_primitive=FRAMEWORK_PRIMITIVE,
            checkpoint_backend=CHECKPOINT_BACKEND,
            native_checkpointing=True,
            recovery_source=str(checkpoint_file),
            failure_injected=failure_injected,
            recovery_attempted=failure_injected,
            recovery_successful=state.recovered,
            native_checkpoint_id=checkpoint_id,
            native_checkpoints_created=1,
        )
        return LlamaIndexRunOutput(
            final_answer=final_answer,
            structured_output=structured_output,
            steps=steps,
            llm_calls=llm_calls,
        )

    return run_with_resource_monitor(execute_workflow)
