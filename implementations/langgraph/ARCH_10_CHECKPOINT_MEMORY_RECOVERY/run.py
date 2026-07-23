"""LangGraph implementation for ARCH_10_CHECKPOINT_MEMORY_RECOVERY."""

from __future__ import annotations

import sqlite3
import uuid
from typing import TypedDict

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
from benchmark_core.resource_monitor import ResourceMonitor
from benchmark_core.schemas import ExperimentConfig, ExperimentInput
from benchmark_core.tracing import utc_now
from implementations.langgraph.utils_langgraph import (
    LangGraphRunContext,
    LangGraphRunOutput,
    complete_llm_step,
    langgraph_architecture_runner,
)


FRAMEWORK_PRIMITIVE = "StateGraph.SqliteSaver.durable_failure_resume"
CHECKPOINT_BACKEND = "langgraph.checkpoint.sqlite.SqliteSaver"


class CheckpointRecoveryState(TypedDict, total=False):
    workflow_state: object
    steps: list[object]
    llm_calls: list[object]


@langgraph_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: LangGraphRunContext,
) -> LangGraphRunOutput:
    """Run a StateGraph, close its database, and resume from durable state."""

    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.graph import END, START, StateGraph

    settings = get_recovery_settings(config)
    checkpoint_id = logical_checkpoint_id(config)
    checkpoint_file = checkpoint_directory(
        context.repo_root, config.framework
    ) / f"{checkpoint_id}.sqlite"
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    recovery_mode = False
    failure_error: str | None = None
    injected_started_at = None
    injected_finished_at = None

    def state_initializer_node(_state: CheckpointRecoveryState) -> dict:
        started_at = utc_now()
        workflow_state = initialize_recovery_state(input_data)
        finished_at = utc_now()
        step = make_recovery_step(
            step_id=1,
            component=STATE_INITIALIZER,
            actor="langgraph.state_initializer_node",
            started_at=started_at,
            finished_at=finished_at,
            framework_primitive=FRAMEWORK_PRIMITIVE,
            output={"state": workflow_state.model_dump(mode="json")},
        )
        return {"workflow_state": workflow_state, "steps": [step], "llm_calls": []}

    def planning_node(state: CheckpointRecoveryState) -> dict:
        workflow_state = RecoveryWorkflowState.model_validate(state["workflow_state"])
        prompt = render_recovery_prompt(input_data, PLANNING_STEP)
        started_at = utc_now()
        call_record = None
        error = None
        try:
            call_record = complete_llm_step(
                llm=context.llm,
                input_data=input_data,
                config=config,
                prompt=prompt,
                step_id=2,
            )
            response = call_record.response.strip()
        except Exception as exc:  # pragma: no cover - integration failure path
            error = f"{type(exc).__name__}: {exc}"
            response = ""
        planning = parse_planning_analysis(response, error=error)
        finished_at = utc_now()
        workflow_state = workflow_state.model_copy(
            update={"planning": planning, "current_stage": PLANNING_STEP}
        )
        step = make_recovery_step(
            step_id=2,
            component=PLANNING_STEP,
            actor="langgraph.planning_or_analysis_node",
            started_at=started_at,
            finished_at=finished_at,
            framework_primitive=FRAMEWORK_PRIMITIVE,
            input_payload={"prompt": prompt},
            output={"planning": planning.model_dump()},
            llm_call_ids=[call_record.metrics.call_id] if call_record else [],
            depends_on=[STATE_INITIALIZER],
            error=error,
        )
        return {
            "workflow_state": workflow_state,
            "steps": [*state["steps"], step],
            "llm_calls": [*state["llm_calls"], *([call_record.metrics] if call_record else [])],
        }

    def checkpoint_writer_node(state: CheckpointRecoveryState) -> dict:
        workflow_state = RecoveryWorkflowState.model_validate(state["workflow_state"])
        started_at = utc_now()
        workflow_state = seal_state_for_checkpoint(
            workflow_state,
            checkpoint_id=checkpoint_id,
            created_at=started_at,
        )
        finished_at = utc_now()
        step = make_recovery_step(
            step_id=3,
            component=CHECKPOINT_WRITER,
            actor="langgraph.checkpoint_writer_node",
            started_at=started_at,
            finished_at=finished_at,
            framework_primitive=FRAMEWORK_PRIMITIVE,
            input_payload={"state_stage": PLANNING_STEP},
            output={
                "checkpoint_id": checkpoint_id,
                "checkpoint_stage": PLANNING_STEP,
                "state_digest": workflow_state.state_digest,
                "native_checkpoint_created_after_node": True,
            },
            depends_on=[PLANNING_STEP],
            checkpoint_backend=CHECKPOINT_BACKEND,
            native_checkpointing=True,
        )
        return {"workflow_state": workflow_state, "steps": [*state["steps"], step]}

    def failure_injector_node(state: CheckpointRecoveryState) -> dict:
        nonlocal failure_error, injected_started_at, injected_finished_at
        if settings.inject_failure and not recovery_mode:
            injected_started_at = utc_now()
            failure_error = controlled_failure_message(checkpoint_id)
            injected_finished_at = utc_now()
            raise ControlledFailure(failure_error)
        started_at = utc_now()
        finished_at = utc_now()
        step = make_recovery_step(
            step_id=4,
            component=FAILURE_INJECTOR,
            actor="langgraph.failure_injector_node",
            started_at=injected_started_at or started_at,
            finished_at=injected_finished_at or finished_at,
            framework_primitive=FRAMEWORK_PRIMITIVE,
            input_payload={"checkpoint_id": checkpoint_id, "enabled": settings.inject_failure},
            output={
                "failure_injected": settings.inject_failure,
                "captured": failure_error is not None,
                "replayed_after_checkpoint_restore": recovery_mode,
            },
            depends_on=[CHECKPOINT_WRITER],
            error=failure_error,
        )
        return {"steps": [*state["steps"], step]}

    def recovery_loader_node(state: CheckpointRecoveryState) -> dict:
        workflow_state = RecoveryWorkflowState.model_validate(state["workflow_state"])
        started_at = utc_now()
        verify_recovered_state(workflow_state)
        workflow_state = workflow_state.model_copy(
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
            actor="langgraph.native_checkpoint_recovery_loader",
            started_at=started_at,
            finished_at=finished_at,
            framework_primitive=FRAMEWORK_PRIMITIVE,
            input_payload={"checkpoint_id": checkpoint_id},
            output={
                "recovery_attempted": settings.inject_failure,
                "recovery_successful": settings.inject_failure,
                "checkpoint_id": checkpoint_id,
                "state_digest_verified": True,
            },
            depends_on=[FAILURE_INJECTOR],
            checkpoint_backend=CHECKPOINT_BACKEND,
            native_checkpointing=True,
        )
        return {"workflow_state": workflow_state, "steps": [*state["steps"], step]}

    def continuation_node(state: CheckpointRecoveryState) -> dict:
        workflow_state = RecoveryWorkflowState.model_validate(state["workflow_state"])
        if workflow_state.planning is None:
            raise RuntimeError("Continuation requires recovered planning state.")
        prompt = render_recovery_prompt(
            input_data,
            CONTINUATION_STEP,
            planning=workflow_state.planning,
        )
        started_at = utc_now()
        call_record = None
        error = None
        try:
            call_record = complete_llm_step(
                llm=context.llm,
                input_data=input_data,
                config=config,
                prompt=prompt,
                step_id=6,
            )
            response = call_record.response.strip()
        except Exception as exc:  # pragma: no cover - integration failure path
            error = f"{type(exc).__name__}: {exc}"
            response = ""
        continuation = parse_continuation_result(response, error=error)
        finished_at = utc_now()
        workflow_state = workflow_state.model_copy(
            update={"continuation": continuation, "current_stage": CONTINUATION_STEP}
        )
        step = make_recovery_step(
            step_id=6,
            component=CONTINUATION_STEP,
            actor="langgraph.continuation_node",
            started_at=started_at,
            finished_at=finished_at,
            framework_primitive=FRAMEWORK_PRIMITIVE,
            input_payload={"prompt": prompt, "recovered": workflow_state.recovered},
            output={"continuation": continuation.model_dump()},
            llm_call_ids=[call_record.metrics.call_id] if call_record else [],
            depends_on=[RECOVERY_LOADER],
            error=error,
        )
        return {
            "workflow_state": workflow_state,
            "steps": [*state["steps"], step],
            "llm_calls": [*state["llm_calls"], *([call_record.metrics] if call_record else [])],
        }

    def finalizer_node(state: CheckpointRecoveryState) -> dict:
        workflow_state = RecoveryWorkflowState.model_validate(state["workflow_state"])
        if workflow_state.continuation is None:
            raise RuntimeError("Finalizer requires continuation output.")
        started_at = utc_now()
        workflow_state = workflow_state.model_copy(
            update={
                "current_stage": FINALIZER,
                "result_generated_after_recovery": settings.inject_failure,
            }
        )
        finished_at = utc_now()
        step = make_recovery_step(
            step_id=7,
            component=FINALIZER,
            actor="langgraph.finalizer_node",
            started_at=started_at,
            finished_at=finished_at,
            framework_primitive=FRAMEWORK_PRIMITIVE,
            output={
                "answer": workflow_state.continuation.answer,
                "result_generated_after_recovery": settings.inject_failure,
            },
            depends_on=[CONTINUATION_STEP],
        )
        return {"workflow_state": workflow_state, "steps": [*state["steps"], step]}

    graph = StateGraph(CheckpointRecoveryState)
    graph.add_node(STATE_INITIALIZER, state_initializer_node)
    graph.add_node(PLANNING_STEP, planning_node)
    graph.add_node(CHECKPOINT_WRITER, checkpoint_writer_node)
    graph.add_node(FAILURE_INJECTOR, failure_injector_node)
    graph.add_node(RECOVERY_LOADER, recovery_loader_node)
    graph.add_node(CONTINUATION_STEP, continuation_node)
    graph.add_node(FINALIZER, finalizer_node)
    graph.add_edge(START, STATE_INITIALIZER)
    graph.add_edge(STATE_INITIALIZER, PLANNING_STEP)
    graph.add_edge(PLANNING_STEP, CHECKPOINT_WRITER)
    graph.add_edge(CHECKPOINT_WRITER, FAILURE_INJECTOR)
    graph.add_edge(FAILURE_INJECTOR, RECOVERY_LOADER)
    graph.add_edge(RECOVERY_LOADER, CONTINUATION_STEP)
    graph.add_edge(CONTINUATION_STEP, FINALIZER)
    graph.add_edge(FINALIZER, END)

    serializer = JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("benchmark_core.checkpoint_memory_recovery", "RecoveryWorkflowState"),
            ("benchmark_core.schemas", "AgentStep"),
            ("benchmark_core.schemas", "LLMCallMetrics"),
        ]
    )
    thread_config = {
        "configurable": {
            "thread_id": (
                f"arch10-{config.framework}-{config.run_id}-{uuid.uuid4().hex}"
            )
        },
        "recursion_limit": 20,
    }
    selected_native_checkpoint_id = None
    failure_injected = False
    native_checkpoints_created = 0
    database_reopened_for_recovery = False

    def open_checkpointer() -> tuple[sqlite3.Connection, object]:
        connection = sqlite3.connect(checkpoint_file, check_same_thread=False)
        return connection, SqliteSaver(connection, serde=serializer)

    with ResourceMonitor() as monitor:
        initial_connection, initial_checkpointer = open_checkpointer()
        compiled = graph.compile(checkpointer=initial_checkpointer)
        try:
            final_state = compiled.invoke({}, thread_config)
        except ControlledFailure:
            failure_injected = True
            snapshot = compiled.get_state(thread_config)
            if snapshot is None or not snapshot.values:
                raise RuntimeError("LangGraph did not retain a checkpoint before failure.")
            selected_native_checkpoint_id = snapshot.config.get("configurable", {}).get(
                "checkpoint_id"
            )
            initial_connection.close()

            # A fresh connection and freshly compiled graph prove that recovery
            # reads the persisted SQLite state instead of retaining Python memory.
            recovery_mode = True
            database_reopened_for_recovery = True
            recovery_connection, recovery_checkpointer = open_checkpointer()
            try:
                recovered_graph = graph.compile(checkpointer=recovery_checkpointer)
                recovered_snapshot = recovered_graph.get_state(snapshot.config)
                if recovered_snapshot is None or not recovered_snapshot.values:
                    raise RuntimeError(
                        "LangGraph SQLite checkpoint was not durable across connection reopen."
                    )
                final_state = recovered_graph.invoke(None, recovered_snapshot.config)
                native_checkpoints_created = sum(
                    1 for _ in recovery_checkpointer.list(thread_config)
                )
            finally:
                recovery_connection.close()
        else:
            native_checkpoints_created = sum(
                1 for _ in initial_checkpointer.list(thread_config)
            )
            initial_connection.close()
        resource_usage = monitor.usage

    if settings.inject_failure and not failure_injected:
        raise RuntimeError("LangGraph failure injector did not interrupt the first execution.")
    workflow_state = RecoveryWorkflowState.model_validate(final_state["workflow_state"])
    steps = final_state["steps"]
    llm_calls = final_state["llm_calls"]
    final_answer, structured_output = build_recovery_structured_output(
        input_data=input_data,
        config=config,
        state=workflow_state,
        steps=steps,
        llm_calls=llm_calls,
        framework_execution="langgraph_native_sqlite_checkpoint_reopen_resume",
        framework_primitive=FRAMEWORK_PRIMITIVE,
        checkpoint_backend=CHECKPOINT_BACKEND,
        native_checkpointing=True,
        recovery_source=(
            f"sqlite={checkpoint_file};"
            f"thread_id={thread_config['configurable']['thread_id']}"
        ),
        failure_injected=failure_injected,
        recovery_attempted=failure_injected,
        recovery_successful=workflow_state.recovered,
        native_checkpoint_id=selected_native_checkpoint_id,
        native_checkpoints_created=native_checkpoints_created,
    )
    structured_output["recovery_execution"]["durable_storage"] = True
    structured_output["recovery_execution"]["database_reopened_for_recovery"] = (
        database_reopened_for_recovery
    )
    return LangGraphRunOutput(
        final_answer=final_answer,
        structured_output=structured_output,
        steps=steps,
        llm_calls=llm_calls,
        resource_usage=resource_usage,
    )
