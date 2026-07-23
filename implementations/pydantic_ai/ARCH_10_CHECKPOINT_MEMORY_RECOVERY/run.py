"""Pydantic AI + pydantic-graph implementation for ARCH_10."""

import asyncio
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_graph import GraphBuilder, StepContext

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
from benchmark_core.schemas import AgentStep, ExperimentConfig, ExperimentInput, LLMCallMetrics
from benchmark_core.tracing import utc_now
from implementations.pydantic_ai.utils_pydantic_ai import (
    PydanticAIRunContext,
    PydanticAIRunOutput,
    build_typed_agent,
    complete_agent_step,
    pydantic_ai_architecture_runner,
)


FRAMEWORK_PRIMITIVE = "GraphBuilder.two_phase_typed_graph.portable_checkpoint"
CHECKPOINT_BACKEND = "pydantic.PortableCheckpoint JSON"


class RecoverySignal(BaseModel):
    stage: str


class CheckpointSignal(BaseModel):
    checkpoint_path: str
    checkpoint_id: str


class RecoveryGraphState(BaseModel):
    workflow_state: RecoveryWorkflowState | None = None
    steps: list[AgentStep] = Field(default_factory=list)
    llm_calls: list[LLMCallMetrics] = Field(default_factory=list)


class RecoveryGraphDeps:
    def __init__(
        self,
        *,
        input_data: ExperimentInput,
        config: ExperimentConfig,
        context: PydanticAIRunContext,
        agents: dict[str, Any],
        checkpoint_file: Path,
    ) -> None:
        self.input_data = input_data
        self.config = config
        self.context = context
        self.agents = agents
        self.checkpoint_file = checkpoint_file


class RecoveryGraphOutput(BaseModel):
    state: RecoveryWorkflowState
    steps: list[AgentStep]
    llm_calls: list[LLMCallMetrics]


@pydantic_ai_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: PydanticAIRunContext,
) -> PydanticAIRunOutput:
    """Execute two typed graph phases separated by a validated JSON checkpoint."""

    settings = get_recovery_settings(config)
    checkpoint_id = logical_checkpoint_id(config)
    checkpoint_file = checkpoint_path(context.repo_root, config.framework, checkpoint_id)
    agents = {
        component: build_typed_agent(
            name=component,
            instructions=(
                f"You are the typed {component} for ARCH_10. "
                "Use one controlled phase without routing, delegation, memory, or hidden retries."
            ),
            context=context,
            input_data=input_data,
            config=config,
        )
        for component in (PLANNING_STEP, CONTINUATION_STEP)
    }
    deps = RecoveryGraphDeps(
        input_data=input_data,
        config=config,
        context=context,
        agents=agents,
        checkpoint_file=checkpoint_file,
    )
    pre_state = RecoveryGraphState()
    injected_started_at = None
    injected_finished_at = None
    pre_builder = GraphBuilder(
        name="ARCH_10_PRE_CHECKPOINT",
        state_type=RecoveryGraphState,
        deps_type=RecoveryGraphDeps,
        input_type=ExperimentInput,
        output_type=RecoverySignal,
        auto_instrument=False,
    )

    async def state_initializer(
        ctx: StepContext[RecoveryGraphState, RecoveryGraphDeps, ExperimentInput],
    ) -> RecoverySignal:
        started_at = utc_now()
        state = initialize_recovery_state(ctx.inputs)
        finished_at = utc_now()
        ctx.state.workflow_state = state
        ctx.state.steps.append(
            make_recovery_step(
                step_id=1,
                component=STATE_INITIALIZER,
                actor="pydantic_ai.graph.state_initializer",
                started_at=started_at,
                finished_at=finished_at,
                framework_primitive=FRAMEWORK_PRIMITIVE,
                output={"state": state.model_dump(mode="json")},
            )
        )
        return RecoverySignal(stage=STATE_INITIALIZER)

    async def planning_or_analysis_step(
        ctx: StepContext[RecoveryGraphState, RecoveryGraphDeps, RecoverySignal],
    ) -> RecoverySignal:
        if ctx.state.workflow_state is None:
            raise RuntimeError("Planning requires initialized workflow state.")
        prompt = render_recovery_prompt(ctx.deps.input_data, PLANNING_STEP)
        started_at = utc_now()
        call_record = None
        error = None
        try:
            call_record = await asyncio.to_thread(
                complete_agent_step,
                agent=ctx.deps.agents[PLANNING_STEP],
                prompt=prompt,
                input_data=ctx.deps.input_data,
                config=ctx.deps.config,
                step_id=2,
            )
            response = call_record.response.strip()
        except Exception as exc:  # pragma: no cover - integration failure path
            error = f"{type(exc).__name__}: {exc}"
            response = ""
        planning = parse_planning_analysis(response, error=error)
        finished_at = utc_now()
        ctx.state.workflow_state = ctx.state.workflow_state.model_copy(
            update={"planning": planning, "current_stage": PLANNING_STEP}
        )
        ctx.state.steps.append(
            make_recovery_step(
                step_id=2,
                component=PLANNING_STEP,
                actor="pydantic_ai.graph.planning_or_analysis_step",
                started_at=started_at,
                finished_at=finished_at,
                framework_primitive=FRAMEWORK_PRIMITIVE,
                input_payload={"prompt": prompt},
                output={"planning": planning.model_dump()},
                llm_call_ids=[call_record.metrics.call_id] if call_record else [],
                depends_on=[STATE_INITIALIZER],
                error=error,
            )
        )
        if call_record:
            ctx.state.llm_calls.append(call_record.metrics)
        return RecoverySignal(stage=PLANNING_STEP)

    async def checkpoint_writer(
        ctx: StepContext[RecoveryGraphState, RecoveryGraphDeps, RecoverySignal],
    ) -> CheckpointSignal:
        if ctx.state.workflow_state is None:
            raise RuntimeError("Checkpoint writer requires workflow state.")
        started_at = utc_now()
        sealed = seal_state_for_checkpoint(
            ctx.state.workflow_state,
            checkpoint_id=checkpoint_id,
            created_at=started_at,
        )
        checkpoint = build_portable_checkpoint(
            framework=ctx.deps.config.framework,
            config=ctx.deps.config,
            state=sealed,
            created_at=started_at,
        )
        write_portable_checkpoint(ctx.deps.checkpoint_file, checkpoint)
        finished_at = utc_now()
        ctx.state.workflow_state = sealed
        ctx.state.steps.append(
            make_recovery_step(
                step_id=3,
                component=CHECKPOINT_WRITER,
                actor="pydantic_ai.graph.typed_checkpoint_writer",
                started_at=started_at,
                finished_at=finished_at,
                framework_primitive=FRAMEWORK_PRIMITIVE,
                output={
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_stage": PLANNING_STEP,
                    "checkpoint_path": str(ctx.deps.checkpoint_file),
                    "state_digest": sealed.state_digest,
                },
                depends_on=[PLANNING_STEP],
                checkpoint_backend=CHECKPOINT_BACKEND,
                native_checkpointing=False,
            )
        )
        return CheckpointSignal(
            checkpoint_path=str(ctx.deps.checkpoint_file),
            checkpoint_id=checkpoint_id,
        )

    async def failure_injector(
        ctx: StepContext[RecoveryGraphState, RecoveryGraphDeps, CheckpointSignal],
    ) -> RecoverySignal:
        nonlocal injected_started_at, injected_finished_at
        if settings.inject_failure:
            injected_started_at = utc_now()
            injected_finished_at = utc_now()
            raise ControlledFailure(controlled_failure_message(ctx.inputs.checkpoint_id))
        return RecoverySignal(stage=FAILURE_INJECTOR)

    initializer_node = pre_builder.step(state_initializer, node_id=STATE_INITIALIZER)
    planning_node = pre_builder.step(planning_or_analysis_step, node_id=PLANNING_STEP)
    writer_node = pre_builder.step(checkpoint_writer, node_id=CHECKPOINT_WRITER)
    failure_node = pre_builder.step(failure_injector, node_id=FAILURE_INJECTOR)
    pre_builder.add(pre_builder.edge_from(pre_builder.start_node).to(initializer_node))
    pre_builder.add(pre_builder.edge_from(initializer_node).to(planning_node))
    pre_builder.add(pre_builder.edge_from(planning_node).to(writer_node))
    pre_builder.add(pre_builder.edge_from(writer_node).to(failure_node))
    pre_builder.add(pre_builder.edge_from(failure_node).to(pre_builder.end_node))
    pre_graph = pre_builder.build()

    failure_injected = False
    failure_error = None
    with ResourceMonitor() as monitor:
        try:
            asyncio.run(
                asyncio.wait_for(
                    pre_graph.run(inputs=input_data, state=pre_state, deps=deps),
                    timeout=float(config.timeout_seconds),
                )
            )
        except ControlledFailure as exc:
            failure_injected = True
            failure_error = str(exc)
        failure_started = injected_started_at or utc_now()
        failure_finished = injected_finished_at or failure_started
        if settings.inject_failure and not failure_injected:
            raise RuntimeError("Pydantic graph did not propagate the controlled failure.")
        pre_state.steps.append(
            make_recovery_step(
                step_id=4,
                component=FAILURE_INJECTOR,
                actor="pydantic_ai.graph.failure_injector",
                started_at=failure_started,
                finished_at=failure_finished,
                framework_primitive=FRAMEWORK_PRIMITIVE,
                input_payload={"checkpoint_id": checkpoint_id, "enabled": settings.inject_failure},
                output={"failure_injected": failure_injected, "captured": failure_error is not None},
                depends_on=[CHECKPOINT_WRITER],
                error=failure_error,
            )
        )

        post_state = RecoveryGraphState(
            steps=list(pre_state.steps),
            llm_calls=list(pre_state.llm_calls),
        )
        post_builder = GraphBuilder(
            name="ARCH_10_POST_RECOVERY",
            state_type=RecoveryGraphState,
            deps_type=RecoveryGraphDeps,
            input_type=CheckpointSignal,
            output_type=RecoveryGraphOutput,
            auto_instrument=False,
        )

        async def recovery_loader(
            ctx: StepContext[RecoveryGraphState, RecoveryGraphDeps, CheckpointSignal],
        ) -> RecoverySignal:
            started_at = utc_now()
            checkpoint = load_portable_checkpoint(Path(ctx.inputs.checkpoint_path))
            state = checkpoint.state
            verify_recovered_state(state)
            state = state.model_copy(
                update={
                    "current_stage": RECOVERY_LOADER,
                    "recovered": failure_injected,
                    "recovery_reason": failure_error,
                }
            )
            finished_at = utc_now()
            ctx.state.workflow_state = state
            ctx.state.steps.append(
                make_recovery_step(
                    step_id=5,
                    component=RECOVERY_LOADER,
                    actor="pydantic_ai.graph.typed_checkpoint_loader",
                    started_at=started_at,
                    finished_at=finished_at,
                    framework_primitive=FRAMEWORK_PRIMITIVE,
                    input_payload={"checkpoint_path": ctx.inputs.checkpoint_path},
                    output={
                        "recovery_attempted": failure_injected,
                        "recovery_successful": failure_injected,
                        "checkpoint_id": ctx.inputs.checkpoint_id,
                        "state_digest_verified": True,
                    },
                    depends_on=[FAILURE_INJECTOR],
                    checkpoint_backend=CHECKPOINT_BACKEND,
                    native_checkpointing=False,
                )
            )
            return RecoverySignal(stage=RECOVERY_LOADER)

        async def continuation_step(
            ctx: StepContext[RecoveryGraphState, RecoveryGraphDeps, RecoverySignal],
        ) -> RecoverySignal:
            state = ctx.state.workflow_state
            if state is None or state.planning is None:
                raise RuntimeError("Continuation requires recovered planning state.")
            prompt = render_recovery_prompt(
                ctx.deps.input_data,
                CONTINUATION_STEP,
                planning=state.planning,
            )
            started_at = utc_now()
            call_record = None
            error = None
            try:
                call_record = await asyncio.to_thread(
                    complete_agent_step,
                    agent=ctx.deps.agents[CONTINUATION_STEP],
                    prompt=prompt,
                    input_data=ctx.deps.input_data,
                    config=ctx.deps.config,
                    step_id=6,
                )
                response = call_record.response.strip()
            except Exception as exc:  # pragma: no cover - integration failure path
                error = f"{type(exc).__name__}: {exc}"
                response = ""
            continuation = parse_continuation_result(response, error=error)
            finished_at = utc_now()
            ctx.state.workflow_state = state.model_copy(
                update={"continuation": continuation, "current_stage": CONTINUATION_STEP}
            )
            ctx.state.steps.append(
                make_recovery_step(
                    step_id=6,
                    component=CONTINUATION_STEP,
                    actor="pydantic_ai.graph.continuation_step",
                    started_at=started_at,
                    finished_at=finished_at,
                    framework_primitive=FRAMEWORK_PRIMITIVE,
                    input_payload={"prompt": prompt, "recovered": state.recovered},
                    output={"continuation": continuation.model_dump()},
                    llm_call_ids=[call_record.metrics.call_id] if call_record else [],
                    depends_on=[RECOVERY_LOADER],
                    error=error,
                )
            )
            if call_record:
                ctx.state.llm_calls.append(call_record.metrics)
            return RecoverySignal(stage=CONTINUATION_STEP)

        async def finalizer(
            ctx: StepContext[RecoveryGraphState, RecoveryGraphDeps, RecoverySignal],
        ) -> RecoveryGraphOutput:
            state = ctx.state.workflow_state
            if state is None or state.continuation is None:
                raise RuntimeError("Finalizer requires continuation output.")
            started_at = utc_now()
            state = state.model_copy(
                update={
                    "current_stage": FINALIZER,
                    "result_generated_after_recovery": failure_injected,
                }
            )
            finished_at = utc_now()
            ctx.state.workflow_state = state
            ctx.state.steps.append(
                make_recovery_step(
                    step_id=7,
                    component=FINALIZER,
                    actor="pydantic_ai.graph.finalizer",
                    started_at=started_at,
                    finished_at=finished_at,
                    framework_primitive=FRAMEWORK_PRIMITIVE,
                    output={
                        "answer": state.continuation.answer,
                        "result_generated_after_recovery": failure_injected,
                    },
                    depends_on=[CONTINUATION_STEP],
                )
            )
            return RecoveryGraphOutput(
                state=state,
                steps=ctx.state.steps,
                llm_calls=ctx.state.llm_calls,
            )

        recovery_node = post_builder.step(recovery_loader, node_id=RECOVERY_LOADER)
        continuation_node = post_builder.step(continuation_step, node_id=CONTINUATION_STEP)
        finalizer_node = post_builder.step(finalizer, node_id=FINALIZER)
        post_builder.add(post_builder.edge_from(post_builder.start_node).to(recovery_node))
        post_builder.add(post_builder.edge_from(recovery_node).to(continuation_node))
        post_builder.add(post_builder.edge_from(continuation_node).to(finalizer_node))
        post_builder.add(post_builder.edge_from(finalizer_node).to(post_builder.end_node))
        post_graph = post_builder.build()
        output = asyncio.run(
            asyncio.wait_for(
                post_graph.run(
                    inputs=CheckpointSignal(
                        checkpoint_path=str(checkpoint_file),
                        checkpoint_id=checkpoint_id,
                    ),
                    state=post_state,
                    deps=deps,
                ),
                timeout=float(config.timeout_seconds),
            )
        )
        resource_usage = monitor.usage

    final_answer, structured_output = build_recovery_structured_output(
        input_data=input_data,
        config=config,
        state=output.state,
        steps=output.steps,
        llm_calls=output.llm_calls,
        framework_execution="pydantic_ai_two_phase_typed_graph_recovery",
        framework_primitive=FRAMEWORK_PRIMITIVE,
        checkpoint_backend=CHECKPOINT_BACKEND,
        native_checkpointing=False,
        recovery_source=str(checkpoint_file),
        failure_injected=failure_injected,
        recovery_attempted=failure_injected,
        recovery_successful=failure_injected,
    )
    return PydanticAIRunOutput(
        final_answer=final_answer,
        structured_output=structured_output,
        steps=output.steps,
        llm_calls=output.llm_calls,
        resource_usage=resource_usage,
    )
