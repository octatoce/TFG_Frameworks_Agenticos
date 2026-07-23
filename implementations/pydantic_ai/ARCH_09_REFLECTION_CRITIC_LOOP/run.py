"""Pydantic AI + pydantic-graph implementation for ARCH_09."""

import asyncio
from typing import Any

from pydantic import BaseModel, Field
from pydantic_graph import GraphBuilder, StepContext

from benchmark_core.reflection_critic_loop import (
    CRITIC,
    GENERATOR,
    REVISER,
    STOP_CONTROLLER,
    CritiqueEvaluation,
    ReflectionSettings,
    ReflectionVersion,
    StopDecision,
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
from benchmark_core.schemas import AgentStep, ExperimentConfig, ExperimentInput, LLMCallMetrics
from benchmark_core.tracing import utc_now
from implementations.pydantic_ai.utils_pydantic_ai import (
    PydanticAIRunContext,
    PydanticAIRunOutput,
    build_typed_agent,
    complete_agent_step,
    pydantic_ai_architecture_runner,
)


FRAMEWORK_PRIMITIVE = "GraphBuilder.Step.Decision.stop_or_reviser_cycle"


class ReflectionSignal(BaseModel):
    current_version_index: int
    iteration: int


class ReflectionGraphState(BaseModel):
    current_version: ReflectionVersion | None = None
    versions: list[ReflectionVersion] = Field(default_factory=list)
    critiques: list[CritiqueEvaluation] = Field(default_factory=list)
    stop_decisions: list[StopDecision] = Field(default_factory=list)
    iteration: int = 0
    steps: list[AgentStep] = Field(default_factory=list)
    llm_calls: list[LLMCallMetrics] = Field(default_factory=list)


class ReflectionGraphDeps:
    def __init__(
        self,
        *,
        input_data: ExperimentInput,
        config: ExperimentConfig,
        context: PydanticAIRunContext,
        settings: ReflectionSettings,
        agents: dict[str, Any],
    ) -> None:
        self.input_data = input_data
        self.config = config
        self.context = context
        self.settings = settings
        self.agents = agents


class ReflectionGraphOutput(BaseModel):
    final_answer: str
    structured_output: dict[str, Any]
    final_version: ReflectionVersion
    final_stop_decision: StopDecision
    steps: list[AgentStep]
    llm_calls: list[LLMCallMetrics]


@pydantic_ai_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: PydanticAIRunContext,
) -> PydanticAIRunOutput:
    """Execute a typed GraphBuilder cycle with a StopDecision branch."""

    settings = get_reflection_settings(config)
    agents = {
        component: build_typed_agent(
            name=component,
            instructions=(
                f"You are the typed bounded {component} component in ARCH_09. "
                "Operate on one answer; do not route, hand off, debate, supervise, or create hidden loops."
            ),
            context=context,
            input_data=input_data,
            config=config,
        )
        for component in (GENERATOR, CRITIC, REVISER)
    }
    state = ReflectionGraphState()
    deps = ReflectionGraphDeps(
        input_data=input_data,
        config=config,
        context=context,
        settings=settings,
        agents=agents,
    )
    builder = GraphBuilder(
        name="ARCH_09_REFLECTION_CRITIC_LOOP",
        state_type=ReflectionGraphState,
        deps_type=ReflectionGraphDeps,
        input_type=ExperimentInput,
        output_type=ReflectionGraphOutput,
        auto_instrument=False,
    )

    async def generator(
        ctx: StepContext[ReflectionGraphState, ReflectionGraphDeps, ExperimentInput],
    ) -> ReflectionSignal:
        prompt = render_reflection_prompt(
            ctx.inputs,
            GENERATOR,
            iteration=0,
            settings=ctx.deps.settings,
        )
        started_at = utc_now()
        call_record = None
        error = None
        try:
            call_record = await asyncio.to_thread(
                complete_agent_step,
                agent=ctx.deps.agents[GENERATOR],
                prompt=prompt,
                input_data=ctx.deps.input_data,
                config=ctx.deps.config,
                step_id=1,
            )
            response = call_record.response.strip()
        except Exception as exc:  # pragma: no cover - integration failure path
            error = f"{type(exc).__name__}: {exc}"
            response = ""
        version = parse_reflection_version(
            response,
            version_index=0,
            iteration=0,
            created_by=GENERATOR,
            error=error,
        )
        finished_at = utc_now()
        ctx.state.current_version = version
        ctx.state.versions.append(version)
        ctx.state.steps.append(
            make_reflection_step(
                step_id=1,
                component=GENERATOR,
                iteration=0,
                actor="pydantic_ai.graph.generator",
                prompt=prompt,
                output={"version": version.model_dump()},
                llm_call_ids=[call_record.metrics.call_id] if call_record else [],
                started_at=started_at,
                finished_at=finished_at,
                framework_primitive=FRAMEWORK_PRIMITIVE,
                max_iterations=ctx.deps.settings.max_iterations,
                error=error,
            )
        )
        if call_record:
            ctx.state.llm_calls.append(call_record.metrics)
        return ReflectionSignal(current_version_index=0, iteration=0)

    async def critic(
        ctx: StepContext[ReflectionGraphState, ReflectionGraphDeps, ReflectionSignal],
    ) -> CritiqueEvaluation:
        if ctx.state.current_version is None:
            raise RuntimeError("Critic requires a current reflection version.")
        iteration = ctx.state.iteration + 1
        current_version = ctx.state.current_version
        prompt = render_reflection_prompt(
            ctx.deps.input_data,
            CRITIC,
            iteration=iteration,
            settings=ctx.deps.settings,
            current_version=current_version.model_dump(),
        )
        step_id = len(ctx.state.steps) + 1
        started_at = utc_now()
        call_record = None
        error = None
        try:
            call_record = await asyncio.to_thread(
                complete_agent_step,
                agent=ctx.deps.agents[CRITIC],
                prompt=prompt,
                input_data=ctx.deps.input_data,
                config=ctx.deps.config,
                step_id=step_id,
            )
            response = call_record.response.strip()
        except Exception as exc:  # pragma: no cover - integration failure path
            error = f"{type(exc).__name__}: {exc}"
            response = ""
        critique = parse_critique(response, iteration=iteration, error=error)
        finished_at = utc_now()
        previous_step = (
            GENERATOR
            if current_version.created_by == GENERATOR
            else reflection_step_name(REVISER, current_version.iteration)
        )
        ctx.state.iteration = iteration
        ctx.state.critiques.append(critique)
        ctx.state.steps.append(
            make_reflection_step(
                step_id=step_id,
                component=CRITIC,
                iteration=iteration,
                actor="pydantic_ai.graph.critic",
                prompt=prompt,
                current_version=current_version.model_dump(),
                output={"critique": critique.model_dump()},
                llm_call_ids=[call_record.metrics.call_id] if call_record else [],
                started_at=started_at,
                finished_at=finished_at,
                framework_primitive=FRAMEWORK_PRIMITIVE,
                max_iterations=ctx.deps.settings.max_iterations,
                depends_on=[previous_step],
                error=error,
            )
        )
        if call_record:
            ctx.state.llm_calls.append(call_record.metrics)
        return critique

    async def stop_controller(
        ctx: StepContext[ReflectionGraphState, ReflectionGraphDeps, CritiqueEvaluation],
    ) -> StopDecision:
        if ctx.state.current_version is None:
            raise RuntimeError("Stop controller requires a current reflection version.")
        started_at = utc_now()
        decision = evaluate_stop(
            ctx.inputs,
            current_version_index=ctx.state.current_version.version_index,
            settings=ctx.deps.settings,
        )
        finished_at = utc_now()
        ctx.state.stop_decisions.append(decision)
        ctx.state.steps.append(
            make_reflection_step(
                step_id=len(ctx.state.steps) + 1,
                component=STOP_CONTROLLER,
                iteration=ctx.state.iteration,
                actor="pydantic_ai.graph.stop_controller",
                current_version=ctx.state.current_version.model_dump(),
                critique=ctx.inputs.model_dump(),
                output={"stop_decision": decision.model_dump()},
                started_at=started_at,
                finished_at=finished_at,
                framework_primitive=FRAMEWORK_PRIMITIVE,
                max_iterations=ctx.deps.settings.max_iterations,
                depends_on=[reflection_step_name(CRITIC, ctx.state.iteration)],
            )
        )
        return decision

    async def reviser(
        ctx: StepContext[ReflectionGraphState, ReflectionGraphDeps, StopDecision],
    ) -> ReflectionSignal:
        if ctx.state.current_version is None:
            raise RuntimeError("Reviser requires a current reflection version.")
        current_version = ctx.state.current_version
        critique = ctx.state.critiques[-1]
        prompt = render_reflection_prompt(
            ctx.deps.input_data,
            REVISER,
            iteration=ctx.state.iteration,
            settings=ctx.deps.settings,
            current_version=current_version.model_dump(),
            critique=critique.model_dump(),
        )
        step_id = len(ctx.state.steps) + 1
        started_at = utc_now()
        call_record = None
        error = None
        try:
            call_record = await asyncio.to_thread(
                complete_agent_step,
                agent=ctx.deps.agents[REVISER],
                prompt=prompt,
                input_data=ctx.deps.input_data,
                config=ctx.deps.config,
                step_id=step_id,
            )
            response = call_record.response.strip()
        except Exception as exc:  # pragma: no cover - integration failure path
            error = f"{type(exc).__name__}: {exc}"
            response = ""
        version = parse_reflection_version(
            response,
            version_index=len(ctx.state.versions),
            iteration=ctx.state.iteration,
            created_by=REVISER,
            error=error,
        )
        finished_at = utc_now()
        ctx.state.current_version = version
        ctx.state.versions.append(version)
        ctx.state.steps.append(
            make_reflection_step(
                step_id=step_id,
                component=REVISER,
                iteration=ctx.state.iteration,
                actor="pydantic_ai.graph.reviser",
                prompt=prompt,
                current_version=current_version.model_dump(),
                critique=critique.model_dump(),
                stop_decision=ctx.inputs.model_dump(),
                output={"version": version.model_dump()},
                llm_call_ids=[call_record.metrics.call_id] if call_record else [],
                started_at=started_at,
                finished_at=finished_at,
                framework_primitive=FRAMEWORK_PRIMITIVE,
                max_iterations=ctx.deps.settings.max_iterations,
                depends_on=[
                    reflection_step_name(CRITIC, ctx.state.iteration),
                    reflection_step_name(STOP_CONTROLLER, ctx.state.iteration),
                ],
                error=error,
            )
        )
        if call_record:
            ctx.state.llm_calls.append(call_record.metrics)
        return ReflectionSignal(
            current_version_index=version.version_index,
            iteration=ctx.state.iteration,
        )

    async def finalize(
        ctx: StepContext[ReflectionGraphState, ReflectionGraphDeps, StopDecision],
    ) -> ReflectionGraphOutput:
        if ctx.state.current_version is None:
            raise RuntimeError("Finalizer requires a current reflection version.")
        final_answer, structured_output = build_reflection_structured_output(
            input_data=ctx.deps.input_data,
            config=ctx.deps.config,
            versions=ctx.state.versions,
            critiques=ctx.state.critiques,
            stop_decisions=ctx.state.stop_decisions,
            steps=ctx.state.steps,
            llm_calls=ctx.state.llm_calls,
            settings=ctx.deps.settings,
            framework_execution="pydantic_ai_typed_graph_reflection_cycle",
            framework_primitive=FRAMEWORK_PRIMITIVE,
        )
        return ReflectionGraphOutput(
            final_answer=final_answer,
            structured_output=structured_output,
            final_version=ctx.state.current_version,
            final_stop_decision=ctx.inputs,
            steps=ctx.state.steps,
            llm_calls=ctx.state.llm_calls,
        )

    generator_step = builder.step(generator, node_id=GENERATOR)
    critic_step = builder.step(critic, node_id=CRITIC)
    stop_step = builder.step(stop_controller, node_id=STOP_CONTROLLER)
    reviser_step = builder.step(reviser, node_id=REVISER)
    finalize_step = builder.step(finalize, node_id="reflection_finalizer")
    stop_decision = builder.decision(node_id="stop_decision")
    stop_decision = stop_decision.branch(
        builder.match(
            StopDecision,
            matches=lambda decision: decision.should_stop,
        ).to(finalize_step)
    )
    stop_decision = stop_decision.branch(
        builder.match(
            StopDecision,
            matches=lambda decision: not decision.should_stop,
        ).to(reviser_step)
    )

    builder.add(builder.edge_from(builder.start_node).to(generator_step))
    builder.add(builder.edge_from(generator_step, reviser_step).to(critic_step))
    builder.add(builder.edge_from(critic_step).to(stop_step))
    builder.add(builder.edge_from(stop_step).to(stop_decision))
    builder.add(builder.edge_from(finalize_step).to(builder.end_node))
    graph = builder.build()

    async def execute_graph():
        return await asyncio.wait_for(
            graph.run(inputs=input_data, state=state, deps=deps),
            timeout=float(config.timeout_seconds),
        )

    with ResourceMonitor() as monitor:
        output = asyncio.run(execute_graph())
        resource_usage = monitor.usage
    return PydanticAIRunOutput(
        final_answer=output.final_answer,
        structured_output=output.structured_output,
        steps=output.steps,
        llm_calls=output.llm_calls,
        resource_usage=resource_usage,
    )
