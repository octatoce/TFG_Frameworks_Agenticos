"""Microsoft Agent Framework implementation for ARCH_09_REFLECTION_CRITIC_LOOP."""

import asyncio
from typing import Any, Never

from pydantic import BaseModel, Field

from benchmark_core.reflection_critic_loop import (
    CRITIC,
    GENERATOR,
    REVISER,
    STOP_CONTROLLER,
    CritiqueEvaluation,
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


FRAMEWORK_PRIMITIVE = "WorkflowBuilder.switch_case.stop_or_reviser_cycle"


class ReflectionPayload(BaseModel):
    current_version: ReflectionVersion
    versions: list[ReflectionVersion]
    critiques: list[CritiqueEvaluation] = Field(default_factory=list)
    stop_decisions: list[StopDecision] = Field(default_factory=list)
    iteration: int = 0
    steps: list[AgentStep] = Field(default_factory=list)
    llm_calls: list[LLMCallMetrics] = Field(default_factory=list)


class CriticPayload(BaseModel):
    state: ReflectionPayload
    critique: CritiqueEvaluation


class StopPayload(BaseModel):
    state: ReflectionPayload
    decision: StopDecision


class WorkflowPayload(BaseModel):
    final_answer: str
    structured_output: dict[str, Any]
    steps: list[AgentStep]
    llm_calls: list[LLMCallMetrics]


@microsoft_agent_framework_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: MicrosoftAgentFrameworkRunContext,
) -> MicrosoftAgentFrameworkRunOutput:
    """Execute a native switch-case workflow with a bounded reviser cycle."""

    settings = get_reflection_settings(config)

    def execute() -> MicrosoftAgentFrameworkRunOutput:
        from agent_framework import (
            Case,
            Default,
            Executor,
            WorkflowBuilder,
            WorkflowContext,
            handler,
        )

        agents = {
            component: build_agent(
                name=component,
                instructions=(
                    f"You are the bounded {component} component of ARCH_09. "
                    "Improve one answer only; do not delegate, hand off, route, debate, or plan."
                ),
                context=context,
                input_data=input_data,
                config=config,
            )
            for component in (GENERATOR, CRITIC, REVISER)
        }

        class GeneratorExecutor(Executor):
            @handler
            async def generate(
                self,
                _message: ExperimentInput,
                ctx: WorkflowContext[ReflectionPayload],
            ) -> None:
                prompt = render_reflection_prompt(
                    input_data,
                    GENERATOR,
                    iteration=0,
                    settings=settings,
                )
                started_at = utc_now()
                call_record = None
                error = None
                try:
                    call_record = await asyncio.to_thread(
                        complete_agent_step,
                        agent=agents[GENERATOR],
                        prompt=prompt,
                        input_data=input_data,
                        config=config,
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
                step = make_reflection_step(
                    step_id=1,
                    component=GENERATOR,
                    iteration=0,
                    actor="microsoft_agent_framework.generator_executor",
                    prompt=prompt,
                    output={"version": version.model_dump()},
                    llm_call_ids=[call_record.metrics.call_id] if call_record else [],
                    started_at=started_at,
                    finished_at=finished_at,
                    framework_primitive=FRAMEWORK_PRIMITIVE,
                    max_iterations=settings.max_iterations,
                    error=error,
                )
                await ctx.send_message(
                    ReflectionPayload(
                        current_version=version,
                        versions=[version],
                        steps=[step],
                        llm_calls=[call_record.metrics] if call_record else [],
                    )
                )

        class CriticExecutor(Executor):
            @handler
            async def criticize(
                self,
                state: ReflectionPayload,
                ctx: WorkflowContext[CriticPayload],
            ) -> None:
                iteration = state.iteration + 1
                prompt = render_reflection_prompt(
                    input_data,
                    CRITIC,
                    iteration=iteration,
                    settings=settings,
                    current_version=state.current_version.model_dump(),
                )
                step_id = len(state.steps) + 1
                started_at = utc_now()
                call_record = None
                error = None
                try:
                    call_record = await asyncio.to_thread(
                        complete_agent_step,
                        agent=agents[CRITIC],
                        prompt=prompt,
                        input_data=input_data,
                        config=config,
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
                    if state.current_version.created_by == GENERATOR
                    else reflection_step_name(REVISER, state.current_version.iteration)
                )
                step = make_reflection_step(
                    step_id=step_id,
                    component=CRITIC,
                    iteration=iteration,
                    actor="microsoft_agent_framework.critic_executor",
                    prompt=prompt,
                    current_version=state.current_version.model_dump(),
                    output={"critique": critique.model_dump()},
                    llm_call_ids=[call_record.metrics.call_id] if call_record else [],
                    started_at=started_at,
                    finished_at=finished_at,
                    framework_primitive=FRAMEWORK_PRIMITIVE,
                    max_iterations=settings.max_iterations,
                    depends_on=[previous_step],
                    error=error,
                )
                updated = state.model_copy(
                    update={
                        "iteration": iteration,
                        "critiques": [*state.critiques, critique],
                        "steps": [*state.steps, step],
                        "llm_calls": [
                            *state.llm_calls,
                            *([call_record.metrics] if call_record else []),
                        ],
                    }
                )
                await ctx.send_message(CriticPayload(state=updated, critique=critique))

        class StopControllerExecutor(Executor):
            @handler
            async def evaluate(
                self,
                payload: CriticPayload,
                ctx: WorkflowContext[StopPayload],
            ) -> None:
                state = payload.state
                started_at = utc_now()
                decision = evaluate_stop(
                    payload.critique,
                    current_version_index=state.current_version.version_index,
                    settings=settings,
                )
                finished_at = utc_now()
                step = make_reflection_step(
                    step_id=len(state.steps) + 1,
                    component=STOP_CONTROLLER,
                    iteration=state.iteration,
                    actor="microsoft_agent_framework.stop_controller_executor",
                    current_version=state.current_version.model_dump(),
                    critique=payload.critique.model_dump(),
                    output={"stop_decision": decision.model_dump()},
                    started_at=started_at,
                    finished_at=finished_at,
                    framework_primitive=FRAMEWORK_PRIMITIVE,
                    max_iterations=settings.max_iterations,
                    depends_on=[reflection_step_name(CRITIC, state.iteration)],
                )
                updated = state.model_copy(
                    update={
                        "stop_decisions": [*state.stop_decisions, decision],
                        "steps": [*state.steps, step],
                    }
                )
                await ctx.send_message(StopPayload(state=updated, decision=decision))

        class ReviserExecutor(Executor):
            @handler
            async def revise(
                self,
                payload: StopPayload,
                ctx: WorkflowContext[ReflectionPayload],
            ) -> None:
                state = payload.state
                critique = state.critiques[-1]
                prompt = render_reflection_prompt(
                    input_data,
                    REVISER,
                    iteration=state.iteration,
                    settings=settings,
                    current_version=state.current_version.model_dump(),
                    critique=critique.model_dump(),
                )
                step_id = len(state.steps) + 1
                started_at = utc_now()
                call_record = None
                error = None
                try:
                    call_record = await asyncio.to_thread(
                        complete_agent_step,
                        agent=agents[REVISER],
                        prompt=prompt,
                        input_data=input_data,
                        config=config,
                        step_id=step_id,
                    )
                    response = call_record.response.strip()
                except Exception as exc:  # pragma: no cover - integration failure path
                    error = f"{type(exc).__name__}: {exc}"
                    response = ""
                version = parse_reflection_version(
                    response,
                    version_index=len(state.versions),
                    iteration=state.iteration,
                    created_by=REVISER,
                    error=error,
                )
                finished_at = utc_now()
                step = make_reflection_step(
                    step_id=step_id,
                    component=REVISER,
                    iteration=state.iteration,
                    actor="microsoft_agent_framework.reviser_executor",
                    prompt=prompt,
                    current_version=state.current_version.model_dump(),
                    critique=critique.model_dump(),
                    stop_decision=payload.decision.model_dump(),
                    output={"version": version.model_dump()},
                    llm_call_ids=[call_record.metrics.call_id] if call_record else [],
                    started_at=started_at,
                    finished_at=finished_at,
                    framework_primitive=FRAMEWORK_PRIMITIVE,
                    max_iterations=settings.max_iterations,
                    depends_on=[
                        reflection_step_name(CRITIC, state.iteration),
                        reflection_step_name(STOP_CONTROLLER, state.iteration),
                    ],
                    error=error,
                )
                await ctx.send_message(
                    state.model_copy(
                        update={
                            "current_version": version,
                            "versions": [*state.versions, version],
                            "steps": [*state.steps, step],
                            "llm_calls": [
                                *state.llm_calls,
                                *([call_record.metrics] if call_record else []),
                            ],
                        }
                    )
                )

        class FinalizerExecutor(Executor):
            @handler
            async def finalize(
                self,
                payload: StopPayload,
                ctx: WorkflowContext[Never, WorkflowPayload],
            ) -> None:
                state = payload.state
                final_answer, structured_output = build_reflection_structured_output(
                    input_data=input_data,
                    config=config,
                    versions=state.versions,
                    critiques=state.critiques,
                    stop_decisions=state.stop_decisions,
                    steps=state.steps,
                    llm_calls=state.llm_calls,
                    settings=settings,
                    framework_execution="microsoft_native_switch_case_reflection_cycle",
                    framework_primitive=FRAMEWORK_PRIMITIVE,
                )
                await ctx.yield_output(
                    WorkflowPayload(
                        final_answer=final_answer,
                        structured_output=structured_output,
                        steps=state.steps,
                        llm_calls=state.llm_calls,
                    )
                )

        generator = GeneratorExecutor(id=GENERATOR)
        critic = CriticExecutor(id=CRITIC)
        stop_controller = StopControllerExecutor(id=STOP_CONTROLLER)
        reviser = ReviserExecutor(id=REVISER)
        finalizer = FinalizerExecutor(id="reflection_finalizer")
        workflow = (
            WorkflowBuilder(
                start_executor=generator,
                output_from=[finalizer],
                max_iterations=settings.max_iterations * 3 + 4,
                name="ARCH_09_REFLECTION_CRITIC_LOOP",
            )
            .add_edge(generator, critic)
            .add_edge(critic, stop_controller)
            .add_switch_case_edge_group(
                stop_controller,
                [
                    Case(condition=lambda payload: payload.decision.should_stop, target=finalizer),
                    Default(target=reviser),
                ],
            )
            .add_edge(reviser, critic)
            .build()
        )

        async def run_workflow():
            return await asyncio.wait_for(
                workflow.run(input_data),
                timeout=float(config.timeout_seconds),
            )

        result = run_async(run_workflow())
        outputs = result.get_outputs()
        if len(outputs) != 1 or not isinstance(outputs[0], WorkflowPayload):
            raise RuntimeError("Microsoft reflection workflow did not produce one WorkflowPayload.")
        payload = outputs[0]
        return MicrosoftAgentFrameworkRunOutput(
            final_answer=payload.final_answer,
            structured_output=payload.structured_output,
            steps=payload.steps,
            llm_calls=payload.llm_calls,
        )

    return run_with_resource_monitor(execute)
