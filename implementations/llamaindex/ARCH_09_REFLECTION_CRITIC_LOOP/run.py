"""LlamaIndex implementation for ARCH_09_REFLECTION_CRITIC_LOOP."""

import asyncio
from typing import Any

from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step

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
from implementations.llamaindex.utils_llamaindex import (
    LlamaIndexRunContext,
    LlamaIndexRunOutput,
    build_function_agent,
    complete_agent_step,
    llamaindex_architecture_runner,
    run_async,
    run_with_resource_monitor,
)


FRAMEWORK_PRIMITIVE = "Workflow.events.generator.critic.stop_or_reviser_cycle"


class ReflectionStateEvent(Event):
    current_version: dict[str, Any]
    versions: list[dict[str, Any]]
    critiques: list[dict[str, Any]]
    stop_decisions: list[dict[str, Any]]
    iteration: int
    steps: list[Any]
    llm_calls: list[Any]


class CriticResultEvent(Event):
    state: ReflectionStateEvent
    critique: dict[str, Any]


class ReviserRequestEvent(Event):
    state: ReflectionStateEvent
    decision: dict[str, Any]


class WorkflowPayload:
    def __init__(
        self,
        *,
        final_answer: str,
        structured_output: dict[str, Any],
        steps: list[AgentStep],
        llm_calls: list[LLMCallMetrics],
    ) -> None:
        self.final_answer = final_answer
        self.structured_output = structured_output
        self.steps = steps
        self.llm_calls = llm_calls


@llamaindex_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: LlamaIndexRunContext,
) -> LlamaIndexRunOutput:
    """Execute an event-driven generator, critic, stop, and reviser cycle."""

    settings = get_reflection_settings(config)
    agents = {
        component: build_function_agent(
            name=component,
            system_prompt=(
                f"You are the bounded {component} component of ARCH_09. "
                "Operate on one evolving answer without handoffs, routing, debate, or hidden loops."
            ),
            context=context,
            input_data=input_data,
            config=config,
        )
        for component in (GENERATOR, CRITIC, REVISER)
    }

    class ReflectionWorkflow(Workflow):
        @step
        async def generator(self, _ev: StartEvent) -> ReflectionStateEvent:
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
            step_value = make_reflection_step(
                step_id=1,
                component=GENERATOR,
                iteration=0,
                actor="llamaindex.workflow.generator",
                prompt=prompt,
                output={"version": version.model_dump()},
                llm_call_ids=[call_record.metrics.call_id] if call_record else [],
                started_at=started_at,
                finished_at=finished_at,
                framework_primitive=FRAMEWORK_PRIMITIVE,
                max_iterations=settings.max_iterations,
                error=error,
            )
            return ReflectionStateEvent(
                current_version=version.model_dump(),
                versions=[version.model_dump()],
                critiques=[],
                stop_decisions=[],
                iteration=0,
                steps=[step_value],
                llm_calls=[call_record.metrics] if call_record else [],
            )

        @step
        async def critic(self, ev: ReflectionStateEvent) -> CriticResultEvent:
            iteration = ev.iteration + 1
            current_version = ReflectionVersion.model_validate(ev.current_version)
            prompt = render_reflection_prompt(
                input_data,
                CRITIC,
                iteration=iteration,
                settings=settings,
                current_version=current_version.model_dump(),
            )
            step_id = len(ev.steps) + 1
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
                if current_version.created_by == GENERATOR
                else reflection_step_name(REVISER, current_version.iteration)
            )
            step_value = make_reflection_step(
                step_id=step_id,
                component=CRITIC,
                iteration=iteration,
                actor="llamaindex.workflow.critic",
                prompt=prompt,
                current_version=current_version.model_dump(),
                output={"critique": critique.model_dump()},
                llm_call_ids=[call_record.metrics.call_id] if call_record else [],
                started_at=started_at,
                finished_at=finished_at,
                framework_primitive=FRAMEWORK_PRIMITIVE,
                max_iterations=settings.max_iterations,
                depends_on=[previous_step],
                error=error,
            )
            state = ReflectionStateEvent(
                current_version=ev.current_version,
                versions=ev.versions,
                critiques=[*ev.critiques, critique.model_dump()],
                stop_decisions=ev.stop_decisions,
                iteration=iteration,
                steps=[*ev.steps, step_value],
                llm_calls=[
                    *ev.llm_calls,
                    *([call_record.metrics] if call_record else []),
                ],
            )
            return CriticResultEvent(state=state, critique=critique.model_dump())

        @step
        async def stop_controller(
            self,
            ev: CriticResultEvent,
        ) -> ReviserRequestEvent | StopEvent:
            state = ev.state
            critique = CritiqueEvaluation.model_validate(ev.critique)
            current_version = ReflectionVersion.model_validate(state.current_version)
            started_at = utc_now()
            decision = evaluate_stop(
                critique,
                current_version_index=current_version.version_index,
                settings=settings,
            )
            finished_at = utc_now()
            step_value = make_reflection_step(
                step_id=len(state.steps) + 1,
                component=STOP_CONTROLLER,
                iteration=state.iteration,
                actor="llamaindex.workflow.stop_controller",
                current_version=current_version.model_dump(),
                critique=critique.model_dump(),
                output={"stop_decision": decision.model_dump()},
                started_at=started_at,
                finished_at=finished_at,
                framework_primitive=FRAMEWORK_PRIMITIVE,
                max_iterations=settings.max_iterations,
                depends_on=[reflection_step_name(CRITIC, state.iteration)],
            )
            updated = ReflectionStateEvent(
                current_version=state.current_version,
                versions=state.versions,
                critiques=state.critiques,
                stop_decisions=[*state.stop_decisions, decision.model_dump()],
                iteration=state.iteration,
                steps=[*state.steps, step_value],
                llm_calls=state.llm_calls,
            )
            if not decision.should_stop:
                return ReviserRequestEvent(
                    state=updated,
                    decision=decision.model_dump(),
                )

            versions = [ReflectionVersion.model_validate(item) for item in updated.versions]
            critiques = [CritiqueEvaluation.model_validate(item) for item in updated.critiques]
            stop_decisions = [StopDecision.model_validate(item) for item in updated.stop_decisions]
            final_answer, structured_output = build_reflection_structured_output(
                input_data=input_data,
                config=config,
                versions=versions,
                critiques=critiques,
                stop_decisions=stop_decisions,
                steps=updated.steps,
                llm_calls=updated.llm_calls,
                settings=settings,
                framework_execution="llamaindex_native_event_reflection_cycle",
                framework_primitive=FRAMEWORK_PRIMITIVE,
            )
            return StopEvent(
                result=WorkflowPayload(
                    final_answer=final_answer,
                    structured_output=structured_output,
                    steps=updated.steps,
                    llm_calls=updated.llm_calls,
                )
            )

        @step
        async def reviser(self, ev: ReviserRequestEvent) -> ReflectionStateEvent:
            state = ev.state
            current_version = ReflectionVersion.model_validate(state.current_version)
            critique = CritiqueEvaluation.model_validate(state.critiques[-1])
            decision = StopDecision.model_validate(ev.decision)
            prompt = render_reflection_prompt(
                input_data,
                REVISER,
                iteration=state.iteration,
                settings=settings,
                current_version=current_version.model_dump(),
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
            step_value = make_reflection_step(
                step_id=step_id,
                component=REVISER,
                iteration=state.iteration,
                actor="llamaindex.workflow.reviser",
                prompt=prompt,
                current_version=current_version.model_dump(),
                critique=critique.model_dump(),
                stop_decision=decision.model_dump(),
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
            return ReflectionStateEvent(
                current_version=version.model_dump(),
                versions=[*state.versions, version.model_dump()],
                critiques=state.critiques,
                stop_decisions=state.stop_decisions,
                iteration=state.iteration,
                steps=[*state.steps, step_value],
                llm_calls=[
                    *state.llm_calls,
                    *([call_record.metrics] if call_record else []),
                ],
            )

    workflow = ReflectionWorkflow(
        timeout=float(config.timeout_seconds),
        verbose=False,
        num_concurrent_runs=1,
    )

    def execute_workflow() -> LlamaIndexRunOutput:
        async def run_workflow():
            handler = workflow.run(common_input=input_data.model_dump(mode="json"))
            return await handler

        payload = run_async(run_workflow())
        if not isinstance(payload, WorkflowPayload):
            raise RuntimeError("LlamaIndex reflection workflow did not return WorkflowPayload.")
        return LlamaIndexRunOutput(
            final_answer=payload.final_answer,
            structured_output=payload.structured_output,
            steps=payload.steps,
            llm_calls=payload.llm_calls,
        )

    return run_with_resource_monitor(execute_workflow)
