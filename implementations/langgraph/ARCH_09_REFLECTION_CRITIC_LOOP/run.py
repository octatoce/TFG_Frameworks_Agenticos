"""LangGraph implementation for ARCH_09_REFLECTION_CRITIC_LOOP."""

from __future__ import annotations

from typing import TypedDict

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
from implementations.langgraph.utils_langgraph import (
    LangGraphRunContext,
    LangGraphRunOutput,
    complete_llm_step,
    langgraph_architecture_runner,
)


FRAMEWORK_PRIMITIVE = "StateGraph.generator.critic.stop_controller.conditional_reviser_cycle"


class ReflectionState(TypedDict, total=False):
    settings: object
    current_version: object
    versions: list[object]
    critiques: list[object]
    stop_decisions: list[object]
    iteration: int
    should_stop: bool
    stop_reason: str | None
    steps: list[object]
    llm_calls: list[object]


@langgraph_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: LangGraphRunContext,
) -> LangGraphRunOutput:
    """Execute an explicit bounded StateGraph reflection cycle."""

    from langgraph.graph import END, START, StateGraph

    settings = get_reflection_settings(config)

    def generator_node(state: ReflectionState) -> dict:
        prompt = render_reflection_prompt(
            input_data,
            GENERATOR,
            iteration=0,
            settings=settings,
        )
        step_id = len(state["steps"]) + 1
        started_at = utc_now()
        call_record = None
        error = None
        try:
            call_record = complete_llm_step(
                llm=context.llm,
                input_data=input_data,
                config=config,
                prompt=prompt,
                step_id=step_id,
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
            step_id=step_id,
            component=GENERATOR,
            iteration=0,
            actor="langgraph.generator_node",
            prompt=prompt,
            output={"version": version.model_dump()},
            llm_call_ids=[call_record.metrics.call_id] if call_record else [],
            started_at=started_at,
            finished_at=finished_at,
            framework_primitive=FRAMEWORK_PRIMITIVE,
            max_iterations=settings.max_iterations,
            error=error,
        )
        return {
            "current_version": version,
            "versions": [version],
            "steps": [step],
            "llm_calls": [call_record.metrics] if call_record else [],
        }

    def critic_node(state: ReflectionState) -> dict:
        iteration = state["iteration"] + 1
        current_version = state["current_version"]
        prompt = render_reflection_prompt(
            input_data,
            CRITIC,
            iteration=iteration,
            settings=settings,
            current_version=current_version.model_dump(),
        )
        step_id = len(state["steps"]) + 1
        started_at = utc_now()
        call_record = None
        error = None
        try:
            call_record = complete_llm_step(
                llm=context.llm,
                input_data=input_data,
                config=config,
                prompt=prompt,
                step_id=step_id,
            )
            response = call_record.response.strip()
        except Exception as exc:  # pragma: no cover - integration failure path
            error = f"{type(exc).__name__}: {exc}"
            response = ""
        critique = parse_critique(response, iteration=iteration, error=error)
        finished_at = utc_now()
        depends_on = [
            GENERATOR
            if current_version.created_by == GENERATOR
            else reflection_step_name(REVISER, current_version.iteration)
        ]
        step = make_reflection_step(
            step_id=step_id,
            component=CRITIC,
            iteration=iteration,
            actor="langgraph.critic_node",
            prompt=prompt,
            current_version=current_version.model_dump(),
            output={"critique": critique.model_dump()},
            llm_call_ids=[call_record.metrics.call_id] if call_record else [],
            started_at=started_at,
            finished_at=finished_at,
            framework_primitive=FRAMEWORK_PRIMITIVE,
            max_iterations=settings.max_iterations,
            depends_on=depends_on,
            error=error,
        )
        return {
            "iteration": iteration,
            "critiques": [*state["critiques"], critique],
            "steps": [*state["steps"], step],
            "llm_calls": [
                *state["llm_calls"],
                *([call_record.metrics] if call_record else []),
            ],
        }

    def stop_controller_node(state: ReflectionState) -> dict:
        critique = state["critiques"][-1]
        current_version = state["current_version"]
        started_at = utc_now()
        decision = evaluate_stop(
            critique,
            current_version_index=current_version.version_index,
            settings=settings,
        )
        finished_at = utc_now()
        step = make_reflection_step(
            step_id=len(state["steps"]) + 1,
            component=STOP_CONTROLLER,
            iteration=state["iteration"],
            actor="langgraph.stop_controller_node",
            current_version=current_version.model_dump(),
            critique=critique.model_dump(),
            output={"stop_decision": decision.model_dump()},
            started_at=started_at,
            finished_at=finished_at,
            framework_primitive=FRAMEWORK_PRIMITIVE,
            max_iterations=settings.max_iterations,
            depends_on=[reflection_step_name(CRITIC, state["iteration"])],
        )
        return {
            "should_stop": decision.should_stop,
            "stop_reason": decision.stop_reason,
            "stop_decisions": [*state["stop_decisions"], decision],
            "steps": [*state["steps"], step],
        }

    def reviser_node(state: ReflectionState) -> dict:
        iteration = state["iteration"]
        current_version = state["current_version"]
        critique = state["critiques"][-1]
        stop_decision = state["stop_decisions"][-1]
        prompt = render_reflection_prompt(
            input_data,
            REVISER,
            iteration=iteration,
            settings=settings,
            current_version=current_version.model_dump(),
            critique=critique.model_dump(),
        )
        step_id = len(state["steps"]) + 1
        started_at = utc_now()
        call_record = None
        error = None
        try:
            call_record = complete_llm_step(
                llm=context.llm,
                input_data=input_data,
                config=config,
                prompt=prompt,
                step_id=step_id,
            )
            response = call_record.response.strip()
        except Exception as exc:  # pragma: no cover - integration failure path
            error = f"{type(exc).__name__}: {exc}"
            response = ""
        version = parse_reflection_version(
            response,
            version_index=len(state["versions"]),
            iteration=iteration,
            created_by=REVISER,
            error=error,
        )
        finished_at = utc_now()
        step = make_reflection_step(
            step_id=step_id,
            component=REVISER,
            iteration=iteration,
            actor="langgraph.reviser_node",
            prompt=prompt,
            current_version=current_version.model_dump(),
            critique=critique.model_dump(),
            stop_decision=stop_decision.model_dump(),
            output={"version": version.model_dump()},
            llm_call_ids=[call_record.metrics.call_id] if call_record else [],
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
        return {
            "current_version": version,
            "versions": [*state["versions"], version],
            "steps": [*state["steps"], step],
            "llm_calls": [
                *state["llm_calls"],
                *([call_record.metrics] if call_record else []),
            ],
        }

    def route_after_stop(state: ReflectionState) -> str:
        return "finish" if state["should_stop"] else REVISER

    graph = StateGraph(ReflectionState)
    graph.add_node(GENERATOR, generator_node)
    graph.add_node(CRITIC, critic_node)
    graph.add_node(STOP_CONTROLLER, stop_controller_node)
    graph.add_node(REVISER, reviser_node)
    graph.add_edge(START, GENERATOR)
    graph.add_edge(GENERATOR, CRITIC)
    graph.add_edge(CRITIC, STOP_CONTROLLER)
    graph.add_conditional_edges(
        STOP_CONTROLLER,
        route_after_stop,
        {"finish": END, REVISER: REVISER},
    )
    graph.add_edge(REVISER, CRITIC)

    initial_state: ReflectionState = {
        "settings": settings,
        "versions": [],
        "critiques": [],
        "stop_decisions": [],
        "iteration": 0,
        "should_stop": False,
        "stop_reason": None,
        "steps": [],
        "llm_calls": [],
    }
    compiled = graph.compile()
    with ResourceMonitor() as monitor:
        state = compiled.invoke(
            initial_state,
            {"recursion_limit": settings.max_iterations * 3 + 5},
        )
        resource_usage = monitor.usage

    final_answer, structured_output = build_reflection_structured_output(
        input_data=input_data,
        config=config,
        versions=state["versions"],
        critiques=state["critiques"],
        stop_decisions=state["stop_decisions"],
        steps=state["steps"],
        llm_calls=state["llm_calls"],
        settings=settings,
        framework_execution="langgraph_native_conditional_reflection_cycle",
        framework_primitive=FRAMEWORK_PRIMITIVE,
    )
    return LangGraphRunOutput(
        final_answer=final_answer,
        structured_output=structured_output,
        steps=state["steps"],
        llm_calls=state["llm_calls"],
        resource_usage=resource_usage,
    )
