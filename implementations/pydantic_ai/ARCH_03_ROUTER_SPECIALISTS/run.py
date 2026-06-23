"""Pydantic AI implementation for ARCH_03_ROUTER_SPECIALISTS."""

from __future__ import annotations

from benchmark_core.llm_wrapper import parse_specialist_selection, render_router_specialists_prompt
from benchmark_core.schemas import AgentStep, ExperimentConfig, ExperimentInput
from benchmark_core.tracing import utc_now
from implementations.pydantic_ai.utils_pydantic_ai import (
    PydanticAIRunContext,
    PydanticAIRunOutput,
    PydanticAIRouterOutput,
    build_typed_agent,
    complete_agent_step,
    document_ids,
    extract_final_answer,
    framework_execution,
    next_step_id,
    pydantic_ai_architecture_runner,
    run_with_resource_monitor,
)


SPECIALISTS = ["data_specialist", "reasoning_specialist", "validation_specialist"]
OUTPUT_KEYS = {
    "router_routing": "router_plan",
    "data_specialist": "evidence",
    "reasoning_specialist": "preliminary_decision",
    "validation_specialist": "validation_report",
    "router_synthesis": "final_output",
}


@pydantic_ai_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: PydanticAIRunContext,
) -> PydanticAIRunOutput:
    """Execute a typed router-specialists flow shaped for pydantic-graph."""

    def execute() -> PydanticAIRunOutput:
        agent = build_typed_agent(
            name="router_specialists_graph",
            instructions=(
                "You are a typed Pydantic AI router graph component. "
                "Select specialists once, execute only the requested phase, and do not request revisions."
            ),
            context=context,
            input_data=input_data,
            config=config,
        )
        state: dict[str, object] = {
            "query": input_data.query,
            "documents": list(input_data.documents),
            "selected_specialists": [],
            "skipped_specialists": [],
            "router_plan": None,
            "evidence": [],
            "preliminary_decision": None,
            "validation_report": None,
            "final_output": None,
            "steps": [],
            "llm_calls": [],
        }

        def run_phase(phase: str) -> None:
            step_id = next_step_id(state)
            step_started_at = utc_now()
            prompt = render_router_specialists_prompt(input_data, phase=phase, state=state)
            call_record = complete_agent_step(
                agent=agent,
                prompt=prompt,
                input_data=input_data,
                config=config,
                step_id=step_id,
            )
            phase_output = call_record.response.strip()

            if phase == "router_routing":
                selected_specialists, skipped_specialists = parse_specialist_selection(phase_output, SPECIALISTS)
                state["selected_specialists"] = selected_specialists
                state["skipped_specialists"] = skipped_specialists

            state[OUTPUT_KEYS[phase]] = phase_output
            step = AgentStep(
                step_id=step_id,
                name=phase,
                step_type="router_specialist_llm_call",
                actor=f"pydantic_ai.graph.{phase}",
                input_data={"phase": phase, "prompt": prompt},
                output_data={
                    "phase_output": phase_output,
                    "selected_specialists": state["selected_specialists"],
                    "skipped_specialists": state["skipped_specialists"],
                },
                llm_call_ids=[call_record.metrics.call_id],
                started_at=step_started_at,
                finished_at=utc_now(),
                metadata={
                    "pipeline_order": step_id,
                    "specialist_role": phase if phase in SPECIALISTS else None,
                    "graph_shape": "router_controlled_state_machine",
                    "native_framework_available": context.native_framework_available,
                    "native_graph_available": context.native_graph_available,
                },
            )
            state["steps"] = [*state["steps"], step]
            state["llm_calls"] = [*state["llm_calls"], call_record.metrics]

        run_phase("router_routing")
        for specialist in SPECIALISTS:
            if specialist in state["selected_specialists"]:
                run_phase(specialist)
        run_phase("router_synthesis")

        final_answer = extract_final_answer(str(state["final_output"]))
        structured_model = PydanticAIRouterOutput(
            answer=final_answer,
            mode=f"{config.model_provider}_router_specialists",
            selected_specialists=list(state["selected_specialists"]),
            skipped_specialists=list(state["skipped_specialists"]),
            evidence=state["evidence"],
            preliminary_decision=state["preliminary_decision"],
            validation_report=state["validation_report"],
            document_ids=document_ids(input_data),
            framework_execution=framework_execution("router_specialists", context),
        )
        structured_output = structured_model.model_dump()
        return PydanticAIRunOutput(
            final_answer=final_answer,
            structured_output=structured_output,
            steps=state["steps"],
            llm_calls=state["llm_calls"],
        )

    return run_with_resource_monitor(execute)

