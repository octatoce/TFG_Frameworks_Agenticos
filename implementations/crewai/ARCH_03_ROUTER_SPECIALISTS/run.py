"""CrewAI implementation for ARCH_03_ROUTER_SPECIALISTS."""

from __future__ import annotations

from benchmark_core.llm_wrapper import parse_specialist_selection, render_router_specialists_prompt
from benchmark_core.resource_monitor import ResourceMonitor
from benchmark_core.schemas import AgentStep, ExperimentConfig, ExperimentInput
from benchmark_core.tracing import utc_now
from implementations.crewai.utils_crewai import (
    CrewAIRunContext,
    CrewAIRunOutput,
    create_agent,
    create_sequential_crew,
    create_task,
    crew_process_value,
    crewai_architecture_runner,
    document_ids,
    extract_final_answer,
    get_llm_call_metrics,
    unique_agents,
)


SPECIALISTS = ["data_specialist", "reasoning_specialist", "validation_specialist"]

AGENT_DEFINITIONS = {
    "data_specialist": {
        "role": "Data Specialist",
        "goal": "Recover and summarize documentary evidence.",
        "backstory": "A controlled data specialist in a router-specialists benchmark.",
    },
    "reasoning_specialist": {
        "role": "Reasoning Specialist",
        "goal": "Analyze evidence and propose a preliminary decision.",
        "backstory": "A controlled reasoning specialist in a router-specialists benchmark.",
    },
    "validation_specialist": {
        "role": "Validation Specialist",
        "goal": "Validate consistency, limitations, and confidence.",
        "backstory": "A controlled validation specialist in a router-specialists benchmark.",
    },
}


@crewai_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: CrewAIRunContext,
) -> CrewAIRunOutput:
    """Execute a router-specialists crew with controlled sequential tasks."""

    router = create_agent(
        role="Router",
        goal="Select which specialists execute and synthesize the final benchmark answer.",
        backstory="A controlled router selecting specialist agents for comparison.",
        crewai_llm=context.crewai_llm,
        config=config,
    )
    agents = {
        "router_routing": router,
        "router_synthesis": router,
    }
    for specialist, definition in AGENT_DEFINITIONS.items():
        agents[specialist] = create_agent(
            **definition,
            crewai_llm=context.crewai_llm,
            config=config,
        )

    prompt_state: dict[str, object] = {
        "selected_specialists": [],
        "skipped_specialists": [],
        "evidence": [],
        "preliminary_decision": None,
        "validation_report": None,
    }

    step_started_at = utc_now()
    with ResourceMonitor() as monitor:
        planning_task = create_task(
            description=render_router_specialists_prompt(
                input_data=input_data,
                phase="router_routing",
                state=prompt_state,
            ),
            expected_output="Selected and skipped specialists in the required format.",
            agent=router,
            config=config,
        )
        planning_crew = create_sequential_crew(agents=[router], tasks=[planning_task])
        planning_crew.kickoff()
        router_plan = str(planning_task.output).strip() if planning_task.output is not None else ""
        selected_specialists, skipped_specialists = parse_specialist_selection(router_plan, SPECIALISTS)

        prompt_state["selected_specialists"] = selected_specialists
        prompt_state["skipped_specialists"] = skipped_specialists

        executed_phases = ["router_routing", *selected_specialists, "router_synthesis"]
        tasks = [planning_task]
        specialist_tasks = []
        for phase in [*selected_specialists, "router_synthesis"]:
            context_tasks = tasks[-1:] if tasks else None
            task = create_task(
                description=render_router_specialists_prompt(
                    input_data=input_data,
                    phase=phase,
                    state=prompt_state,
                ),
                expected_output=f"Concise output for {phase}.",
                agent=agents[phase],
                context=context_tasks,
                config=config,
            )
            specialist_tasks.append(task)
            tasks.append(task)

        execution_crew = create_sequential_crew(
            agents=unique_agents([agents[phase] for phase in executed_phases]),
            tasks=specialist_tasks,
        )
        execution_crew.kickoff()
        resource_usage = monitor.usage

    phase_outputs = {
        phase: str(task.output).strip() if task.output is not None else ""
        for phase, task in zip(executed_phases, tasks, strict=True)
    }
    final_answer = extract_final_answer(phase_outputs["router_synthesis"])
    llm_calls = get_llm_call_metrics(context.crewai_llm)
    steps = []
    for index, phase in enumerate(executed_phases, start=1):
        call_id = llm_calls[index - 1].call_id if len(llm_calls) >= index else None
        steps.append(
            AgentStep(
                step_id=index,
                name=phase,
                step_type="router_specialist_llm_call",
                actor=f"crewai.{phase}",
                input_data={
                    "phase": phase,
                    "prompt": tasks[index - 1].description,
                },
                output_data={
                    "phase_output": phase_outputs[phase],
                    "selected_specialists": selected_specialists,
                    "skipped_specialists": skipped_specialists,
                },
                llm_call_ids=[call_id] if call_id is not None else [],
                started_at=step_started_at,
                finished_at=utc_now(),
                metadata={
                    "pipeline_order": index,
                    "crew_process": crew_process_value(),
                    "agent_role": agents[phase].role,
                    "specialist_role": phase if phase in SPECIALISTS else None,
                },
            )
        )

    structured_output = {
        "answer": final_answer,
        "mode": f"{config.model_provider}_router_specialists",
        "selected_specialists": selected_specialists,
        "skipped_specialists": skipped_specialists,
        "evidence": phase_outputs.get("data_specialist", []),
        "preliminary_decision": phase_outputs.get("reasoning_specialist"),
        "validation_report": phase_outputs.get("validation_specialist"),
        "document_ids": document_ids(input_data),
        "framework_execution": "crewai_router_specialists_sequence",
    }

    return CrewAIRunOutput(
        final_answer=final_answer,
        structured_output=structured_output,
        steps=steps,
        resource_usage=resource_usage,
    )

