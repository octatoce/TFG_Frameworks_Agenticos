"""CrewAI implementation for ARCH_03_SUPERVISOR_WORKERS."""

from __future__ import annotations

from benchmark_core.llm_wrapper import parse_worker_selection, render_supervisor_workers_prompt
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


WORKERS = ["data_worker", "reasoning_worker", "validation_worker"]

AGENT_DEFINITIONS = {
    "data_worker": {
        "role": "Data Worker",
        "goal": "Recover and summarize documentary evidence.",
        "backstory": "A controlled data worker in a supervisor-workers benchmark.",
    },
    "reasoning_worker": {
        "role": "Reasoning Worker",
        "goal": "Analyze evidence and propose a preliminary decision.",
        "backstory": "A controlled reasoning worker in a supervisor-workers benchmark.",
    },
    "validation_worker": {
        "role": "Validation Worker",
        "goal": "Validate consistency, limitations, and confidence.",
        "backstory": "A controlled validation worker in a supervisor-workers benchmark.",
    },
}


@crewai_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: CrewAIRunContext,
) -> CrewAIRunOutput:
    """Execute a supervisor-workers crew with controlled sequential tasks."""

    supervisor = create_agent(
        role="Supervisor",
        goal="Plan worker execution and synthesize the final benchmark answer.",
        backstory="A controlled supervisor coordinating specialist workers for comparison.",
        crewai_llm=context.crewai_llm,
        config=config,
    )
    agents = {
        "supervisor_planning": supervisor,
        "supervisor_synthesis": supervisor,
    }
    for worker, definition in AGENT_DEFINITIONS.items():
        agents[worker] = create_agent(
            **definition,
            crewai_llm=context.crewai_llm,
            config=config,
        )

    prompt_state: dict[str, object] = {
        "selected_workers": [],
        "skipped_workers": [],
        "evidence": [],
        "preliminary_decision": None,
        "validation_report": None,
    }

    step_started_at = utc_now()
    with ResourceMonitor() as monitor:
        planning_task = create_task(
            description=render_supervisor_workers_prompt(
                input_data=input_data,
                phase="supervisor_planning",
                state=prompt_state,
            ),
            expected_output="Selected and skipped workers in the required format.",
            agent=supervisor,
            config=config,
        )
        planning_crew = create_sequential_crew(agents=[supervisor], tasks=[planning_task])
        planning_crew.kickoff()
        supervisor_plan = str(planning_task.output).strip() if planning_task.output is not None else ""
        selected_workers, skipped_workers = parse_worker_selection(supervisor_plan, WORKERS)

        prompt_state["selected_workers"] = selected_workers
        prompt_state["skipped_workers"] = skipped_workers

        executed_phases = ["supervisor_planning", *selected_workers, "supervisor_synthesis"]
        tasks = [planning_task]
        worker_tasks = []
        for phase in [*selected_workers, "supervisor_synthesis"]:
            context_tasks = tasks[-1:] if tasks else None
            task = create_task(
                description=render_supervisor_workers_prompt(
                    input_data=input_data,
                    phase=phase,
                    state=prompt_state,
                ),
                expected_output=f"Concise output for {phase}.",
                agent=agents[phase],
                context=context_tasks,
                config=config,
            )
            worker_tasks.append(task)
            tasks.append(task)

        execution_crew = create_sequential_crew(
            agents=unique_agents([agents[phase] for phase in executed_phases]),
            tasks=worker_tasks,
        )
        execution_crew.kickoff()
        resource_usage = monitor.usage

    phase_outputs = {
        phase: str(task.output).strip() if task.output is not None else ""
        for phase, task in zip(executed_phases, tasks, strict=True)
    }
    final_answer = extract_final_answer(phase_outputs["supervisor_synthesis"])
    llm_calls = get_llm_call_metrics(context.crewai_llm)
    steps = []
    for index, phase in enumerate(executed_phases, start=1):
        call_id = llm_calls[index - 1].call_id if len(llm_calls) >= index else None
        steps.append(
            AgentStep(
                step_id=index,
                name=phase,
                step_type="supervisor_worker_llm_call",
                actor=f"crewai.{phase}",
                input_data={
                    "phase": phase,
                    "prompt": tasks[index - 1].description,
                },
                output_data={
                    "phase_output": phase_outputs[phase],
                    "selected_workers": selected_workers,
                    "skipped_workers": skipped_workers,
                },
                llm_call_ids=[call_id] if call_id is not None else [],
                started_at=step_started_at,
                finished_at=utc_now(),
                metadata={
                    "pipeline_order": index,
                    "crew_process": crew_process_value(),
                    "agent_role": agents[phase].role,
                    "worker_role": phase if phase in WORKERS else None,
                },
            )
        )

    structured_output = {
        "answer": final_answer,
        "mode": f"{config.model_provider}_supervisor_workers",
        "selected_workers": selected_workers,
        "skipped_workers": skipped_workers,
        "evidence": phase_outputs.get("data_worker", []),
        "preliminary_decision": phase_outputs.get("reasoning_worker"),
        "validation_report": phase_outputs.get("validation_worker"),
        "document_ids": document_ids(input_data),
        "framework_execution": "crewai_supervisor_workers_sequence",
    }

    return CrewAIRunOutput(
        final_answer=final_answer,
        structured_output=structured_output,
        steps=steps,
        resource_usage=resource_usage,
    )
