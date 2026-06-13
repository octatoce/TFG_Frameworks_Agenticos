"""CrewAI implementation for ARCH_02_SEQUENTIAL_PIPELINE."""

from __future__ import annotations

from benchmark_core.llm_wrapper import render_sequential_pipeline_prompt
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
    kickoff_with_resource_monitor,
)


PHASES = ["planner", "retriever", "analyst", "writer"]

AGENT_DEFINITIONS = {
    "planner": {
        "role": "Pipeline planner",
        "goal": "Produce a concise plan for solving the benchmark case.",
        "backstory": "A controlled planning component in a sequential benchmark pipeline.",
    },
    "retriever": {
        "role": "Pipeline retriever",
        "goal": "Select the evidence needed to answer the benchmark case.",
        "backstory": "A controlled retrieval component in a sequential benchmark pipeline.",
    },
    "analyst": {
        "role": "Pipeline analyst",
        "goal": "Analyze the selected evidence and produce a concise conclusion.",
        "backstory": "A controlled analysis component in a sequential benchmark pipeline.",
    },
    "writer": {
        "role": "Pipeline writer",
        "goal": "Produce the final structured answer for the benchmark case.",
        "backstory": "A controlled writing component in a sequential benchmark pipeline.",
    },
}


@crewai_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: CrewAIRunContext,
) -> CrewAIRunOutput:
    """Execute a four-task sequential pipeline through CrewAI."""

    agents = {
        phase: create_agent(
            **definition,
            crewai_llm=context.crewai_llm,
            config=config,
        )
        for phase, definition in AGENT_DEFINITIONS.items()
    }

    initial_prompt_state: dict[str, object] = {
        "plan": None,
        "evidence": [],
        "analysis": None,
    }
    tasks = []
    for phase in PHASES:
        context_tasks = tasks[-1:] if tasks else None
        tasks.append(
            create_task(
                description=render_sequential_pipeline_prompt(
                    input_data=input_data,
                    phase=phase,
                    state=initial_prompt_state,
                ),
                expected_output=f"Concise output for the {phase} phase.",
                agent=agents[phase],
                context=context_tasks,
                config=config,
            )
        )

    crew = create_sequential_crew(agents=list(agents.values()), tasks=tasks)
    step_started_at = utc_now()
    _, resource_usage = kickoff_with_resource_monitor(crew)

    phase_outputs = {
        phase: str(task.output).strip() if task.output is not None else ""
        for phase, task in zip(PHASES, tasks, strict=True)
    }
    final_answer = extract_final_answer(phase_outputs["writer"])
    llm_calls = get_llm_call_metrics(context.crewai_llm)
    steps = []
    for index, phase in enumerate(PHASES, start=1):
        call_id = llm_calls[index - 1].call_id if len(llm_calls) >= index else None
        steps.append(
            AgentStep(
                step_id=index,
                name=phase,
                step_type="pipeline_phase_llm_call",
                actor=f"crewai.{phase}_agent",
                input_data={
                    "phase": phase,
                    "prompt": tasks[index - 1].description,
                },
                output_data={
                    "phase_output": phase_outputs[phase],
                },
                llm_call_ids=[call_id] if call_id is not None else [],
                started_at=step_started_at,
                finished_at=utc_now(),
                metadata={
                    "pipeline_order": index,
                    "crew_process": crew_process_value(),
                    "agent_role": agents[phase].role,
                },
            )
        )

    structured_output = {
        "answer": final_answer,
        "mode": f"{config.model_provider}_sequential_pipeline",
        "plan": phase_outputs["planner"],
        "evidence": phase_outputs["retriever"],
        "analysis": phase_outputs["analyst"],
        "document_ids": document_ids(input_data),
        "framework_execution": "crewai_four_task_sequence",
    }

    return CrewAIRunOutput(
        final_answer=final_answer,
        structured_output=structured_output,
        steps=steps,
        resource_usage=resource_usage,
    )
