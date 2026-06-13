"""CrewAI implementation for ARCH_01_SINGLE_REACT."""

from __future__ import annotations

from benchmark_core.llm_wrapper import render_single_react_prompt
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
    get_llm_call_metrics,
    kickoff_with_resource_monitor,
)


@crewai_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: CrewAIRunContext,
) -> CrewAIRunOutput:
    """Execute a single CrewAI agent with an instrumented deterministic LLM."""

    task_prompt = render_single_react_prompt(input_data)
    agent = create_agent(
        role="Single ReAct benchmark agent",
        goal="Answer the benchmark case using one deterministic ReAct-style reasoning step.",
        backstory="A controlled benchmark agent used for framework comparison.",
        crewai_llm=context.crewai_llm,
        config=config,
    )
    task = create_task(
        description=task_prompt,
        expected_output="A concise final answer grounded in the provided input.",
        agent=agent,
        config=config,
    )
    crew = create_sequential_crew(agents=[agent], tasks=[task])

    step_started_at = utc_now()
    crew_output, resource_usage = kickoff_with_resource_monitor(crew)
    final_answer = str(crew_output).strip()
    llm_calls = get_llm_call_metrics(context.crewai_llm)
    structured_output = {
        "answer": final_answer,
        "mode": f"{config.model_provider}_react",
        "document_ids": document_ids(input_data),
        "framework_execution": "crewai_agent_task_crew",
    }
    steps = [
        AgentStep(
            step_id=1,
            name="single_react_agent",
            step_type="agent_llm_call",
            actor="crewai.single_agent",
            input_data={"prompt": task_prompt},
            output_data=structured_output,
            llm_call_ids=[call.call_id for call in llm_calls],
            started_at=step_started_at,
            finished_at=utc_now(),
            metadata={
                "crew_process": crew_process_value(),
                "agent_role": agent.role,
            },
        )
    ]

    return CrewAIRunOutput(
        final_answer=final_answer,
        structured_output=structured_output,
        steps=steps,
        resource_usage=resource_usage,
    )
