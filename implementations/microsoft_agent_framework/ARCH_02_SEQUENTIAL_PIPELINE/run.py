"""Microsoft Agent Framework implementation for ARCH_02_SEQUENTIAL_PIPELINE."""

from __future__ import annotations

from benchmark_core.llm_wrapper import render_sequential_pipeline_prompt
from benchmark_core.schemas import AgentStep, ExperimentConfig, ExperimentInput
from benchmark_core.tracing import utc_now
from implementations.microsoft_agent_framework.utils_microsoft_agent_framework import (
    MicrosoftAgentFrameworkRunContext,
    MicrosoftAgentFrameworkRunOutput,
    build_agent,
    complete_agent_step,
    document_ids,
    extract_final_answer,
    framework_execution,
    microsoft_agent_framework_architecture_runner,
    next_step_id,
    run_with_resource_monitor,
)


PHASES = [
    ("planner", "plan"),
    ("retriever", "evidence"),
    ("analyst", "analysis"),
    ("writer", "final_output"),
]


@microsoft_agent_framework_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: MicrosoftAgentFrameworkRunContext,
) -> MicrosoftAgentFrameworkRunOutput:
    """Execute a four-step Microsoft Agent Framework-style functional workflow."""

    def execute() -> MicrosoftAgentFrameworkRunOutput:
        agent = build_agent(
            name="sequential_pipeline_workflow",
            instructions=(
                "You are executing one phase of a controlled sequential benchmark workflow. "
                "Do not route dynamically or call other agents."
            ),
            context=context,
            input_data=input_data,
            config=config,
        )
        state: dict[str, object] = {
            "query": input_data.query,
            "documents": list(input_data.documents),
            "plan": None,
            "evidence": [],
            "analysis": None,
            "final_output": None,
            "steps": [],
            "llm_calls": [],
        }

        for phase, output_key in PHASES:
            step_id = next_step_id(state)
            step_started_at = utc_now()
            prompt = render_sequential_pipeline_prompt(input_data, phase=phase, state=state)
            call_record = complete_agent_step(
                agent=agent,
                prompt=prompt,
                input_data=input_data,
                config=config,
                step_id=step_id,
            )
            phase_output = call_record.response.strip()
            state[output_key] = phase_output
            step = AgentStep(
                step_id=step_id,
                name=phase,
                step_type="pipeline_phase_llm_call",
                actor=f"microsoft_agent_framework.{phase}_step",
                input_data={"phase": phase, "prompt": prompt},
                output_data={"phase_output": phase_output},
                llm_call_ids=[call_record.metrics.call_id],
                started_at=step_started_at,
                finished_at=utc_now(),
                metadata={
                    "pipeline_order": step_id,
                    "workflow_api": "functional_workflow_adapter",
                    "native_framework_available": context.native_framework_available,
                },
            )
            state["steps"] = [*state["steps"], step]
            state["llm_calls"] = [*state["llm_calls"], call_record.metrics]

        final_answer = extract_final_answer(str(state["final_output"]))
        structured_output = {
            "answer": final_answer,
            "mode": f"{config.model_provider}_sequential_pipeline",
            "plan": state["plan"],
            "evidence": state["evidence"],
            "analysis": state["analysis"],
            "document_ids": document_ids(input_data),
            "framework_execution": framework_execution("functional_workflow", context),
        }
        return MicrosoftAgentFrameworkRunOutput(
            final_answer=final_answer,
            structured_output=structured_output,
            steps=state["steps"],
            llm_calls=state["llm_calls"],
        )

    return run_with_resource_monitor(execute)
