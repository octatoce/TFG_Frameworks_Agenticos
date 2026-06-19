"""LlamaIndex implementation for ARCH_01_SINGLE_REACT."""

from __future__ import annotations

from benchmark_core.llm_wrapper import render_single_react_prompt
from benchmark_core.schemas import AgentStep, ExperimentConfig, ExperimentInput
from benchmark_core.tracing import utc_now
from implementations.llamaindex.utils_llamaindex import (
    LlamaIndexRunContext,
    LlamaIndexRunOutput,
    build_function_agent,
    complete_agent_step,
    document_ids,
    extract_final_answer,
    framework_execution,
    llamaindex_architecture_runner,
    run_with_resource_monitor,
)


@llamaindex_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: LlamaIndexRunContext,
) -> LlamaIndexRunOutput:
    """Execute a single LlamaIndex FunctionAgent-style agent."""

    def execute() -> LlamaIndexRunOutput:
        prompt = render_single_react_prompt(input_data)
        agent = build_function_agent(
            name="single_react_agent",
            system_prompt="You are a single LlamaIndex FunctionAgent for a benchmark run.",
            context=context,
            input_data=input_data,
            config=config,
        )
        step_started_at = utc_now()
        call_record = complete_agent_step(
            agent=agent,
            prompt=prompt,
            input_data=input_data,
            config=config,
            step_id=1,
        )
        final_answer = extract_final_answer(call_record.response)
        structured_output = {
            "answer": final_answer,
            "mode": f"{config.model_provider}_react",
            "document_ids": document_ids(input_data),
            "framework_execution": framework_execution("agent_workflow_single_agent", context),
        }
        step = AgentStep(
            step_id=1,
            name="single_react_agent",
            step_type="agent_llm_call",
            actor="llamaindex.function_agent",
            input_data={"prompt": prompt},
            output_data=structured_output,
            llm_call_ids=[call_record.metrics.call_id],
            started_at=step_started_at,
            finished_at=utc_now(),
            metadata={
                "agent_name": getattr(agent, "name", "single_react_agent"),
                "agent_workflow_shape": "single_function_agent",
                "native_framework_available": context.native_framework_available,
            },
        )
        return LlamaIndexRunOutput(
            final_answer=final_answer,
            structured_output=structured_output,
            steps=[step],
            llm_calls=[call_record.metrics],
        )

    return run_with_resource_monitor(execute)
