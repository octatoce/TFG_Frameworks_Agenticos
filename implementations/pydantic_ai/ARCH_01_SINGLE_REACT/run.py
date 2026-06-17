"""Pydantic AI implementation for ARCH_01_SINGLE_REACT."""

from __future__ import annotations

from benchmark_core.llm_wrapper import render_single_react_prompt
from benchmark_core.schemas import AgentStep, ExperimentConfig, ExperimentInput
from benchmark_core.tracing import utc_now
from implementations.pydantic_ai.utils_pydantic_ai import (
    DeterministicPydanticAgent,
    PydanticAIRunContext,
    PydanticAIRunOutput,
    PydanticAIStructuredOutput,
    document_ids,
    extract_final_answer,
    framework_execution,
    pydantic_ai_architecture_runner,
    run_with_resource_monitor,
)


@pydantic_ai_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: PydanticAIRunContext,
) -> PydanticAIRunOutput:
    """Execute a single typed Pydantic AI-style agent."""

    def execute() -> PydanticAIRunOutput:
        prompt = render_single_react_prompt(input_data)
        agent = DeterministicPydanticAgent(
            name="single_react_agent",
            llm=context.llm,
            input_data=input_data,
            config=config,
        )
        step_started_at = utc_now()
        call_record = agent.run_sync(prompt, step_id=1)
        final_answer = extract_final_answer(call_record.response)
        structured_model = PydanticAIStructuredOutput(
            answer=final_answer,
            mode=f"{config.model_provider}_react",
            document_ids=document_ids(input_data),
            framework_execution=framework_execution("typed_single_agent", context),
        )
        structured_output = structured_model.model_dump()
        step = AgentStep(
            step_id=1,
            name="single_react_agent",
            step_type="agent_llm_call",
            actor="pydantic_ai.typed_agent",
            input_data={"prompt": prompt},
            output_data=structured_output,
            llm_call_ids=[call_record.metrics.call_id],
            started_at=step_started_at,
            finished_at=utc_now(),
            metadata={
                "agent_name": agent.name,
                "output_model": "PydanticAIStructuredOutput",
                "native_framework_available": context.native_framework_available,
            },
        )
        return PydanticAIRunOutput(
            final_answer=final_answer,
            structured_output=structured_output,
            steps=[step],
            llm_calls=[call_record.metrics],
        )

    return run_with_resource_monitor(execute)

