from datetime import datetime, timezone

from benchmark_core.schemas import (
    AgentStep,
    DocumentInput,
    ExperimentConfig,
    ExperimentInput,
    ExperimentMetrics,
    ExperimentResult,
    ResourceUsage,
    RunStatus,
    TokenUsage,
)


def test_schemas_can_be_instantiated() -> None:
    input_data = ExperimentInput(
        case_id="case-001",
        dataset_id="samples",
        task_type="qa",
        query="Summarize the document.",
        documents=[
            DocumentInput(
                document_id="doc-001",
                content="Example document",
            )
        ],
        metadata={"source": "test"},
    )
    config = ExperimentConfig(
        experiment_id="schema-test",
        framework="langgraph",
        architecture="ARCH_01_SINGLE_REACT",
        model_provider="local",
        model_name="deterministic-local-v1",
        run_id="test-run",
    )
    step = AgentStep(
        step_id=1,
        name="test-step",
        step_type="agent_llm_call",
        llm_call_ids=["call-001"],
    )
    token_usage = TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15)
    resource_usage = ResourceUsage(memory_mb_start=100.0, memory_mb_end=101.0)
    metrics = ExperimentMetrics(
        total_latency_seconds=0.1,
        step_count=1,
        llm_call_count=1,
        token_usage=token_usage,
        resource_usage=resource_usage,
    )
    now = datetime.now(timezone.utc)

    result = ExperimentResult(
        case_id=input_data.case_id,
        dataset_id=input_data.dataset_id,
        framework=config.framework,
        architecture=config.architecture,
        run_id=config.run_id,
        status=RunStatus.SUCCESS,
        final_answer="The benchmark document was summarized successfully.",
        input_snapshot=input_data,
        config_snapshot=config,
        metrics=metrics,
        steps=[step],
        errors=[],
        started_at=now,
        finished_at=now,
    )

    assert result.case_id == "case-001"
    assert result.metrics.token_usage.total_tokens == 15
