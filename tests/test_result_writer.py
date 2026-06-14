import json
from datetime import datetime, timezone

from benchmark_core.result_writer import save_result_json
from benchmark_core.schemas import ExperimentConfig, ExperimentInput, ExperimentResult, RunStatus


def test_result_writer_saves_valid_json(tmp_path) -> None:
    now = datetime.now(timezone.utc)
    input_data = ExperimentInput(
        case_id="case-001",
        dataset_id="samples",
        task_type="qa",
        query="Summarize the benchmark document.",
    )
    config = ExperimentConfig(
        experiment_id="writer-test",
        framework="langgraph",
        architecture="ARCH_01_SINGLE_REACT",
        model_provider="local",
        model_name="deterministic-local-v1",
        run_id="run-001",
    )
    result = ExperimentResult(
        case_id=input_data.case_id,
        dataset_id=input_data.dataset_id,
        framework=config.framework,
        architecture=config.architecture,
        run_id=config.run_id,
        status=RunStatus.SUCCESS,
        final_answer="The benchmark document was summarized successfully.",
        structured_output={"answer": "The benchmark document was summarized successfully."},
        input_snapshot=input_data,
        config_snapshot=config,
        started_at=now,
        finished_at=now,
    )

    output_path = save_result_json(result, base_dir=tmp_path)

    assert output_path == tmp_path / "langgraph" / "ARCH_01_SINGLE_REACT" / "run-001.json"
    assert output_path.exists()

    with output_path.open(encoding="utf-8") as file:
        data = json.load(file)

    assert data["case_id"] == "case-001"
    assert data["framework"] == "langgraph"
    assert data["architecture"] == "ARCH_01_SINGLE_REACT"
    assert data["run_id"] == "run-001"
