import importlib.util
import inspect
from pathlib import Path

from benchmark_core.schemas import (
    DocumentInput,
    ExperimentConfig,
    ExperimentInput,
    ExperimentResult,
    RunStatus,
)


EXPECTED_RUNNERS = [
    Path("implementations/langgraph/ARCH_01_SINGLE_REACT/run.py"),
    Path("implementations/langgraph/ARCH_02_SEQUENTIAL_PIPELINE/run.py"),
    Path("implementations/langgraph/ARCH_03_SUPERVISOR_WORKERS/run.py"),
    Path("implementations/crewai/ARCH_01_SINGLE_REACT/run.py"),
    Path("implementations/crewai/ARCH_02_SEQUENTIAL_PIPELINE/run.py"),
    Path("implementations/crewai/ARCH_03_SUPERVISOR_WORKERS/run.py"),
    Path("implementations/microsoft_agent_framework/ARCH_01_SINGLE_REACT/run.py"),
    Path("implementations/microsoft_agent_framework/ARCH_02_SEQUENTIAL_PIPELINE/run.py"),
    Path("implementations/microsoft_agent_framework/ARCH_03_SUPERVISOR_WORKERS/run.py"),
    Path("implementations/llamaindex/ARCH_01_SINGLE_REACT/run.py"),
    Path("implementations/llamaindex/ARCH_02_SEQUENTIAL_PIPELINE/run.py"),
    Path("implementations/llamaindex/ARCH_03_SUPERVISOR_WORKERS/run.py"),
    Path("implementations/pydantic_ai/ARCH_01_SINGLE_REACT/run.py"),
    Path("implementations/pydantic_ai/ARCH_02_SEQUENTIAL_PIPELINE/run.py"),
    Path("implementations/pydantic_ai/ARCH_03_SUPERVISOR_WORKERS/run.py"),
]


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_architecture_runners_expose_common_function() -> None:
    root = Path(__file__).resolve().parents[1]

    for relative_path in EXPECTED_RUNNERS:
        module = load_module(root / relative_path)
        assert hasattr(module, "run_architecture")
        assert callable(module.run_architecture)

        signature = inspect.signature(module.run_architecture)
        assert list(signature.parameters) == ["input_data", "config"]


def test_architecture_runners_return_experiment_result() -> None:
    root = Path(__file__).resolve().parents[1]

    for relative_path in EXPECTED_RUNNERS:
        framework = relative_path.parts[1]
        architecture = relative_path.parts[2]
        module = load_module(root / relative_path)

        input_data = ExperimentInput(
            case_id="case-001",
            dataset_id="samples",
            task_type="qa",
            query="Summarize the benchmark document.",
            documents=[
                DocumentInput(
                    document_id="doc-001",
                    content="This is a small benchmark document.",
                )
            ],
            metadata={},
        )
        config = ExperimentConfig(
            experiment_id="contract-test",
            framework=framework,
            architecture=architecture,
            model_provider="local",
            model_name="deterministic-local-v1",
            run_id="contract-test",
        )

        result = module.run_architecture(input_data, config)

        assert isinstance(result, ExperimentResult)
        assert result.input_snapshot.case_id == input_data.case_id
        assert result.config_snapshot.framework == framework

        if architecture == "ARCH_01_SINGLE_REACT":
            assert result.status == RunStatus.SUCCESS
            assert result.metrics.step_count == 1
            assert result.metrics.llm_call_count >= 1
            assert result.final_answer
        elif architecture == "ARCH_02_SEQUENTIAL_PIPELINE":
            assert result.status == RunStatus.SUCCESS
            assert result.metrics.step_count == 4
            assert result.metrics.llm_call_count == 4
            assert [step.name for step in result.steps] == [
                "planner",
                "retriever",
                "analyst",
                "writer",
            ]
            assert result.final_answer
        elif architecture == "ARCH_03_SUPERVISOR_WORKERS":
            assert result.status == RunStatus.SUCCESS
            selected_workers = result.structured_output["selected_workers"]
            skipped_workers = result.structured_output["skipped_workers"]
            expected_steps = [
                "supervisor_planning",
                *selected_workers,
                "supervisor_synthesis",
            ]
            assert result.metrics.step_count == len(expected_steps)
            assert result.metrics.llm_call_count == len(expected_steps)
            assert [step.name for step in result.steps] == expected_steps
            assert "reasoning_worker" in selected_workers
            assert sorted(selected_workers + skipped_workers) == [
                "data_worker",
                "reasoning_worker",
                "validation_worker",
            ]
            assert result.final_answer
        else:
            raise AssertionError(f"Unexpected architecture under test: {architecture}")
