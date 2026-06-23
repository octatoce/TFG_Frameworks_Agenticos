"""Run the shared ARCH_03 smoke case for all implemented frameworks."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from benchmark_core.result_writer import save_result_json
from benchmark_core.schemas import DocumentInput, ExperimentConfig, ExperimentInput


ARCHITECTURE = "ARCH_03_ROUTER_SPECIALISTS"
FRAMEWORKS = [
    "langgraph",
    "crewai",
    "microsoft_agent_framework",
    "llamaindex",
    "pydantic_ai",
]


def load_runner(repo_root: Path, framework: str):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    module_path = repo_root / "implementations" / framework / ARCHITECTURE / "run.py"
    spec = importlib.util.spec_from_file_location(f"{framework}_{ARCHITECTURE}_run", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load runner from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_architecture


def build_smoke_case() -> ExperimentInput:
    return ExperimentInput(
        case_id="arch03-smoke-001",
        dataset_id="samples",
        task_type="document_qa",
        query=(
            "Resume el objetivo principal del documento y menciona que criterio "
            "de evaluacion se valida."
        ),
        documents=[
            DocumentInput(
                document_id="sample-doc-001",
                content=(
                    "El TFG compara frameworks agenticos modernos mediante prototipos equivalentes. "
                    "La primera iteracion valida estructura del repositorio, schemas comunes, "
                    "recogida de metricas, ejecucion comparable y persistencia JSON de resultados raw."
                ),
                metadata={"source": "synthetic_arch03_smoke_case"},
            )
        ],
        evaluation_criteria={
            "expected_contains": ["frameworks agenticos", "schemas", "metricas", "JSON"]
        },
        metadata={"purpose": "arch03_smoke_test"},
    )


def build_config(framework: str) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="arch03-smoke",
        run_id="arch03-smoke-001",
        framework=framework,
        architecture=ARCHITECTURE,
        model_provider="local",
        model_name="deterministic-router-v1",
        temperature=0.0,
        max_agent_iterations=3,
        timeout_seconds=120,
        retry_count=0,
        random_seed=42,
        metadata={
            "input_cost_per_1k_tokens": 0.0,
            "output_cost_per_1k_tokens": 0.0,
            "notes": "Deterministic local LLM for ARCH_03 smoke comparability.",
        },
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    input_data = build_smoke_case()

    for framework in FRAMEWORKS:
        runner = load_runner(repo_root, framework)
        result = runner(input_data, build_config(framework))
        output_path = save_result_json(result, base_dir=repo_root / "results" / "raw")
        print(
            framework,
            result.status.value,
            f"steps={result.metrics.step_count}",
            f"llm_calls={result.metrics.llm_call_count}",
            f"tokens={result.metrics.token_usage.total_tokens}",
            output_path,
        )


if __name__ == "__main__":
    main()
