"""Run the shared ARCH_01 smoke case against the OpenAI API.

Requires a local .env file with OPENAI_API_KEY. MODEL_NAME defaults to gpt-4o-mini
when it is not set.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from benchmark_core.result_writer import save_result_json
from benchmark_core.schemas import DocumentInput, ExperimentConfig, ExperimentInput


ARCHITECTURE = "ARCH_01_SINGLE_REACT"
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
    spec = importlib.util.spec_from_file_location(f"{framework}_{ARCHITECTURE}_openai_run", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load runner from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_architecture


def build_smoke_case() -> ExperimentInput:
    return ExperimentInput(
        case_id="arch01-openai-smoke-001",
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
                    "La matriz experimental compara cinco frameworks y tres arquitecturas, "
                    "manteniendo contratos comunes, recogida de metricas, ejecucion comparable "
                    "y persistencia JSON de resultados raw."
                ),
                metadata={"source": "synthetic_openai_smoke_case"},
            )
        ],
        evaluation_criteria={
            "expected_contains": ["frameworks", "metricas", "JSON"]
        },
        metadata={"purpose": "arch01_openai_smoke_test"},
    )


def build_config(framework: str) -> ExperimentConfig:
    max_iterations = int(
        os.getenv("MAX_AGENT_ITERATIONS")
        or os.getenv("MAX_ITERATIONS")
        or "3"
    )
    return ExperimentConfig(
        experiment_id="arch01-openai-smoke",
        run_id=f"arch01-openai-smoke-001-{framework}",
        framework=framework,
        architecture=ARCHITECTURE,
        model_provider="openai",
        model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        temperature=float(os.getenv("TEMPERATURE", "0.0")),
        max_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "256")),
        max_agent_iterations=max_iterations,
        timeout_seconds=int(os.getenv("TIMEOUT_SECONDS", "120")),
        retry_count=int(os.getenv("RETRY_COUNT", "0")),
        random_seed=42,
        metadata={
            "env_file": ".env",
            "microsoft_openai_client": os.getenv("MICROSOFT_OPENAI_CLIENT", "responses"),
            "input_cost_per_1k_tokens": float(os.getenv("INPUT_COST_PER_1K_TOKENS", "0.0")),
            "output_cost_per_1k_tokens": float(os.getenv("OUTPUT_COST_PER_1K_TOKENS", "0.0")),
            "notes": "OpenAI API smoke run for ARCH_01 across all implemented frameworks.",
        },
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")
    input_data = build_smoke_case()

    for framework in FRAMEWORKS:
        runner = load_runner(repo_root, framework)
        result = runner(input_data, build_config(framework))
        output_path = save_result_json(result, base_dir=repo_root / "results" / "raw")
        print(
            framework,
            result.status.value,
            f"llm_calls={result.metrics.llm_call_count}",
            f"tokens={result.metrics.token_usage.total_tokens}",
            output_path,
        )


if __name__ == "__main__":
    main()
