"""Run the shared ARCH_06 smoke case against OpenAI in all frameworks.

Requires .env with OPENAI_API_KEY. MODEL_NAME defaults to gpt-4o-mini.
The script validates the canonical fan-out/fan-in trace and prints the token
counting method used by each framework adapter.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from benchmark_core.parallel_fanout_fanin import AGGREGATOR, PARALLEL_BRANCHES
from benchmark_core.result_writer import save_result_json
from benchmark_core.schemas import DocumentInput, ExperimentConfig, ExperimentInput, RunStatus


ARCHITECTURE = "ARCH_06_PARALLEL_FANOUT_FANIN"
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
    spec = importlib.util.spec_from_file_location(
        f"{framework}_{ARCHITECTURE}_openai_run",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load runner from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_architecture


def build_smoke_case() -> ExperimentInput:
    return ExperimentInput(
        case_id="arch06-openai-smoke-001",
        dataset_id="samples",
        task_type="document_qa",
        query=(
            "Evalua si el benchmark descrito permite una comparacion tecnica justa, "
            "incluyendo evidencias, riesgos y una alternativa razonable."
        ),
        documents=[
            DocumentInput(
                document_id="sample-doc-arch06-001",
                content=(
                    "El TFG compara cinco frameworks con el mismo modelo, temperatura, "
                    "dataset, timeout, herramientas, contratos y schemas. ARCH_06 ejecuta "
                    "cuatro perspectivas independientes sobre el mismo input y un agregador "
                    "final. benchmark_core registra llamadas, tokens, latencias, pasos, errores "
                    "y resultados raw JSON."
                ),
                metadata={"source": "synthetic_arch06_openai_smoke_case"},
            )
        ],
        evaluation_criteria={
            "expected_contains": ["comparacion", "riesgos", "alternativa"]
        },
        metadata={"purpose": "arch06_openai_smoke_test"},
    )


def build_config(framework: str) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="arch06-openai-smoke",
        run_id=f"arch06-openai-smoke-001-{framework}",
        framework=framework,
        architecture=ARCHITECTURE,
        model_provider="openai",
        model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        temperature=float(os.getenv("TEMPERATURE", "0.0")),
        max_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "512")),
        max_agent_iterations=int(os.getenv("MAX_AGENT_ITERATIONS", "3")),
        timeout_seconds=int(os.getenv("TIMEOUT_SECONDS", "180")),
        retry_count=int(os.getenv("RETRY_COUNT", "0")),
        random_seed=42,
        metadata={
            "env_file": ".env",
            "microsoft_openai_client": os.getenv("MICROSOFT_OPENAI_CLIENT", "responses"),
            "input_cost_per_1k_tokens": float(os.getenv("INPUT_COST_PER_1K_TOKENS", "0.0")),
            "output_cost_per_1k_tokens": float(os.getenv("OUTPUT_COST_PER_1K_TOKENS", "0.0")),
            "notes": "OpenAI API smoke run for ARCH_06 across all frameworks.",
        },
    )


def validate_result(result) -> None:
    expected_steps = [*PARALLEL_BRANCHES, AGGREGATOR]
    step_names = [step.name for step in result.steps]
    execution = result.metrics.metadata["parallel_execution"]
    branch_steps = result.steps[:-1]
    aggregator_step = result.steps[-1]

    assert result.status == RunStatus.SUCCESS
    assert result.final_answer
    assert step_names == expected_steps
    assert result.metrics.llm_call_count == 5
    assert len(result.llm_calls) == 5
    assert result.metrics.metadata["latency_total_ms"] >= 0
    assert execution["parallelism_used"] is True
    assert execution["fallback_sequential"] is False
    assert execution["branches_completed"] == list(PARALLEL_BRANCHES)
    assert execution["branches_failed"] == []
    assert set(execution["branch_metrics"]) == set(PARALLEL_BRANCHES)
    assert all(values["llm_call_count"] == 1 for values in execution["branch_metrics"].values())
    assert execution["aggregator_metrics"]["llm_call_count"] == 1
    assert aggregator_step.input_data["depends_on"] == list(PARALLEL_BRANCHES)
    assert set(aggregator_step.input_data["partial_outputs"]) == set(PARALLEL_BRANCHES)
    assert aggregator_step.started_at >= max(step.finished_at for step in branch_steps)
    branch_span = (
        max(step.finished_at for step in branch_steps)
        - min(step.started_at for step in branch_steps)
    ).total_seconds()
    summed_branch_latency = sum(
        (step.finished_at - step.started_at).total_seconds() for step in branch_steps
    )
    assert branch_span < summed_branch_latency
    assert all(call.latency_seconds >= 0 for call in result.llm_calls)
    assert all(call.token_usage.total_tokens > 0 for call in result.llm_calls)
    assert all(
        call.metadata.get("token_counting_method") == "openai_usage"
        for call in result.llm_calls
    )
    assert result.metrics.token_usage.total_tokens == sum(
        call.token_usage.total_tokens for call in result.llm_calls
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured in .env or the environment.")

    input_data = build_smoke_case()
    failures: list[str] = []
    for framework in FRAMEWORKS:
        try:
            result = load_runner(repo_root, framework)(input_data, build_config(framework))
            validate_result(result)
            output_path = save_result_json(result, base_dir=repo_root / "results" / "raw")
            counting_methods = sorted(
                {str(call.metadata.get("token_counting_method", "unknown")) for call in result.llm_calls}
            )
            print(
                framework,
                result.status.value,
                f"latency_ms={result.metrics.metadata['latency_total_ms']:.1f}",
                f"llm_calls={result.metrics.llm_call_count}",
                f"tokens={result.metrics.token_usage.total_tokens}",
                f"token_counting={','.join(counting_methods)}",
                f"parallel={result.structured_output['parallelism_used']}",
                output_path,
            )
        except Exception as exc:  # pragma: no cover - real API integration path
            failures.append(f"{framework}: {type(exc).__name__}: {exc}")
            print(f"{framework} FAILED {type(exc).__name__}: {exc}")

    if failures:
        raise RuntimeError("ARCH_06 OpenAI smoke failures:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
