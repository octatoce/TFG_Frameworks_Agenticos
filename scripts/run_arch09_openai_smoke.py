"""Run ARCH_09 Reflection/Critic Loop against OpenAI in all frameworks.

Requires .env with OPENAI_API_KEY. MODEL_NAME defaults to gpt-4o-mini.
The smoke accepts an early quality stop or one bounded revision, while requiring
real OpenAI usage counters, canonical traces, and raw-result persistence.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from benchmark_core.reflection_critic_loop import (
    CRITIC,
    GENERATOR,
    REVISER,
    STOP_CONTROLLER,
)
from benchmark_core.result_writer import save_result_json
from benchmark_core.schemas import DocumentInput, ExperimentConfig, ExperimentInput, RunStatus


ARCHITECTURE = "ARCH_09_REFLECTION_CRITIC_LOOP"
FRAMEWORKS = [
    "langgraph",
    "crewai",
    "microsoft_agent_framework",
    "llamaindex",
    "pydantic_ai",
]
MAX_REFLECTION_ITERATIONS = 2


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
        case_id="arch09-openai-smoke-001",
        dataset_id="synthetic-reflection-smoke",
        task_type="ambiguous_risk_decision",
        query=(
            "Decide si un hospital debe desplegar ahora un asistente LLM para priorizar consultas, "
            "limitarlo a un piloto supervisado o posponerlo. Explica la decisión, evidencia, "
            "incertidumbres y controles necesarios."
        ),
        documents=[
            DocumentInput(
                document_id="arch09-performance",
                content=(
                    "En 500 consultas retrospectivas, el prototipo clasificó correctamente el 94 por "
                    "ciento y redujo el tiempo medio de primera revisión de 18 a 4 minutos."
                ),
                metadata={"topic": "performance", "synthetic": True},
            ),
            DocumentInput(
                document_id="arch09-safety",
                content=(
                    "En 22 casos ambiguos hubo 4 prioridades incorrectas; una de ellas habría retrasado "
                    "una revisión urgente. La prueba no incluyó pacientes pediátricos."
                ),
                metadata={"topic": "safety", "synthetic": True},
            ),
            DocumentInput(
                document_id="arch09-controls",
                content=(
                    "El hospital puede financiar un piloto de tres meses con revisión humana obligatoria, "
                    "auditoría semanal, exclusión pediátrica y parada automática ante incidencias graves."
                ),
                metadata={"topic": "controls", "synthetic": True},
            ),
        ],
        evaluation_criteria={
            "must_consider": [
                "performance",
                "safety_error",
                "population_gap",
                "human_review",
            ],
            "reasonable_decisions": ["supervised_pilot", "postpone_pending_validation"],
        },
        metadata={"purpose": "arch09_openai_smoke", "synthetic": True},
    )


def build_config(framework: str) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="arch09-openai-smoke",
        run_id=f"arch09-openai-smoke-001-{framework}",
        framework=framework,
        architecture=ARCHITECTURE,
        model_provider="openai",
        model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        temperature=float(os.getenv("TEMPERATURE", "0.0")),
        max_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "700")),
        max_agent_iterations=MAX_REFLECTION_ITERATIONS,
        timeout_seconds=int(os.getenv("TIMEOUT_SECONDS", "240")),
        retry_count=int(os.getenv("RETRY_COUNT", "0")),
        random_seed=42,
        metadata={
            "env_file": ".env",
            "reflection_max_iterations": MAX_REFLECTION_ITERATIONS,
            "reflection_quality_threshold": float(
                os.getenv("REFLECTION_QUALITY_THRESHOLD", "0.95")
            ),
            "microsoft_openai_client": os.getenv("MICROSOFT_OPENAI_CLIENT", "responses"),
            "input_cost_per_1k_tokens": float(
                os.getenv("INPUT_COST_PER_1K_TOKENS", "0.0")
            ),
            "output_cost_per_1k_tokens": float(
                os.getenv("OUTPUT_COST_PER_1K_TOKENS", "0.0")
            ),
            "notes": "Synthetic OpenAI smoke for ARCH_09 across all frameworks.",
        },
    )


def validate_result(result, repo_root: Path) -> None:
    structured = result.structured_output
    execution = result.metrics.metadata["reflection_execution"]
    iterations = structured["iterations_executed"]
    revisions = structured["revision_count"]

    assert result.status == RunStatus.SUCCESS
    assert result.final_answer
    assert result.steps[0].name == GENERATOR
    assert result.steps[-1].name.startswith(STOP_CONTROLLER)
    assert 1 <= iterations <= MAX_REFLECTION_ITERATIONS
    assert revisions in {0, 1}
    assert structured["number_of_versions"] == revisions + 1
    assert len(structured["version_history"]) == revisions + 1
    assert len(structured["version_summaries"]) == revisions + 1
    assert len(structured["critique_history"]) == iterations
    assert len(structured["stop_decisions"]) == iterations
    assert structured["stop_decisions"][-1]["should_stop"] is True
    assert structured["stop_reason"] in {
        "quality_sufficient",
        "quality_score_threshold_reached",
        "minor_issues_only",
        "max_iterations_reached",
    }
    assert execution["iterations_executed"] == iterations
    assert execution["revision_count"] == revisions
    assert execution["parallelism_used"] is False

    expected_calls = 1 + iterations + revisions
    assert result.metrics.llm_call_count == expected_calls
    assert len(result.llm_calls) == expected_calls
    assert result.metrics.step_count == 1 + (2 * iterations) + revisions
    assert sum(step.name.startswith(CRITIC) for step in result.steps) == iterations
    assert sum(step.name.startswith(REVISER) for step in result.steps) == revisions
    assert sum(step.name.startswith(STOP_CONTROLLER) for step in result.steps) == iterations
    assert all(call.latency_seconds >= 0 for call in result.llm_calls)
    assert all(call.token_usage.total_tokens > 0 for call in result.llm_calls)
    assert all(
        call.metadata.get("token_counting_method") == "openai_usage"
        for call in result.llm_calls
    )
    assert result.metrics.token_usage.total_tokens == sum(
        call.token_usage.total_tokens for call in result.llm_calls
    )

    for index, step in enumerate(result.steps):
        if step.name.startswith(REVISER):
            assert result.steps[index - 1].name.startswith(STOP_CONTROLLER)
            assert result.steps[index + 1].name.startswith(CRITIC)

    raw_path = (
        repo_root
        / "results"
        / "raw"
        / result.framework
        / ARCHITECTURE
        / f"{result.run_id}.json"
    )
    assert raw_path.exists()


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
            output_path = save_result_json(result, base_dir=repo_root / "results" / "raw")
            validate_result(result, repo_root)
            methods = sorted(
                {
                    str(call.metadata.get("token_counting_method", "unknown"))
                    for call in result.llm_calls
                }
            )
            print(
                framework,
                result.status.value,
                f"latency_ms={result.metrics.metadata['latency_total_ms']:.1f}",
                f"llm_calls={result.metrics.llm_call_count}",
                f"tokens={result.metrics.token_usage.total_tokens}",
                f"iterations={result.structured_output['iterations_executed']}",
                f"revisions={result.structured_output['revision_count']}",
                f"stop={result.structured_output['stop_reason']}",
                f"token_counting={','.join(methods)}",
                output_path,
            )
        except Exception as exc:  # pragma: no cover - real API integration path
            failures.append(f"{framework}: {type(exc).__name__}: {exc}")
            print(f"{framework} FAILED {type(exc).__name__}: {exc}")

    if failures:
        raise RuntimeError("ARCH_09 OpenAI smoke failures:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
