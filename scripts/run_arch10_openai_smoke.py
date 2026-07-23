"""Run ARCH_10 checkpoint/recovery against OpenAI in all frameworks.

Requires .env with OPENAI_API_KEY. MODEL_NAME defaults to gpt-4o-mini.
The smoke injects one deterministic failure after the common checkpoint,
requires successful recovery, verifies real OpenAI usage, and prints a
comparable Markdown table.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from benchmark_core.checkpoint_memory_recovery import RECOVERY_COMPONENTS
from benchmark_core.schemas import (
    DocumentInput,
    ExperimentConfig,
    ExperimentInput,
    RunStatus,
)


ARCHITECTURE = "ARCH_10_CHECKPOINT_MEMORY_RECOVERY"
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
        case_id="arch10-openai-smoke-001",
        dataset_id="synthetic-recovery-smoke",
        task_type="resilient_operational_decision",
        query=(
            "Propón una decisión operativa segura para desplegar un asistente LLM de triaje "
            "hospitalario. El análisis inicial debe sobrevivir a un fallo controlado y la "
            "respuesta final debe usar únicamente la evidencia recuperada."
        ),
        documents=[
            DocumentInput(
                document_id="arch10-performance",
                content=(
                    "En 500 consultas retrospectivas, el prototipo acertó el 94 por ciento "
                    "y redujo de 18 a 4 minutos el tiempo medio de primera revisión."
                ),
                metadata={"topic": "performance", "synthetic": True},
            ),
            DocumentInput(
                document_id="arch10-safety",
                content=(
                    "En 22 casos ambiguos hubo cuatro prioridades incorrectas, incluida una "
                    "que habría retrasado una revisión urgente; no hubo casos pediátricos."
                ),
                metadata={"topic": "safety", "synthetic": True},
            ),
            DocumentInput(
                document_id="arch10-controls",
                content=(
                    "Es viable un piloto de tres meses con revisión humana obligatoria, "
                    "auditoría semanal, exclusión pediátrica y parada ante incidentes graves."
                ),
                metadata={"topic": "controls", "synthetic": True},
            ),
        ],
        evaluation_criteria={
            "must_survive_checkpoint": [
                "performance",
                "safety_error",
                "population_gap",
                "human_review",
            ],
            "reasonable_decision": "supervised_pilot",
        },
        metadata={"purpose": "arch10_openai_smoke", "synthetic": True},
    )


def build_config(framework: str) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="arch10-openai-smoke",
        run_id=f"arch10-openai-smoke-001-{framework}",
        framework=framework,
        architecture=ARCHITECTURE,
        model_provider="openai",
        model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        temperature=float(os.getenv("TEMPERATURE", "0.0")),
        max_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "700")),
        max_agent_iterations=2,
        timeout_seconds=int(os.getenv("TIMEOUT_SECONDS", "240")),
        retry_count=int(os.getenv("RETRY_COUNT", "0")),
        random_seed=42,
        metadata={
            "env_file": ".env",
            "checkpoint_inject_failure": True,
            "microsoft_openai_client": os.getenv(
                "MICROSOFT_OPENAI_CLIENT", "responses"
            ),
            "input_cost_per_1k_tokens": float(
                os.getenv("INPUT_COST_PER_1K_TOKENS", "0.0")
            ),
            "output_cost_per_1k_tokens": float(
                os.getenv("OUTPUT_COST_PER_1K_TOKENS", "0.0")
            ),
            "notes": "Synthetic OpenAI smoke for ARCH_10 across all frameworks.",
        },
    )


def validate_result(result, repo_root: Path) -> None:
    structured = result.structured_output
    execution = result.metrics.metadata["recovery_execution"]

    assert result.status == RunStatus.SUCCESS
    assert result.final_answer
    assert [step.name for step in result.steps] == list(RECOVERY_COMPONENTS)
    assert result.metrics.step_count == 7
    assert result.metrics.llm_call_count == 2
    assert len(result.llm_calls) == 2
    assert structured["checkpoint_used"] is True
    assert structured["failure_injected"] is True
    assert structured["recovery_attempted"] is True
    assert structured["recovery_successful"] is True
    assert structured["result_generated_after_recovery"] is True
    assert execution["checkpoints_created"] == 1
    assert execution["controlled_error_count"] == 1
    assert execution["uncontrolled_error_count"] == 0
    assert execution["state_digest_verified"] is True
    assert execution["steps_before_failure"] == 3
    assert execution["steps_after_recovery"] == 2
    assert all(call.latency_seconds >= 0 for call in result.llm_calls)
    assert all(call.token_usage.total_tokens > 0 for call in result.llm_calls)
    assert all(
        call.metadata.get("token_counting_method") == "openai_usage"
        for call in result.llm_calls
    )
    assert result.metrics.token_usage.total_tokens == sum(
        call.token_usage.total_tokens for call in result.llm_calls
    )

    if result.framework == "langgraph":
        assert execution["checkpoint_backend"].endswith("SqliteSaver")
        assert execution["native_checkpointing"] is True
        assert execution["durable_storage"] is True
        assert execution["database_reopened_for_recovery"] is True
        assert ".sqlite" in execution["recovery_source"]

    raw_path = (
        repo_root
        / "results"
        / "raw"
        / result.framework
        / ARCHITECTURE
        / f"{result.run_id}.json"
    )
    assert raw_path.exists()


def markdown_table(rows: list[dict[str, object]]) -> str:
    headers = [
        "framework",
        "status",
        "backend",
        "native",
        "total_ms",
        "checkpoint_step_ms",
        "recovery_step_ms",
        "post_ms",
        "calls",
        "tokens",
    ]
    rendered = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        rendered.append(
            "| " + " | ".join(str(row[header]) for header in headers) + " |"
        )
    return "\n".join(rendered)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured in .env or the environment.")

    input_data = build_smoke_case()
    failures: list[str] = []
    rows: list[dict[str, object]] = []
    for framework in FRAMEWORKS:
        try:
            result = load_runner(repo_root, framework)(
                input_data, build_config(framework)
            )
            validate_result(result, repo_root)
            execution = result.structured_output["recovery_execution"]
            rows.append(
                {
                    "framework": framework,
                    "status": result.status.value,
                    "backend": execution["checkpoint_backend"],
                    "native": execution["native_checkpointing"],
                    "total_ms": f"{execution.get('latency_total_ms', result.metrics.metadata['latency_total_ms']):.1f}",
                    "checkpoint_step_ms": f"{execution['checkpoint_write_latency_ms']:.1f}",
                    "recovery_step_ms": f"{execution['recovery_latency_ms']:.1f}",
                    "post_ms": f"{execution['latency_after_recovery_ms']:.1f}",
                    "calls": result.metrics.llm_call_count,
                    "tokens": result.metrics.token_usage.total_tokens,
                }
            )
        except Exception as exc:  # pragma: no cover - real API integration path
            failures.append(f"{framework}: {type(exc).__name__}: {exc}")
            print(f"{framework} FAILED {type(exc).__name__}: {exc}")

    if rows:
        print(markdown_table(rows))
    if failures:
        raise RuntimeError("ARCH_10 OpenAI smoke failures:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
