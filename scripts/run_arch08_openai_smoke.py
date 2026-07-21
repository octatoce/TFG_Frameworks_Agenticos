"""Run an ambiguous ARCH_08 Debate + Judge smoke against OpenAI.

Requires .env with OPENAI_API_KEY. MODEL_NAME defaults to gpt-4o-mini.
Every framework must produce three independent proposals, one explicit debate
round, and one final judge decision: five OpenAI calls per framework.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from benchmark_core.debate_judge import DEBATERS, DEBATE_ROUND, JUDGE
from benchmark_core.result_writer import save_result_json
from benchmark_core.schemas import DocumentInput, ExperimentConfig, ExperimentInput, RunStatus


ARCHITECTURE = "ARCH_08_DEBATE_JUDGE"
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
        case_id="arch08-openai-smoke-001",
        dataset_id="synthetic-debate-judge-smoke",
        task_type="ambiguous_policy_decision",
        query=(
            "Una universidad quiere usar un asistente LLM para orientar a estudiantes. "
            "Decide si debe desplegarlo directamente, limitarlo a un piloto o rechazarlo."
        ),
        documents=[
            DocumentInput(
                document_id="arch08-benefits",
                content=(
                    "El prototipo resolvio correctamente el 92 por ciento de 200 consultas "
                    "informativas y redujo el tiempo medio de respuesta de 14 minutos a 40 segundos."
                ),
                metadata={"topic": "benefits", "synthetic": True},
            ),
            DocumentInput(
                document_id="arch08-risks",
                content=(
                    "En 16 consultas ambiguas el prototipo dio 5 respuestas demasiado seguras. "
                    "Las decisiones academicas vinculantes deben seguir siendo revisadas por personal humano."
                ),
                metadata={"topic": "risks", "synthetic": True},
            ),
            DocumentInput(
                document_id="arch08-constraints",
                content=(
                    "Existe presupuesto para un piloto de seis meses con supervision humana, "
                    "registro de incidencias y prohibicion de conservar datos personales."
                ),
                metadata={"topic": "constraints", "synthetic": True},
            ),
        ],
        evaluation_criteria={
            "must_consider": ["benefits", "ambiguity", "human_review", "privacy"],
            "reasonable_decisions": ["limited_pilot", "reject_pending_controls"],
        },
        metadata={"purpose": "arch08_openai_smoke", "synthetic": True},
    )


def build_config(framework: str) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="arch08-openai-smoke",
        run_id=f"arch08-openai-smoke-001-{framework}",
        framework=framework,
        architecture=ARCHITECTURE,
        model_provider="openai",
        model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        temperature=float(os.getenv("TEMPERATURE", "0.0")),
        max_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "700")),
        max_agent_iterations=int(os.getenv("MAX_AGENT_ITERATIONS", "5")),
        timeout_seconds=int(os.getenv("TIMEOUT_SECONDS", "240")),
        retry_count=int(os.getenv("RETRY_COUNT", "0")),
        random_seed=42,
        metadata={
            "env_file": ".env",
            "microsoft_openai_client": os.getenv("MICROSOFT_OPENAI_CLIENT", "responses"),
            "input_cost_per_1k_tokens": float(os.getenv("INPUT_COST_PER_1K_TOKENS", "0.0")),
            "output_cost_per_1k_tokens": float(os.getenv("OUTPUT_COST_PER_1K_TOKENS", "0.0")),
            "notes": "Synthetic OpenAI smoke for ARCH_08 across all frameworks.",
        },
    )


def validate_result(result) -> None:
    expected_steps = [*DEBATERS, DEBATE_ROUND, JUDGE]
    proposal_steps = result.steps[:3]
    debate_step = result.steps[3]
    judge_step = result.steps[4]
    structured = result.structured_output
    execution = result.metrics.metadata["debate_execution"]

    assert result.status == RunStatus.SUCCESS
    assert result.final_answer
    assert [step.name for step in result.steps] == expected_steps
    assert result.metrics.step_count == 5
    assert result.metrics.llm_call_count == 5
    assert len(result.llm_calls) == 5
    assert set(structured["proposals"]) == set(DEBATERS)
    assert structured["number_of_proposals"] == 3
    assert structured["number_of_debate_rounds"] == 1
    assert structured["critique_count"] == 3
    assert structured["decision_mode"] in {"select", "combine", "reject"}
    assert structured["judge"]["answer"] == result.final_answer
    assert structured["judge"]["rationale"]
    assert debate_step.input_data["depends_on"] == list(DEBATERS)
    assert judge_step.input_data["depends_on"] == [*DEBATERS, DEBATE_ROUND]
    assert debate_step.started_at >= max(step.finished_at for step in proposal_steps)
    assert judge_step.started_at >= debate_step.finished_at
    assert execution["proposal_count"] == 3
    assert execution["debate_round_count"] == 1
    assert set(execution["component_metrics"]) == set(expected_steps)
    assert all(
        values["llm_call_count"] == 1
        for values in execution["component_metrics"].values()
    )
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
            methods = sorted(
                {str(call.metadata.get("token_counting_method", "unknown")) for call in result.llm_calls}
            )
            print(
                framework,
                result.status.value,
                f"latency_ms={result.metrics.metadata['latency_total_ms']:.1f}",
                f"llm_calls={result.metrics.llm_call_count}",
                f"tokens={result.metrics.token_usage.total_tokens}",
                f"decision={result.structured_output['decision_mode']}",
                f"disagreements={result.structured_output['disagreement_count']}",
                f"token_counting={','.join(methods)}",
                output_path,
            )
        except Exception as exc:  # pragma: no cover - real API integration path
            failures.append(f"{framework}: {type(exc).__name__}: {exc}")
            print(f"{framework} FAILED {type(exc).__name__}: {exc}")

    if failures:
        raise RuntimeError("ARCH_08 OpenAI smoke failures:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
