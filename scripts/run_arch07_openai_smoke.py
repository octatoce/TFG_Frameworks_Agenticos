"""Run a synthetic ARCH_07 Map-Reduce smoke against OpenAI.

Requires .env with OPENAI_API_KEY. MODEL_NAME defaults to gpt-4o-mini.
Seven invented documents with batch_size=3 produce three mappers and one
reducer in every framework.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from benchmark_core.result_writer import save_result_json
from benchmark_core.schemas import DocumentInput, ExperimentConfig, ExperimentInput, RunStatus


ARCHITECTURE = "ARCH_07_MAP_REDUCE_AGENTIC"
FRAMEWORKS = [
    "langgraph",
    "crewai",
    "microsoft_agent_framework",
    "llamaindex",
    "pydantic_ai",
]
BATCH_SIZE = 3


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
    facts = [
        (
            "infraestructura",
            "El sistema actual procesa 120 expedientes diarios y alcanza picos de 210. "
            "El servidor disponible tiene 16 nucleos y 64 GB de memoria.",
        ),
        (
            "latencia",
            "El objetivo operativo exige finalizar cada lote en menos de 30 segundos. "
            "La solucion anterior tarda 52 segundos con 100 expedientes.",
        ),
        (
            "coste",
            "El presupuesto mensual maximo es 900 euros. La estimacion inicial del servicio "
            "gestionado es de 620 euros mensuales.",
        ),
        (
            "seguridad",
            "Los documentos contienen datos personales. El procesamiento debe mantener trazas "
            "auditables y no puede conservar contenido despues de 30 dias.",
        ),
        (
            "calidad",
            "Una prueba interna sobre 80 casos obtuvo 91 por ciento de respuestas correctas, "
            "pero 6 casos carecian de evidencia suficiente.",
        ),
        (
            "riesgo",
            "El proveedor declara una disponibilidad de 99,5 por ciento. El requisito interno "
            "preferido es 99,9 por ciento, por lo que existe una discrepancia pendiente.",
        ),
        (
            "alternativa",
            "Una opcion hibrida mantiene los documentos sensibles localmente y envia solo "
            "fragmentos anonimizados, con un coste estimado de 780 euros mensuales.",
        ),
    ]
    return ExperimentInput(
        case_id="arch07-openai-smoke-001",
        dataset_id="synthetic-map-reduce-smoke",
        task_type="multi_document_decision_qa",
        query=(
            "Decide si conviene adoptar el servicio gestionado, rechazarlo o usar la opcion "
            "hibrida. Fundamenta la respuesta con evidencias, contradicciones y limitaciones."
        ),
        documents=[
            DocumentInput(
                document_id=f"synthetic-{index:03d}-{topic}",
                content=content,
                metadata={"synthetic": True, "topic": topic},
            )
            for index, (topic, content) in enumerate(facts, start=1)
        ],
        evaluation_criteria={
            "expected_evidence_topics": ["coste", "seguridad", "disponibilidad"],
            "must_acknowledge_uncertainty": True,
        },
        metadata={"purpose": "arch07_openai_basic_smoke", "synthetic": True},
    )


def build_config(framework: str) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="arch07-openai-smoke",
        run_id=f"arch07-openai-smoke-001-{framework}",
        framework=framework,
        architecture=ARCHITECTURE,
        model_provider="openai",
        model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        temperature=float(os.getenv("TEMPERATURE", "0.0")),
        max_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "512")),
        max_agent_iterations=int(os.getenv("MAX_AGENT_ITERATIONS", "5")),
        timeout_seconds=int(os.getenv("TIMEOUT_SECONDS", "180")),
        retry_count=int(os.getenv("RETRY_COUNT", "0")),
        random_seed=42,
        metadata={
            "env_file": ".env",
            "map_reduce_batch_size": BATCH_SIZE,
            "microsoft_openai_client": os.getenv("MICROSOFT_OPENAI_CLIENT", "responses"),
            "input_cost_per_1k_tokens": float(os.getenv("INPUT_COST_PER_1K_TOKENS", "0.0")),
            "output_cost_per_1k_tokens": float(os.getenv("OUTPUT_COST_PER_1K_TOKENS", "0.0")),
            "notes": "Synthetic OpenAI smoke for ARCH_07 across all frameworks.",
        },
    )


def validate_result(result, input_data: ExperimentInput) -> None:
    mapper_steps = [step for step in result.steps if step.step_type == "map_batch_llm_call"]
    reducer_step = result.steps[-1]
    execution = result.metrics.metadata["map_reduce_execution"]
    mapped_ids = [
        document_id
        for step in mapper_steps
        for document_id in step.input_data["document_ids"]
    ]

    assert result.status == RunStatus.SUCCESS
    assert result.final_answer
    assert [step.name for step in result.steps] == [
        "document_partitioner",
        "mapper_001",
        "mapper_002",
        "mapper_003",
        "reducer",
    ]
    assert result.metrics.llm_call_count == 4
    assert len(result.llm_calls) == 4
    assert mapped_ids == [document.document_id for document in input_data.documents]
    assert len(mapped_ids) == len(set(mapped_ids))
    assert reducer_step.started_at >= max(step.finished_at for step in mapper_steps)
    assert reducer_step.input_data["original_documents_included"] is False
    assert set(reducer_step.input_data["partial_outputs"]) == {
        "batch_001",
        "batch_002",
        "batch_003",
    }
    mapper_span = (
        max(step.finished_at for step in mapper_steps)
        - min(step.started_at for step in mapper_steps)
    ).total_seconds()
    summed_mapper_latency = sum(
        (step.finished_at - step.started_at).total_seconds() for step in mapper_steps
    )
    assert mapper_span < summed_mapper_latency
    assert execution["total_documents"] == 7
    assert execution["batch_size"] == BATCH_SIZE
    assert execution["batch_count"] == 3
    assert execution["mapper_count"] == 3
    assert execution["batches_failed"] == []
    assert execution["parallelism_used"] is True
    assert execution["fallback_sequential"] is False
    assert execution["throughput_docs_per_second"] > 0
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
            validate_result(result, input_data)
            output_path = save_result_json(result, base_dir=repo_root / "results" / "raw")
            methods = sorted(
                {str(call.metadata.get("token_counting_method", "unknown")) for call in result.llm_calls}
            )
            execution = result.metrics.metadata["map_reduce_execution"]
            print(
                framework,
                result.status.value,
                f"latency_ms={result.metrics.metadata['latency_total_ms']:.1f}",
                f"llm_calls={result.metrics.llm_call_count}",
                f"tokens={result.metrics.token_usage.total_tokens}",
                f"throughput_docs_s={execution['throughput_docs_per_second']:.3f}",
                f"token_counting={','.join(methods)}",
                output_path,
            )
        except Exception as exc:  # pragma: no cover - real API integration path
            failures.append(f"{framework}: {type(exc).__name__}: {exc}")
            print(f"{framework} FAILED {type(exc).__name__}: {exc}")

    if failures:
        raise RuntimeError("ARCH_07 OpenAI smoke failures:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
