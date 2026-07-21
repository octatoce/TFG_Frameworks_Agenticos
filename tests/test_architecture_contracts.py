import importlib.util
import inspect
import re
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
    Path("implementations/langgraph/ARCH_03_ROUTER_SPECIALISTS/run.py"),
    Path("implementations/langgraph/ARCH_04_SUPERVISOR_WORKERS/run.py"),
    Path("implementations/langgraph/ARCH_05_HANDOFF_SWARM/run.py"),
    Path("implementations/langgraph/ARCH_06_PARALLEL_FANOUT_FANIN/run.py"),
    Path("implementations/langgraph/ARCH_07_MAP_REDUCE_AGENTIC/run.py"),
    Path("implementations/langgraph/ARCH_08_DEBATE_JUDGE/run.py"),
    Path("implementations/crewai/ARCH_01_SINGLE_REACT/run.py"),
    Path("implementations/crewai/ARCH_02_SEQUENTIAL_PIPELINE/run.py"),
    Path("implementations/crewai/ARCH_03_ROUTER_SPECIALISTS/run.py"),
    Path("implementations/crewai/ARCH_04_SUPERVISOR_WORKERS/run.py"),
    Path("implementations/crewai/ARCH_05_HANDOFF_SWARM/run.py"),
    Path("implementations/crewai/ARCH_06_PARALLEL_FANOUT_FANIN/run.py"),
    Path("implementations/crewai/ARCH_07_MAP_REDUCE_AGENTIC/run.py"),
    Path("implementations/crewai/ARCH_08_DEBATE_JUDGE/run.py"),
    Path("implementations/microsoft_agent_framework/ARCH_01_SINGLE_REACT/run.py"),
    Path("implementations/microsoft_agent_framework/ARCH_02_SEQUENTIAL_PIPELINE/run.py"),
    Path("implementations/microsoft_agent_framework/ARCH_03_ROUTER_SPECIALISTS/run.py"),
    Path("implementations/microsoft_agent_framework/ARCH_04_SUPERVISOR_WORKERS/run.py"),
    Path("implementations/microsoft_agent_framework/ARCH_05_HANDOFF_SWARM/run.py"),
    Path("implementations/microsoft_agent_framework/ARCH_06_PARALLEL_FANOUT_FANIN/run.py"),
    Path("implementations/microsoft_agent_framework/ARCH_07_MAP_REDUCE_AGENTIC/run.py"),
    Path("implementations/microsoft_agent_framework/ARCH_08_DEBATE_JUDGE/run.py"),
    Path("implementations/llamaindex/ARCH_01_SINGLE_REACT/run.py"),
    Path("implementations/llamaindex/ARCH_02_SEQUENTIAL_PIPELINE/run.py"),
    Path("implementations/llamaindex/ARCH_03_ROUTER_SPECIALISTS/run.py"),
    Path("implementations/llamaindex/ARCH_04_SUPERVISOR_WORKERS/run.py"),
    Path("implementations/llamaindex/ARCH_05_HANDOFF_SWARM/run.py"),
    Path("implementations/llamaindex/ARCH_06_PARALLEL_FANOUT_FANIN/run.py"),
    Path("implementations/llamaindex/ARCH_07_MAP_REDUCE_AGENTIC/run.py"),
    Path("implementations/llamaindex/ARCH_08_DEBATE_JUDGE/run.py"),
    Path("implementations/pydantic_ai/ARCH_01_SINGLE_REACT/run.py"),
    Path("implementations/pydantic_ai/ARCH_02_SEQUENTIAL_PIPELINE/run.py"),
    Path("implementations/pydantic_ai/ARCH_03_ROUTER_SPECIALISTS/run.py"),
    Path("implementations/pydantic_ai/ARCH_04_SUPERVISOR_WORKERS/run.py"),
    Path("implementations/pydantic_ai/ARCH_05_HANDOFF_SWARM/run.py"),
    Path("implementations/pydantic_ai/ARCH_06_PARALLEL_FANOUT_FANIN/run.py"),
    Path("implementations/pydantic_ai/ARCH_07_MAP_REDUCE_AGENTIC/run.py"),
    Path("implementations/pydantic_ai/ARCH_08_DEBATE_JUDGE/run.py"),
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

        documents = (
            [
                DocumentInput(
                    document_id=f"doc-{index:03d}",
                    content=f"Benchmark evidence fragment {index}.",
                )
                for index in range(1, 8)
            ]
            if architecture == "ARCH_07_MAP_REDUCE_AGENTIC"
            else [
                DocumentInput(
                    document_id="doc-001",
                    content="This is a small benchmark document.",
                )
            ]
        )
        input_data = ExperimentInput(
            case_id="case-001",
            dataset_id="samples",
            task_type="qa",
            query="Summarize the benchmark document.",
            documents=documents,
            metadata=(
                {"map_reduce_batch_size": 3}
                if architecture == "ARCH_07_MAP_REDUCE_AGENTIC"
                else {}
            ),
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
        elif architecture == "ARCH_03_ROUTER_SPECIALISTS":
            assert result.status == RunStatus.SUCCESS
            selected_specialists = result.structured_output["selected_specialists"]
            skipped_specialists = result.structured_output["skipped_specialists"]
            expected_steps = [
                "router_routing",
                *selected_specialists,
                "router_synthesis",
            ]
            assert result.metrics.step_count == len(expected_steps)
            assert result.metrics.llm_call_count == len(expected_steps)
            assert [step.name for step in result.steps] == expected_steps
            assert "reasoning_specialist" in selected_specialists
            assert sorted(selected_specialists + skipped_specialists) == [
                "data_specialist",
                "reasoning_specialist",
                "validation_specialist",
            ]
            assert result.final_answer
        elif architecture == "ARCH_04_SUPERVISOR_WORKERS":
            assert result.status == RunStatus.SUCCESS
            structured_output = result.structured_output
            assert structured_output["supervisor_plan"]["workers_to_run"]
            assert structured_output["workers_executed"]
            assert structured_output["number_of_workers_executed"] == len(
                structured_output["workers_executed"]
            )
            assert structured_output["supervisor_iterations"] <= structured_output["max_supervisor_iterations"]
            assert "revisions_requested" in structured_output
            assert "stop_reason" in structured_output
            assert "warnings" in structured_output
            assert "accepted_worker_outputs" in structured_output
            assert "rejected_worker_outputs" in structured_output
            assert any(step.name == "supervisor_decision" for step in result.steps)
            assert any(step.step_type == "supervisor_review_llm_call" for step in result.steps)
            assert any(step.step_type == "worker_llm_call" for step in result.steps)
            assert structured_output["workers_executed"] != [
                "data_worker",
                "reasoning_worker",
                "validation_worker",
                "synthesis_worker",
            ]
            assert result.final_answer
        elif architecture == "ARCH_05_HANDOFF_SWARM":
            assert result.status == RunStatus.SUCCESS
            structured_output = result.structured_output
            assert structured_output["answer"]
            assert structured_output["decision"]
            assert "confidence" in structured_output
            assert "evidence" in structured_output
            assert "limitations" in structured_output
            assert structured_output["initial_agent"]
            assert structured_output["active_agent_history"]
            assert structured_output["handoff_history"]
            assert structured_output["number_of_handoffs"] == len(structured_output["handoff_history"])
            assert structured_output["number_of_handoffs"] <= structured_output["max_handoffs"]
            assert structured_output["number_of_agent_invocations"] <= structured_output["max_agent_invocations"]
            assert structured_output["unique_agents_executed"]
            assert structured_output["finalizing_agent"]
            assert "stop_reason" in structured_output
            assert structured_output["parallelism_used"] is False
            assert any(step.step_type == "handoff_agent_llm_call" for step in result.steps)
            assert all("supervisor" not in (step.name or "").lower() for step in result.steps)
            assert all("manager" not in (step.actor or "").lower() for step in result.steps)
            first_handoff = structured_output["handoff_history"][0]
            assert first_handoff["target_agent"] in structured_output["active_agent_history"]
            assert structured_output["active_agent_history"] != [
                "data_specialist",
                "reasoning_specialist",
                "validation_specialist",
                "synthesis_specialist",
            ]
            assert result.final_answer
        elif architecture == "ARCH_06_PARALLEL_FANOUT_FANIN":
            branches = [
                "factual_analysis_branch",
                "technical_reasoning_branch",
                "risk_constraints_branch",
                "alternative_solution_branch",
            ]
            assert result.status == RunStatus.SUCCESS
            assert result.metrics.step_count == 5
            assert result.metrics.llm_call_count == 5
            assert [step.name for step in result.steps] == [*branches, "aggregator"]
            assert set(result.structured_output["partial_outputs"]) == set(branches)
            assert result.structured_output["branches_completed"] == branches
            assert result.structured_output["branches_failed"] == []
            assert result.structured_output["parallelism_used"] is True
            assert result.structured_output["fallback_sequential"] is False
            assert result.metrics.metadata["latency_total_ms"] >= 0
            assert set(result.metrics.metadata["parallel_execution"]["branch_metrics"]) == set(branches)

            for step in result.steps[:4]:
                assert step.step_type == "parallel_branch_llm_call"
                assert step.input_data["depends_on"] == []
                assert "partial_outputs" not in step.input_data
                assert input_data.query in step.input_data["prompt"]
                assert "doc-001" in step.input_data["prompt"]

            aggregator_step = result.steps[-1]
            assert aggregator_step.step_type == "parallel_aggregator_llm_call"
            assert aggregator_step.input_data["depends_on"] == branches
            assert set(aggregator_step.input_data["partial_outputs"]) == set(branches)
            assert result.final_answer

            raw_path = (
                root
                / "results"
                / "raw"
                / framework
                / architecture
                / f"{config.run_id}.json"
            )
            assert raw_path.exists()
        elif architecture == "ARCH_07_MAP_REDUCE_AGENTIC":
            mapper_steps = [step for step in result.steps if step.step_type == "map_batch_llm_call"]
            expected_mapper_names = ["mapper_001", "mapper_002", "mapper_003"]
            execution = result.structured_output["map_reduce_execution"]

            assert result.status == RunStatus.SUCCESS
            assert [step.name for step in result.steps] == [
                "document_partitioner",
                *expected_mapper_names,
                "reducer",
            ]
            assert result.metrics.step_count == 5
            assert result.metrics.llm_call_count == 4
            assert result.steps[0].step_type == "document_partition"
            assert result.steps[0].output_data["batch_count"] == 3
            assert len(mapper_steps) == 3
            assert all(
                step.metadata["mapper_equivalence_group"] == "document_batch_mapper"
                for step in mapper_steps
            )
            mapped_document_ids = [
                document_id
                for step in mapper_steps
                for document_id in step.input_data["document_ids"]
            ]
            assert mapped_document_ids == [document.document_id for document in input_data.documents]
            assert len(mapped_document_ids) == len(set(mapped_document_ids))
            for step in mapper_steps:
                assert step.input_data["depends_on"] == ["document_partitioner"]
                assert step.input_data["component"] == "mapper"
                assert "perspective" not in step.input_data["prompt"].lower()
                assert all(
                    document_id in step.input_data["prompt"]
                    for document_id in step.input_data["document_ids"]
                )

            reducer_step = result.steps[-1]
            assert reducer_step.step_type == "reduce_llm_call"
            assert reducer_step.input_data["depends_on"] == expected_mapper_names
            assert reducer_step.input_data["original_documents_included"] is False
            assert set(reducer_step.input_data["partial_outputs"]) == {
                "batch_001",
                "batch_002",
                "batch_003",
            }
            assert execution["total_documents"] == 7
            assert execution["batch_size"] == 3
            assert execution["batch_count"] == 3
            assert execution["mapper_count"] == 3
            assert execution["batches_failed"] == []
            assert execution["parallelism_used"] is True
            assert execution["fallback_sequential"] is False
            assert execution["throughput_docs_per_second"] >= 0
            assert result.metrics.metadata["map_reduce_execution"] == execution
            assert result.final_answer

            raw_path = (
                root
                / "results"
                / "raw"
                / framework
                / architecture
                / f"{config.run_id}.json"
            )
            assert raw_path.exists()
        elif architecture == "ARCH_08_DEBATE_JUDGE":
            expected_steps = [
                "debater_a",
                "debater_b",
                "debater_c",
                "debate_round",
                "judge",
            ]
            structured_output = result.structured_output
            execution = structured_output["debate_execution"]

            assert result.status == RunStatus.SUCCESS
            assert result.metrics.step_count == 5
            assert result.metrics.llm_call_count == 5
            assert [step.name for step in result.steps] == expected_steps
            assert set(structured_output["proposals"]) == {
                "debater_a",
                "debater_b",
                "debater_c",
            }
            assert structured_output["number_of_proposals"] == 3
            assert structured_output["number_of_debate_rounds"] == 1
            assert structured_output["critique_count"] == 3
            assert structured_output["disagreement_count"] >= 1
            assert structured_output["decision_mode"] in {"select", "combine", "reject"}
            assert structured_output["judge"]["answer"] == result.final_answer
            assert structured_output["judge"]["rationale"]
            assert execution["proposal_count"] == 3
            assert execution["debate_round_count"] == 1
            assert set(execution["component_metrics"]) == set(expected_steps)
            assert result.metrics.metadata["debate_execution"] == execution

            for step, debater in zip(result.steps[:3], expected_steps[:3], strict=True):
                assert step.step_type == "debate_proposal_llm_call"
                assert step.name == debater
                assert step.input_data["depends_on"] == []
                assert "proposals" not in step.input_data
                assert input_data.query in step.input_data["prompt"]
                assert "doc-001" in step.input_data["prompt"]
                assert step.output_data["proposal"]["debater_name"] == debater
                assert step.output_data["proposal"]["proposal"]

            debate_step = result.steps[3]
            assert debate_step.step_type == "debate_round_llm_call"
            assert debate_step.input_data["depends_on"] == expected_steps[:3]
            assert set(debate_step.input_data["proposals"]) == set(expected_steps[:3])
            assert debate_step.metadata["debate_round_number"] == 1
            assert {
                critique["target_debater"]
                for critique in debate_step.output_data["debate_round"]["critiques"]
            } == set(expected_steps[:3])

            judge_step = result.steps[4]
            assert judge_step.step_type == "debate_judge_llm_call"
            assert judge_step.input_data["depends_on"] == expected_steps[:4]
            assert set(judge_step.input_data["proposals"]) == set(expected_steps[:3])
            assert judge_step.input_data["debate_round"]["round_number"] == 1
            assert judge_step.output_data["judge_decision"]["answer"] == result.final_answer

            raw_path = (
                root
                / "results"
                / "raw"
                / framework
                / architecture
                / f"{config.run_id}.json"
            )
            assert raw_path.exists()
        else:
            raise AssertionError(f"Unexpected architecture under test: {architecture}")


def test_arch04_supervisor_workers_are_not_fixed_pipeline() -> None:
    root = Path(__file__).resolve().parents[1]
    arch04_runners = [
        path for path in EXPECTED_RUNNERS if path.parts[2] == "ARCH_04_SUPERVISOR_WORKERS"
    ]

    documented_input = ExperimentInput(
        case_id="case-with-docs",
        dataset_id="samples",
        task_type="qa",
        query="Summarize the benchmark document.",
        documents=[
            DocumentInput(
                document_id="doc-001",
                content="This is a small benchmark document.",
            )
        ],
    )
    no_document_input = ExperimentInput(
        case_id="case-no-docs",
        dataset_id="samples",
        task_type="qa",
        query="Summarize without document context.",
        documents=[],
    )

    for relative_path in arch04_runners:
        framework = relative_path.parts[1]
        module = load_module(root / relative_path)

        documented_result = module.run_architecture(
            documented_input,
            ExperimentConfig(
                experiment_id="arch04-fixed-pipeline-test",
                framework=framework,
                architecture="ARCH_04_SUPERVISOR_WORKERS",
                model_provider="local",
                model_name="deterministic-local-v1",
                run_id=f"{framework}-docs",
            ),
        )
        no_document_result = module.run_architecture(
            no_document_input,
            ExperimentConfig(
                experiment_id="arch04-fixed-pipeline-test",
                framework=framework,
                architecture="ARCH_04_SUPERVISOR_WORKERS",
                model_provider="local",
                model_name="deterministic-local-v1",
                run_id=f"{framework}-no-docs",
            ),
        )

        documented_workers = documented_result.structured_output["workers_executed"]
        no_document_workers = no_document_result.structured_output["workers_executed"]
        assert documented_workers != no_document_workers
        assert "data_worker" in documented_workers
        assert "data_worker" not in no_document_workers
        assert no_document_result.structured_output["workers_not_used"]


def test_arch05_handoff_swarm_direct_finalize_fallback_and_cycle() -> None:
    root = Path(__file__).resolve().parents[1]
    arch05_runners = [
        path for path in EXPECTED_RUNNERS if path.parts[2] == "ARCH_05_HANDOFF_SWARM"
    ]

    direct_input = ExperimentInput(
        case_id="case-direct",
        dataset_id="samples",
        task_type="qa",
        query="Direct answer without document context.",
        documents=[],
    )
    invalid_input = ExperimentInput(
        case_id="case-invalid",
        dataset_id="samples",
        task_type="qa",
        query="Trigger invalid handoff decision.",
        documents=[DocumentInput(document_id="doc-001", content="Small document.")],
    )
    cycle_input = ExperimentInput(
        case_id="case-cycle",
        dataset_id="samples",
        task_type="qa",
        query="Trigger cycle handoff decision.",
        documents=[DocumentInput(document_id="doc-001", content="Small document.")],
    )

    for relative_path in arch05_runners:
        framework = relative_path.parts[1]
        module = load_module(root / relative_path)

        direct_result = module.run_architecture(
            direct_input,
            ExperimentConfig(
                experiment_id="arch05-direct-test",
                framework=framework,
                architecture="ARCH_05_HANDOFF_SWARM",
                model_provider="local",
                model_name="deterministic-local-v1",
                run_id=f"{framework}-arch05-direct",
            ),
        )
        assert direct_result.status == RunStatus.SUCCESS
        assert direct_result.structured_output["number_of_handoffs"] == 0
        assert direct_result.structured_output["stop_reason"] == "agent_finalized"

        invalid_result = module.run_architecture(
            invalid_input,
            ExperimentConfig(
                experiment_id="arch05-invalid-test",
                framework=framework,
                architecture="ARCH_05_HANDOFF_SWARM",
                model_provider="local",
                model_name="deterministic-local-v1",
                run_id=f"{framework}-arch05-invalid",
            ),
        )
        assert invalid_result.status == RunStatus.SUCCESS
        assert invalid_result.structured_output["fallback_used"] is True
        assert invalid_result.structured_output["warnings"]

        cycle_result = module.run_architecture(
            cycle_input,
            ExperimentConfig(
                experiment_id="arch05-cycle-test",
                framework=framework,
                architecture="ARCH_05_HANDOFF_SWARM",
                model_provider="local",
                model_name="deterministic-local-v1",
                run_id=f"{framework}-arch05-cycle",
                metadata={
                    "max_handoffs": 4,
                    "max_agent_invocations": 6,
                    "max_consecutive_visits_per_agent": 2,
                },
            ),
        )
        assert cycle_result.status == RunStatus.SUCCESS
        assert cycle_result.structured_output["cycle_detected"] is True
        assert cycle_result.structured_output["number_of_agent_invocations"] <= 6
        assert cycle_result.structured_output["number_of_handoffs"] <= 4


def test_arch05_handoff_swarm_dynamic_paths_and_limits() -> None:
    root = Path(__file__).resolve().parents[1]
    arch05_runners = [
        path for path in EXPECTED_RUNNERS if path.parts[2] == "ARCH_05_HANDOFF_SWARM"
    ]

    scenarios = {
        "one_handoff": ExperimentInput(
            case_id="case-one-handoff",
            dataset_id="samples",
            task_type="qa",
            query="Summarize without document context.",
            documents=[],
        ),
        "multi_handoff": ExperimentInput(
            case_id="case-multi-handoff",
            dataset_id="samples",
            task_type="qa",
            query="Summarize and validate possible risks.",
            documents=[DocumentInput(document_id="doc-001", content="Small document.")],
        ),
        "return_handoff": ExperimentInput(
            case_id="case-return",
            dataset_id="samples",
            task_type="qa",
            query="Trigger return handoff decision.",
            documents=[DocumentInput(document_id="doc-001", content="Small document.")],
        ),
        "limited": ExperimentInput(
            case_id="case-limit",
            dataset_id="samples",
            task_type="qa",
            query="Trigger cycle handoff decision.",
            documents=[DocumentInput(document_id="doc-001", content="Small document.")],
        ),
    }

    for relative_path in arch05_runners:
        framework = relative_path.parts[1]
        module = load_module(root / relative_path)

        one_result = module.run_architecture(
            scenarios["one_handoff"],
            ExperimentConfig(
                experiment_id="arch05-one-handoff-test",
                framework=framework,
                architecture="ARCH_05_HANDOFF_SWARM",
                model_provider="local",
                model_name="deterministic-local-v1",
                run_id=f"{framework}-arch05-one-handoff",
            ),
        )
        one_output = one_result.structured_output
        assert one_result.status == RunStatus.SUCCESS
        assert one_output["number_of_handoffs"] == 1
        assert one_output["handoff_history"][0]["source_agent"] == "reasoning_specialist"
        assert one_output["handoff_history"][0]["target_agent"] == "synthesis_specialist"
        assert one_output["active_agent_history"] == ["reasoning_specialist", "synthesis_specialist"]

        multi_result = module.run_architecture(
            scenarios["multi_handoff"],
            ExperimentConfig(
                experiment_id="arch05-multi-handoff-test",
                framework=framework,
                architecture="ARCH_05_HANDOFF_SWARM",
                model_provider="local",
                model_name="deterministic-local-v1",
                run_id=f"{framework}-arch05-multi-handoff",
            ),
        )
        multi_output = multi_result.structured_output
        assert multi_result.status == RunStatus.SUCCESS
        assert multi_output["active_agent_history"][:4] == [
            "data_specialist",
            "reasoning_specialist",
            "validation_specialist",
            "synthesis_specialist",
        ]
        assert multi_output["number_of_handoffs"] == 3
        assert "Validation" in multi_output["handoff_history"][2]["context_summary"]
        if framework == "microsoft_agent_framework":
            assert multi_output["native_handoff_events"] == [
                {
                    "source_agent": handoff["source_agent"],
                    "target_agent": handoff["target_agent"],
                }
                for handoff in multi_output["handoff_history"]
            ]

        return_result = module.run_architecture(
            scenarios["return_handoff"],
            ExperimentConfig(
                experiment_id="arch05-return-test",
                framework=framework,
                architecture="ARCH_05_HANDOFF_SWARM",
                model_provider="local",
                model_name="deterministic-local-v1",
                run_id=f"{framework}-arch05-return",
            ),
        )
        return_output = return_result.structured_output
        assert return_result.status == RunStatus.SUCCESS
        assert return_output["active_agent_history"][:3] == [
            "data_specialist",
            "reasoning_specialist",
            "data_specialist",
        ]
        assert return_output["handoff_history"][1]["source_agent"] == "reasoning_specialist"
        assert return_output["handoff_history"][1]["target_agent"] == "data_specialist"
        assert all("supervisor" not in step.actor.lower() for step in return_result.steps)

        limited_result = module.run_architecture(
            scenarios["limited"],
            ExperimentConfig(
                experiment_id="arch05-limit-test",
                framework=framework,
                architecture="ARCH_05_HANDOFF_SWARM",
                model_provider="local",
                model_name="deterministic-local-v1",
                run_id=f"{framework}-arch05-limit",
                metadata={
                    "max_handoffs": 1,
                    "max_agent_invocations": 3,
                    "max_consecutive_visits_per_agent": 2,
                },
            ),
        )
        limited_output = limited_result.structured_output
        assert limited_result.status == RunStatus.SUCCESS
        assert limited_output["number_of_handoffs"] <= 1
        assert limited_output["number_of_agent_invocations"] <= 3
        assert limited_output["stop_reason"] in {
            "max_handoffs_reached",
            "max_agent_invocations_reached",
            "max_consecutive_visits_per_agent_reached",
        }

        assert limited_result.metrics.llm_call_count == limited_output["number_of_agent_invocations"]
        assert limited_result.metrics.token_usage.total_tokens > 0
        assert all(call.latency_seconds is not None for call in limited_result.llm_calls)


def test_arch05_structural_constraints() -> None:
    root = Path(__file__).resolve().parents[1]
    arch05_runners = [
        path for path in EXPECTED_RUNNERS if path.parts[2] == "ARCH_05_HANDOFF_SWARM"
    ]

    for relative_path in arch05_runners:
        source = (root / relative_path).read_text(encoding="utf-8")
        assert "run_supervisor_workers_loop" not in source
        assert "ARCH_04_SUPERVISOR_WORKERS" not in source
        assert "Process.hierarchical" not in source
        assert "manager_agent" not in source
        assert "async_execution=True" not in source
        assert "allow_parallel" not in source
        assert "fanout" not in source.lower()

    common_source = (root / "implementations" / "supervisor_workers_common.py").read_text(encoding="utf-8")
    assert "ARCH_05_HANDOFF_SWARM" not in common_source


def test_arch05_framework_specific_native_primitives_are_audited() -> None:
    root = Path(__file__).resolve().parents[1]
    decisions = (root / "docs" / "decisions.md").read_text(encoding="utf-8")
    common_source = (root / "benchmark_core" / "handoff_swarm.py").read_text(encoding="utf-8")
    assert not re.search(r"while\s+.*stop_reason", common_source)
    assert "crew.kickoff" not in common_source
    assert "complete_agent_step" not in common_source

    langgraph_source = (root / "implementations/langgraph/ARCH_05_HANDOFF_SWARM/run.py").read_text(encoding="utf-8")
    assert "StateGraph" in langgraph_source
    assert "add_conditional_edges" in langgraph_source
    assert "END" in langgraph_source
    assert "for agent in HANDOFF_AGENTS" in langgraph_source
    assert "graph.add_node(agent" in langgraph_source

    microsoft_source = (root / "implementations/microsoft_agent_framework/ARCH_05_HANDOFF_SWARM/run.py").read_text(encoding="utf-8")
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert "HandoffBuilder" in microsoft_source
    assert "handoff_to_" in microsoft_source
    assert "native_handoff_events" in microsoft_source
    assert "agent-framework-orchestrations" in pyproject
    assert "agent-framework-orchestrations==1.0.0" in decisions

    llamaindex_source = (root / "implementations/llamaindex/ARCH_05_HANDOFF_SWARM/run.py").read_text(encoding="utf-8")
    assert "FunctionAgent" in llamaindex_source or "build_function_agent" in llamaindex_source
    assert "AgentWorkflow" in decisions
    assert "can_handoff_to" in decisions

    pydantic_source = (root / "implementations/pydantic_ai/ARCH_05_HANDOFF_SWARM/run.py").read_text(encoding="utf-8")
    assert "GraphBuilder" in pydantic_source
    assert "builder.step" in pydantic_source
    assert "builder.decision" in pydantic_source
    assert "builder.match" in pydantic_source
    assert "BaseNode" not in pydantic_source
    assert not re.search(r"from pydantic_graph import .*\bGraph\b", pydantic_source)
    assert not re.search(r"while\s+.*stop_reason", pydantic_source)

    crewai_source = (root / "implementations/crewai/ARCH_05_HANDOFF_SWARM/run.py").read_text(encoding="utf-8")
    assert "Flow[" in crewai_source
    assert "@listen" in crewai_source
    assert "@router" in crewai_source
    assert "Process.hierarchical" not in crewai_source
    assert "manager_agent" not in crewai_source


def test_arch06_structural_constraints_and_native_primitives() -> None:
    root = Path(__file__).resolve().parents[1]
    arch06_runners = [
        path for path in EXPECTED_RUNNERS if path.parts[2] == "ARCH_06_PARALLEL_FANOUT_FANIN"
    ]
    branches = [
        "factual_analysis_branch",
        "technical_reasoning_branch",
        "risk_constraints_branch",
        "alternative_solution_branch",
    ]

    assert len(arch06_runners) == 5
    for relative_path in arch06_runners:
        source = (root / relative_path).read_text(encoding="utf-8")
        assert "PARALLEL_BRANCHES" in source
        assert "aggregator" in source
        assert "ARCH_07" not in source
        assert "run_supervisor_workers_loop" not in source
        assert "Process.hierarchical" not in source
        assert "manager_agent" not in source
        assert "HandoffBuilder" not in source
        assert not re.search(r"while\s+", source)

    langgraph_source = (
        root / "implementations/langgraph/ARCH_06_PARALLEL_FANOUT_FANIN/run.py"
    ).read_text(encoding="utf-8")
    assert "StateGraph" in langgraph_source
    assert "START" in langgraph_source
    assert "graph.add_edge(list(PARALLEL_BRANCHES), AGGREGATOR)" in langgraph_source

    microsoft_source = (
        root
        / "implementations/microsoft_agent_framework/ARCH_06_PARALLEL_FANOUT_FANIN/run.py"
    ).read_text(encoding="utf-8")
    assert "WorkflowBuilder" in microsoft_source
    assert ".add_fan_out_edges(source, branches)" in microsoft_source
    assert ".add_fan_in_edges(branches, aggregator)" in microsoft_source

    crewai_source = (
        root / "implementations/crewai/ARCH_06_PARALLEL_FANOUT_FANIN/run.py"
    ).read_text(encoding="utf-8")
    assert "async_execution=True" in crewai_source
    assert "context=[]" in crewai_source
    assert "context=branch_tasks" in crewai_source
    assert "allow_delegation=False" in crewai_source

    llamaindex_source = (
        root / "implementations/llamaindex/ARCH_06_PARALLEL_FANOUT_FANIN/run.py"
    ).read_text(encoding="utf-8")
    assert "Workflow" in llamaindex_source
    assert "ctx.send_event" in llamaindex_source
    assert "ctx.collect_events" in llamaindex_source

    pydantic_source = (
        root / "implementations/pydantic_ai/ARCH_06_PARALLEL_FANOUT_FANIN/run.py"
    ).read_text(encoding="utf-8")
    assert "GraphBuilder" in pydantic_source
    assert "builder.join" in pydantic_source
    assert 'fork_id="perspective_fan_out"' in pydantic_source

    common_source = (
        root / "benchmark_core/parallel_fanout_fanin.py"
    ).read_text(encoding="utf-8")
    assert "implementations." not in common_source
    assert "asyncio.gather" not in common_source


def test_arch06_configuration_and_documentation_are_registered() -> None:
    root = Path(__file__).resolve().parents[1]
    experiments = (root / "configs/experiments.yaml").read_text(encoding="utf-8")
    decisions = (root / "docs/decisions.md").read_text(encoding="utf-8")
    spec = root / "docs/architecture_specs/ARCH_06_PARALLEL_FANOUT_FANIN.md"

    assert "ARCH_06_PARALLEL_FANOUT_FANIN" in experiments
    assert "ARCH_06_PARALLEL_FANOUT_FANIN" in decisions
    assert spec.exists()


def test_arch07_partitioning_is_deterministic_and_configurable() -> None:
    from benchmark_core.map_reduce_agentic import partition_documents

    input_data = ExperimentInput(
        case_id="arch07-partition-test",
        dataset_id="samples",
        task_type="qa",
        query="Map these documents.",
        documents=[
            DocumentInput(document_id=f"doc-{index:03d}", content=str(index))
            for index in range(1, 8)
        ],
    )
    config = ExperimentConfig(
        experiment_id="arch07-partition-test",
        run_id="arch07-partition-test",
        framework="langgraph",
        architecture="ARCH_07_MAP_REDUCE_AGENTIC",
        model_provider="local",
        model_name="deterministic-local-v1",
        metadata={"map_reduce_batch_size": 3},
    )

    first = partition_documents(input_data, config)
    second = partition_documents(input_data, config)
    assert [batch.model_dump() for batch in first] == [batch.model_dump() for batch in second]
    assert [batch.document_ids for batch in first] == [
        ["doc-001", "doc-002", "doc-003"],
        ["doc-004", "doc-005", "doc-006"],
        ["doc-007"],
    ]


def test_arch07_structural_constraints_and_native_primitives() -> None:
    root = Path(__file__).resolve().parents[1]
    runners = [
        path for path in EXPECTED_RUNNERS if path.parts[2] == "ARCH_07_MAP_REDUCE_AGENTIC"
    ]
    assert len(runners) == 5
    for relative_path in runners:
        source = (root / relative_path).read_text(encoding="utf-8")
        assert "partition_documents" in source
        assert "render_map_reduce_prompt" in source
        assert "make_mapper_step" in source
        assert "make_reducer_step" in source
        assert "ARCH_08" not in source
        assert "Process.hierarchical" not in source
        assert "HandoffBuilder" not in source
        assert "router" not in source.lower()
        assert "supervisor" not in source.lower()

    langgraph_source = (root / runners[0]).read_text(encoding="utf-8")
    assert "StateGraph" in langgraph_source
    assert "Send(MAPPER" in langgraph_source
    assert "add_conditional_edges" in langgraph_source

    crewai_source = (root / runners[1]).read_text(encoding="utf-8")
    assert "async_execution=True" in crewai_source
    assert "context=[]" in crewai_source
    assert "context=mapper_tasks" in crewai_source
    assert "allow_delegation=False" in crewai_source

    microsoft_source = (root / runners[2]).read_text(encoding="utf-8")
    assert "WorkflowBuilder" in microsoft_source
    assert ".add_fan_out_edges(partitioner, mappers)" in microsoft_source
    assert ".add_fan_in_edges(mappers, reducer)" in microsoft_source

    llamaindex_source = (root / runners[3]).read_text(encoding="utf-8")
    assert "MapperEvent" in llamaindex_source
    assert "num_workers=max(len(batches), 1)" in llamaindex_source
    assert "ctx.collect_events" in llamaindex_source

    pydantic_source = (root / runners[4]).read_text(encoding="utf-8")
    assert "GraphBuilder" in pydantic_source
    assert '.map(fork_id="document_batch_map"' in pydantic_source
    assert "builder.join" in pydantic_source

    common_source = (root / "benchmark_core/map_reduce_agentic.py").read_text(encoding="utf-8")
    assert "implementations." not in common_source
    assert "asyncio.gather" not in common_source


def test_arch07_configuration_and_documentation_are_registered() -> None:
    root = Path(__file__).resolve().parents[1]
    experiments = (root / "configs/experiments.yaml").read_text(encoding="utf-8")
    decisions = (root / "docs/decisions.md").read_text(encoding="utf-8")
    spec = root / "docs/architecture_specs/ARCH_07_MAP_REDUCE_AGENTIC.md"

    assert "ARCH_07_MAP_REDUCE_AGENTIC" in experiments
    assert "ARCH_07_MAP_REDUCE_AGENTIC" in decisions
    assert spec.exists()


def test_arch08_structural_constraints_and_native_primitives() -> None:
    root = Path(__file__).resolve().parents[1]
    runners = [
        path for path in EXPECTED_RUNNERS if path.parts[2] == "ARCH_08_DEBATE_JUDGE"
    ]
    assert len(runners) == 5
    for relative_path in runners:
        source = (root / relative_path).read_text(encoding="utf-8")
        assert "DEBATERS" in source
        assert "DEBATE_ROUND" in source
        assert "JUDGE" in source
        assert "render_debate_judge_prompt" in source
        assert "make_debate_step" in source
        assert "Process.hierarchical" not in source
        assert "manager_agent" not in source
        assert "HandoffBuilder" not in source
        assert "partition_documents" not in source
        assert "router" not in source.lower()
        assert "supervisor" not in source.lower()
        assert not re.search(r"while\s+", source)

    langgraph_source = (root / runners[0]).read_text(encoding="utf-8")
    assert "StateGraph" in langgraph_source
    assert "graph.add_edge(list(DEBATERS), DEBATE_ROUND)" in langgraph_source
    assert "graph.add_edge(DEBATE_ROUND, JUDGE)" in langgraph_source

    crewai_source = (root / runners[1]).read_text(encoding="utf-8")
    assert "async_execution=True" in crewai_source
    assert "context=[]" in crewai_source
    assert "context=proposal_tasks" in crewai_source
    assert "context=[*proposal_tasks, debate_task]" in crewai_source
    assert "allow_delegation=False" in crewai_source

    microsoft_source = (root / runners[2]).read_text(encoding="utf-8")
    assert "WorkflowBuilder" in microsoft_source
    assert ".add_fan_out_edges(source, debaters)" in microsoft_source
    assert ".add_fan_in_edges(debaters, debate_round)" in microsoft_source
    assert ".add_edge(debate_round, judge)" in microsoft_source

    llamaindex_source = (root / runners[3]).read_text(encoding="utf-8")
    assert "Workflow" in llamaindex_source
    assert "ctx.send_event" in llamaindex_source
    assert "ctx.collect_events" in llamaindex_source
    assert "async def debate_round" in llamaindex_source
    assert "async def judge" in llamaindex_source

    pydantic_source = (root / runners[4]).read_text(encoding="utf-8")
    assert "GraphBuilder" in pydantic_source
    assert "builder.join" in pydantic_source
    assert 'fork_id="debater_fan_out"' in pydantic_source
    assert "DebateProposal" in pydantic_source
    assert "DebateRoundOutput" in pydantic_source
    assert "JudgeDecision" in pydantic_source

    common_source = (root / "benchmark_core/debate_judge.py").read_text(encoding="utf-8")
    assert "implementations." not in common_source
    assert "asyncio.gather" not in common_source


def test_arch08_configuration_and_documentation_are_registered() -> None:
    root = Path(__file__).resolve().parents[1]
    experiments = (root / "configs/experiments.yaml").read_text(encoding="utf-8")
    decisions = (root / "docs/decisions.md").read_text(encoding="utf-8")
    spec = root / "docs/architecture_specs/ARCH_08_DEBATE_JUDGE.md"

    assert "ARCH_08_DEBATE_JUDGE" in experiments
    assert "ARCH_08_DEBATE_JUDGE" in decisions
    assert spec.exists()
