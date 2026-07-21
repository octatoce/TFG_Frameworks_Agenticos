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
    Path("implementations/crewai/ARCH_01_SINGLE_REACT/run.py"),
    Path("implementations/crewai/ARCH_02_SEQUENTIAL_PIPELINE/run.py"),
    Path("implementations/crewai/ARCH_03_ROUTER_SPECIALISTS/run.py"),
    Path("implementations/crewai/ARCH_04_SUPERVISOR_WORKERS/run.py"),
    Path("implementations/crewai/ARCH_05_HANDOFF_SWARM/run.py"),
    Path("implementations/microsoft_agent_framework/ARCH_01_SINGLE_REACT/run.py"),
    Path("implementations/microsoft_agent_framework/ARCH_02_SEQUENTIAL_PIPELINE/run.py"),
    Path("implementations/microsoft_agent_framework/ARCH_03_ROUTER_SPECIALISTS/run.py"),
    Path("implementations/microsoft_agent_framework/ARCH_04_SUPERVISOR_WORKERS/run.py"),
    Path("implementations/microsoft_agent_framework/ARCH_05_HANDOFF_SWARM/run.py"),
    Path("implementations/llamaindex/ARCH_01_SINGLE_REACT/run.py"),
    Path("implementations/llamaindex/ARCH_02_SEQUENTIAL_PIPELINE/run.py"),
    Path("implementations/llamaindex/ARCH_03_ROUTER_SPECIALISTS/run.py"),
    Path("implementations/llamaindex/ARCH_04_SUPERVISOR_WORKERS/run.py"),
    Path("implementations/llamaindex/ARCH_05_HANDOFF_SWARM/run.py"),
    Path("implementations/pydantic_ai/ARCH_01_SINGLE_REACT/run.py"),
    Path("implementations/pydantic_ai/ARCH_02_SEQUENTIAL_PIPELINE/run.py"),
    Path("implementations/pydantic_ai/ARCH_03_ROUTER_SPECIALISTS/run.py"),
    Path("implementations/pydantic_ai/ARCH_04_SUPERVISOR_WORKERS/run.py"),
    Path("implementations/pydantic_ai/ARCH_05_HANDOFF_SWARM/run.py"),
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
    assert "BaseNode" in pydantic_source
    assert "Graph(" in pydantic_source
    assert "End(" in pydantic_source
    assert not re.search(r"while\s+.*stop_reason", pydantic_source)

    crewai_source = (root / "implementations/crewai/ARCH_05_HANDOFF_SWARM/run.py").read_text(encoding="utf-8")
    assert "Flow[" in crewai_source
    assert "@listen" in crewai_source
    assert "@router" in crewai_source
    assert "Process.hierarchical" not in crewai_source
    assert "manager_agent" not in crewai_source
