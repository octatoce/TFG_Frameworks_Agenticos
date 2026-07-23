import json

import pytest

from benchmark_core.checkpoint_memory_recovery import (
    CHECKPOINT_STAGE,
    CONTINUATION_STEP,
    PLANNING_STEP,
    ControlledFailure,
    build_deterministic_recovery_output,
    build_portable_checkpoint,
    controlled_failure_message,
    detect_recovery_component,
    get_recovery_settings,
    initialize_recovery_state,
    load_portable_checkpoint,
    logical_checkpoint_id,
    parse_planning_analysis,
    render_recovery_prompt,
    seal_state_for_checkpoint,
    verify_recovered_state,
    write_portable_checkpoint,
)
from benchmark_core.schemas import DocumentInput, ExperimentConfig, ExperimentInput
from benchmark_core.tracing import utc_now


def sample_input() -> ExperimentInput:
    return ExperimentInput(
        case_id="recovery-unit",
        dataset_id="samples",
        task_type="qa",
        query="What should survive recovery?",
        documents=[DocumentInput(document_id="doc-001", content="Persist this evidence.")],
        metadata={"failure_mode": "controlled"},
    )


def sample_config(**metadata) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="recovery-unit",
        framework="crewai",
        architecture="ARCH_10_CHECKPOINT_MEMORY_RECOVERY",
        model_provider="local",
        model_name="deterministic-local-v1",
        run_id="recovery/unit",
        metadata=metadata,
    )


def test_recovery_settings_and_prompts_are_explicit() -> None:
    assert get_recovery_settings(sample_config()).inject_failure is True
    assert get_recovery_settings(sample_config(checkpoint_inject_failure="false")).inject_failure is False
    with pytest.raises(ValueError, match="boolean-like"):
        get_recovery_settings(sample_config(checkpoint_inject_failure="sometimes"))

    planning_prompt = render_recovery_prompt(sample_input(), PLANNING_STEP)
    assert detect_recovery_component(planning_prompt) == PLANNING_STEP
    planning = parse_planning_analysis(
        build_deterministic_recovery_output(sample_input(), PLANNING_STEP)
    )
    continuation_prompt = render_recovery_prompt(
        sample_input(), CONTINUATION_STEP, planning=planning
    )
    assert detect_recovery_component(continuation_prompt) == CONTINUATION_STEP


def test_portable_checkpoint_round_trip_and_integrity(tmp_path) -> None:
    input_data = sample_input()
    config = sample_config()
    state = initialize_recovery_state(input_data)
    planning = parse_planning_analysis(
        build_deterministic_recovery_output(input_data, PLANNING_STEP)
    )
    state = state.model_copy(update={"planning": planning, "current_stage": PLANNING_STEP})
    created_at = utc_now()
    state = seal_state_for_checkpoint(
        state,
        checkpoint_id=logical_checkpoint_id(config),
        created_at=created_at,
    )
    checkpoint = build_portable_checkpoint(
        framework=config.framework,
        config=config,
        state=state,
        created_at=created_at,
    )
    path = tmp_path / "checkpoint.json"
    write_portable_checkpoint(path, checkpoint)

    loaded = load_portable_checkpoint(path)
    assert loaded.checkpoint_stage == CHECKPOINT_STAGE
    assert loaded.state.planning == planning
    assert verify_recovered_state(loaded.state) == loaded.state_digest

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["state"]["query"] = "tampered"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        load_portable_checkpoint(path)


def test_controlled_failure_is_named_and_recoverable_by_design() -> None:
    message = controlled_failure_message("checkpoint-001")
    assert "ControlledFailure" in message
    assert "checkpoint-001" in message
    assert issubclass(ControlledFailure, RuntimeError)
