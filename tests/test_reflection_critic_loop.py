from benchmark_core.reflection_critic_loop import (
    CRITIC,
    GENERATOR,
    REVISER,
    CritiqueEvaluation,
    ReflectionSettings,
    build_deterministic_reflection_output,
    detect_reflection_component,
    evaluate_stop,
    get_reflection_settings,
    parse_critique,
    parse_reflection_version,
    render_reflection_prompt,
)
from benchmark_core.schemas import DocumentInput, ExperimentConfig, ExperimentInput


def sample_input() -> ExperimentInput:
    return ExperimentInput(
        case_id="reflection-unit",
        dataset_id="samples",
        task_type="qa",
        query="What conclusion is supported?",
        documents=[DocumentInput(document_id="doc-001", content="Evidence fragment.")],
        metadata={"ambiguity": "medium"},
    )


def sample_config(**metadata) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="reflection-unit",
        framework="langgraph",
        architecture="ARCH_09_REFLECTION_CRITIC_LOOP",
        model_provider="local",
        model_name="deterministic-local-v1",
        run_id="reflection-unit",
        max_agent_iterations=4,
        metadata=metadata,
    )


def test_settings_default_alias_cap_and_validation() -> None:
    assert get_reflection_settings(sample_config()).max_iterations == 2
    assert get_reflection_settings(sample_config(max_iterations=3)).max_iterations == 3
    capped = get_reflection_settings(sample_config(reflection_max_iterations=20))
    assert capped.max_iterations == 4

    try:
        get_reflection_settings(sample_config(reflection_max_iterations=0))
    except ValueError as exc:
        assert "at least 1" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("A zero iteration limit must be rejected")


def test_prompts_are_detectable_and_intermediates_are_typed() -> None:
    input_data = sample_input()
    settings = ReflectionSettings(max_iterations=2, quality_threshold=0.85)
    generator_prompt = render_reflection_prompt(
        input_data,
        GENERATOR,
        iteration=0,
        settings=settings,
    )
    assert detect_reflection_component(generator_prompt) == (GENERATOR, 0)

    initial_output = build_deterministic_reflection_output(input_data, GENERATOR, 0)
    initial = parse_reflection_version(
        initial_output,
        version_index=0,
        iteration=0,
        created_by=GENERATOR,
    )
    critic_prompt = render_reflection_prompt(
        input_data,
        CRITIC,
        iteration=1,
        settings=settings,
        current_version=initial.model_dump(),
    )
    assert detect_reflection_component(critic_prompt) == (CRITIC, 1)

    critique = parse_critique(
        build_deterministic_reflection_output(input_data, CRITIC, 1),
        iteration=1,
    )
    reviser_prompt = render_reflection_prompt(
        input_data,
        REVISER,
        iteration=1,
        settings=settings,
        current_version=initial.model_dump(),
        critique=critique.model_dump(),
    )
    assert detect_reflection_component(reviser_prompt) == (REVISER, 1)
    revised = parse_reflection_version(
        build_deterministic_reflection_output(input_data, REVISER, 1),
        version_index=1,
        iteration=1,
        created_by=REVISER,
    )
    assert revised.answer != initial.answer
    assert revised.changes_applied


def test_stop_policy_is_deterministic_and_bounded() -> None:
    settings = ReflectionSettings(max_iterations=2, quality_threshold=0.85)
    major = CritiqueEvaluation(
        iteration=1,
        no_critical_issues=False,
        score=0.40,
        severity="major",
        issues=["unsupported claim"],
        improvements=["add evidence"],
        format_valid=True,
        raw_output="test",
    )
    assert evaluate_stop(major, current_version_index=0, settings=settings).stop_reason == (
        "continue_revision"
    )

    at_limit = major.model_copy(update={"iteration": 2})
    max_decision = evaluate_stop(at_limit, current_version_index=1, settings=settings)
    assert max_decision.should_stop is True
    assert max_decision.stop_reason == "max_iterations_reached"
    assert max_decision.stopped_by_max_iterations is True

    sufficient = major.model_copy(
        update={"iteration": 2, "no_critical_issues": True, "score": 0.93, "severity": "none"}
    )
    quality_decision = evaluate_stop(sufficient, current_version_index=1, settings=settings)
    assert quality_decision.stop_reason == "quality_sufficient"
    assert quality_decision.stopped_by_quality is True
    assert quality_decision.stopped_by_max_iterations is False
