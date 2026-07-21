from benchmark_core.debate_judge import (
    DEBATERS,
    DEBATE_ROUND,
    JUDGE,
    build_deterministic_debate_output,
    detect_debate_component,
    parse_debate_proposal,
    parse_debate_round,
    parse_judge_decision,
    render_debate_judge_prompt,
)
from benchmark_core.schemas import DocumentInput, ExperimentInput


def sample_input() -> ExperimentInput:
    return ExperimentInput(
        case_id="arch08-unit",
        dataset_id="samples",
        task_type="ambiguous_qa",
        query="Choose a robust interpretation.",
        documents=[DocumentInput(document_id="doc-001", content="Bounded evidence.")],
        metadata={"ambiguity": "intentional"},
    )


def test_arch08_prompts_and_deterministic_outputs_are_structured() -> None:
    input_data = sample_input()
    proposals = {}
    for debater in DEBATERS:
        prompt = render_debate_judge_prompt(input_data, debater)
        assert detect_debate_component(prompt) == debater
        proposal = parse_debate_proposal(
            debater,
            build_deterministic_debate_output(input_data, debater),
        )
        assert proposal.debater_name == debater
        assert proposal.proposal
        proposals[debater] = proposal.model_dump()

    debate_prompt = render_debate_judge_prompt(
        input_data,
        DEBATE_ROUND,
        proposals=proposals,
    )
    assert detect_debate_component(debate_prompt) == DEBATE_ROUND
    debate = parse_debate_round(
        build_deterministic_debate_output(input_data, DEBATE_ROUND)
    )
    assert len(debate.critiques) == 3
    assert len(debate.disagreements) >= 1

    judge_prompt = render_debate_judge_prompt(
        input_data,
        JUDGE,
        proposals=proposals,
        debate=debate.model_dump(),
    )
    assert detect_debate_component(judge_prompt) == JUDGE
    decision = parse_judge_decision(build_deterministic_debate_output(input_data, JUDGE))
    assert decision.decision_mode == "combine"
    assert decision.selected_proposals == list(DEBATERS)
    assert decision.answer
