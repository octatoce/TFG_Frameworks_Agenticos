from types import SimpleNamespace

from benchmark_core.llm_wrapper import OpenAIInstrumentedLLM, parse_worker_selection


def test_openai_instrumented_llm_extracts_usage() -> None:
    llm = OpenAIInstrumentedLLM(model_name="test-model", env_file=None)
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            input_tokens_details=SimpleNamespace(cached_tokens=2),
            output_tokens_details=SimpleNamespace(reasoning_tokens=1),
        )
    )

    usage = llm._extract_token_usage(response)

    assert usage.input_tokens == 10
    assert usage.output_tokens == 5
    assert usage.cached_input_tokens == 2
    assert usage.reasoning_tokens == 1
    assert usage.total_tokens == 15


def test_parse_worker_selection_from_supervisor_output() -> None:
    selected, skipped = parse_worker_selection(
        "\n".join(
            [
                "SELECTED_WORKERS=data_worker, reasoning_worker",
                "SKIPPED_WORKERS=validation_worker",
                "RATIONALE=Simple evidence question.",
            ]
        )
    )

    assert selected == ["data_worker", "reasoning_worker"]
    assert skipped == ["validation_worker"]
