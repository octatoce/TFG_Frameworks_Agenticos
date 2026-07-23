from types import SimpleNamespace
from pathlib import Path

import pytest

from benchmark_core.openai_usage import (
    OpenAIUsageMissingError,
    extract_openai_token_usage,
    normalize_openai_token_usage,
)


def test_normalizes_openai_responses_usage() -> None:
    usage = SimpleNamespace(
        input_tokens=100,
        output_tokens=20,
        total_tokens=120,
        input_tokens_details=SimpleNamespace(cached_tokens=10),
        output_tokens_details=SimpleNamespace(reasoning_tokens=4),
    )

    assert normalize_openai_token_usage(usage).model_dump() == {
        "input_tokens": 100,
        "output_tokens": 20,
        "cached_input_tokens": 10,
        "reasoning_tokens": 4,
        "total_tokens": 120,
    }


def test_normalizes_openai_chat_completions_usage() -> None:
    usage = {
        "prompt_tokens": 80,
        "completion_tokens": 15,
        "total_tokens": 95,
        "prompt_tokens_details": {"cached_tokens": 6},
        "completion_tokens_details": {"reasoning_tokens": 3},
    }

    normalized = normalize_openai_token_usage(usage)

    assert normalized.input_tokens == 80
    assert normalized.output_tokens == 15
    assert normalized.cached_input_tokens == 6
    assert normalized.reasoning_tokens == 3
    assert normalized.total_tokens == 95


def test_extracts_pydantic_ai_run_usage() -> None:
    usage = SimpleNamespace(
        input_tokens=70,
        output_tokens=12,
        cache_read_tokens=5,
        details={"reasoning_tokens": 2},
    )
    result = SimpleNamespace(usage=lambda: usage)

    normalized = extract_openai_token_usage(result)

    assert normalized.input_tokens == 70
    assert normalized.output_tokens == 12
    assert normalized.cached_input_tokens == 5
    assert normalized.reasoning_tokens == 2
    assert normalized.total_tokens == 82


def test_extracts_usage_from_framework_raw_wrapper() -> None:
    result = SimpleNamespace(
        raw={
            "usage": {
                "prompt_tokens": 40,
                "completion_tokens": 8,
                "total_tokens": 48,
            }
        }
    )

    assert extract_openai_token_usage(result).total_tokens == 48


def test_missing_provider_usage_is_not_replaced_by_proxy_count() -> None:
    with pytest.raises(OpenAIUsageMissingError, match="refusing proxy token counts"):
        extract_openai_token_usage(SimpleNamespace(output="response without usage"))


def test_pydantic_ai_pins_the_openai_api_surface() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "implementations/pydantic_ai/utils_pydantic_ai.py").read_text(
        encoding="utf-8"
    )

    assert 'f"openai-chat:{config.model_name}"' in source
    assert 'f"openai:{config.model_name}"' not in source


def test_arch_01_to_05_route_calls_through_instrumented_adapters() -> None:
    root = Path(__file__).resolve().parents[1]
    expected_entrypoint = {
        "langgraph": "complete_llm_step(",
        "crewai": "context.crewai_llm",
        "microsoft_agent_framework": "complete_agent_step(",
        "llamaindex": "complete_agent_step(",
        "pydantic_ai": "complete_agent_step(",
    }

    for framework, marker in expected_entrypoint.items():
        architecture_dirs = sorted(
            path
            for path in (root / "implementations" / framework).glob("ARCH_0[1-5]_*")
            if path.is_dir()
        )
        assert len(architecture_dirs) == 5
        for architecture_dir in architecture_dirs:
            source = (architecture_dir / "run.py").read_text(encoding="utf-8")
            expected = (
                "context.llm.complete("
                if framework == "microsoft_agent_framework"
                and architecture_dir.name.startswith("ARCH_05_")
                else marker
            )
            assert expected in source, f"{framework}/{architecture_dir.name} bypasses {expected}"
