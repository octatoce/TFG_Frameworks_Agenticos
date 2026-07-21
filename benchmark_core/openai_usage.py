"""Strict normalization of token usage reported by OpenAI-backed SDKs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from benchmark_core.schemas import TokenUsage


class OpenAIUsageMissingError(ValueError):
    """Raised when a real OpenAI response does not expose provider usage."""


def _value(source: Any, name: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


def _first_value(source: Any, *names: str) -> Any:
    for name in names:
        value = _value(source, name)
        if value is not None:
            return value
    return None


def normalize_openai_token_usage(usage: Any) -> TokenUsage:
    """Normalize Responses, Chat Completions, or Pydantic AI usage fields."""

    if usage is None:
        raise OpenAIUsageMissingError("OpenAI response did not include a usage object.")
    input_tokens = _first_value(usage, "input_tokens", "prompt_tokens")
    output_tokens = _first_value(usage, "output_tokens", "completion_tokens")
    if input_tokens is None or output_tokens is None:
        raise OpenAIUsageMissingError(
            f"Unsupported OpenAI usage payload: {type(usage).__name__}."
        )
    input_tokens = int(input_tokens or 0)
    output_tokens = int(output_tokens or 0)
    total_tokens = _first_value(usage, "total_tokens")
    total_tokens = int(total_tokens if total_tokens is not None else input_tokens + output_tokens)

    input_details = _first_value(usage, "input_tokens_details", "prompt_tokens_details")
    output_details = _first_value(usage, "output_tokens_details", "completion_tokens_details")
    cached_input_tokens = _first_value(usage, "cache_read_tokens")
    if cached_input_tokens is None and input_details is not None:
        cached_input_tokens = _first_value(input_details, "cached_tokens")
    reasoning_tokens = None
    if output_details is not None:
        reasoning_tokens = _first_value(output_details, "reasoning_tokens")
    details = _first_value(usage, "details")
    if reasoning_tokens is None and details is not None:
        reasoning_tokens = _first_value(details, "reasoning_tokens")

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=int(cached_input_tokens or 0),
        reasoning_tokens=int(reasoning_tokens or 0),
        total_tokens=total_tokens,
    )


def extract_openai_token_usage(source: Any) -> TokenUsage:
    """Find and normalize provider usage in common framework result wrappers."""

    queue: list[tuple[Any, int]] = [(source, 0)]
    seen: set[int] = set()
    while queue:
        current, depth = queue.pop(0)
        if current is None or id(current) in seen or depth > 4:
            continue
        seen.add(id(current))

        if _first_value(current, "input_tokens", "prompt_tokens") is not None:
            return normalize_openai_token_usage(current)

        for name in ("usage", "raw_representation", "raw", "response", "additional_kwargs"):
            candidate = _value(current, name)
            if (
                callable(candidate)
                and name == "usage"
                and _first_value(candidate, "input_tokens", "prompt_tokens") is None
            ):
                candidate = candidate()
            if candidate is not None:
                queue.append((candidate, depth + 1))

    raise OpenAIUsageMissingError(
        f"No OpenAI usage payload found in {type(source).__name__}; refusing proxy token counts."
    )
