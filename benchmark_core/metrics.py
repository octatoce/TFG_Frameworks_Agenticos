"""Small metric helpers shared by all framework implementations."""

from __future__ import annotations

from datetime import datetime

from benchmark_core.schemas import (
    AgentStep,
    ExperimentError,
    ExperimentMetrics,
    LLMCallMetrics,
    ResourceUsage,
    TokenUsage,
)


def compute_latency_seconds(started_at: datetime, finished_at: datetime) -> float:
    """Return total wall-clock latency in seconds."""

    return max((finished_at - started_at).total_seconds(), 0.0)


def count_steps(steps: list[AgentStep]) -> int:
    """Count recorded agent steps."""

    return len(steps)


def count_llm_calls(steps: list[AgentStep]) -> int:
    """Count LLM calls from trace steps."""

    return sum(len(step.llm_call_ids) for step in steps)


def build_token_usage(input_tokens: int = 0, output_tokens: int = 0) -> TokenUsage:
    """Build a normalized token usage object."""

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


def aggregate_token_usage(llm_calls: list[LLMCallMetrics]) -> TokenUsage:
    """Aggregate token usage from all recorded LLM calls."""

    input_tokens = sum(call.token_usage.input_tokens for call in llm_calls)
    output_tokens = sum(call.token_usage.output_tokens for call in llm_calls)
    cached_input_tokens = sum(call.token_usage.cached_input_tokens for call in llm_calls)
    reasoning_tokens = sum(call.token_usage.reasoning_tokens for call in llm_calls)
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=input_tokens + output_tokens + cached_input_tokens + reasoning_tokens,
    )


def count_errors(errors: list[ExperimentError], steps: list[AgentStep] | None = None) -> int:
    """Count explicit errors and step-level errors."""

    step_errors = 0 if steps is None else sum(1 for step in steps if step.error)
    return len(errors) + step_errors


def estimate_cost_usd(
    token_usage: TokenUsage,
    input_cost_per_1k: float = 0.0,
    output_cost_per_1k: float = 0.0,
) -> float:
    """Estimate run cost. Defaults to zero until model prices are configured."""

    input_cost = token_usage.input_tokens * input_cost_per_1k / 1000
    output_cost = token_usage.output_tokens * output_cost_per_1k / 1000
    return round(input_cost + output_cost, 8)


def build_metrics(
    started_at: datetime,
    finished_at: datetime,
    steps: list[AgentStep],
    errors: list[ExperimentError],
    llm_calls: list[LLMCallMetrics] | None = None,
    estimated_cost_usd: float | None = None,
    resource_usage: ResourceUsage | None = None,
) -> ExperimentMetrics:
    """Build a comparable metric bundle for one experiment result."""

    recorded_llm_calls = llm_calls or []
    token_usage = aggregate_token_usage(recorded_llm_calls)
    cost = (
        sum(call.estimated_cost_usd for call in recorded_llm_calls)
        if estimated_cost_usd is None
        else estimated_cost_usd
    )
    return ExperimentMetrics(
        total_latency_seconds=compute_latency_seconds(started_at, finished_at),
        llm_latency_seconds=sum(call.latency_seconds for call in recorded_llm_calls),
        step_count=count_steps(steps),
        llm_call_count=len(recorded_llm_calls),
        token_usage=token_usage,
        estimated_cost_usd=cost,
        error_count=count_errors(errors, steps),
        resource_usage=resource_usage or ResourceUsage(),
    )
