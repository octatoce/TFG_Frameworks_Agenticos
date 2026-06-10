"""Tracing helpers for architecture implementations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from benchmark_core.schemas import AgentStep


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


def make_step(
    step_id: int,
    name: str,
    step_type: str,
    input_data: dict[str, Any] | None = None,
    output_data: dict[str, Any] | None = None,
    llm_call_ids: list[str] | None = None,
    actor: str | None = None,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentStep:
    """Create a normalized step trace entry."""

    now = utc_now()
    return AgentStep(
        step_id=step_id,
        name=name,
        step_type=step_type,
        actor=actor,
        input_data=input_data or {},
        output_data=output_data or {},
        llm_call_ids=llm_call_ids or [],
        started_at=now,
        finished_at=now,
        error=error,
        metadata=metadata or {},
    )
