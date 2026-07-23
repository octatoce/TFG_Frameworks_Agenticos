"""Importable LlamaIndex event types used by serialized ARCH_10 contexts."""

from typing import Any

from llama_index.core.workflow import Event


class InitializedEvent(Event):
    payload: dict[str, Any]


class PlannedEvent(Event):
    payload: dict[str, Any]


class CheckpointReadyEvent(Event):
    payload: dict[str, Any]


class RecoveredEvent(Event):
    payload: dict[str, Any]


class ContinuedEvent(Event):
    payload: dict[str, Any]
