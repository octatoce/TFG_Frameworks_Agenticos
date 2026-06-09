"""Common execution contract for benchmark architectures."""

from __future__ import annotations

from typing import Protocol

from benchmark_core.schemas import ExperimentConfig, ExperimentInput, ExperimentResult


class ArchitectureRunner(Protocol):
    """Callable contract every architecture module must expose."""

    def __call__(
        self,
        input_data: ExperimentInput,
        config: ExperimentConfig,
    ) -> ExperimentResult:
        ...


def run_with_contract(
    runner: ArchitectureRunner,
    input_data: ExperimentInput,
    config: ExperimentConfig,
) -> ExperimentResult:
    """Execute a runner through the shared interface."""

    return runner(input_data, config)
