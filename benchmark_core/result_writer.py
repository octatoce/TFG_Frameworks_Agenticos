"""Result persistence utilities."""

from __future__ import annotations

import json
from pathlib import Path

from benchmark_core.schemas import ExperimentResult


def save_result_json(
    result: ExperimentResult,
    base_dir: str | Path = "results/raw",
) -> Path:
    """Save one experiment result as JSON under the canonical raw path."""

    output_dir = Path(base_dir) / result.framework / result.architecture
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{result.run_id}.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result.model_dump(mode="json"), file, ensure_ascii=False, indent=2)

    return output_path
