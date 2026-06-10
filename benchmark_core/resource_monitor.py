"""Resource monitoring utilities.

The monitor uses psutil when available and degrades gracefully when it is not
installed, so experiments can still run in minimal environments.
"""

from __future__ import annotations

from benchmark_core.schemas import ResourceUsage

try:
    import psutil
except ImportError:  # pragma: no cover - depends on optional environment
    psutil = None


class ResourceMonitor:
    """Context manager that captures simple CPU and RAM measurements."""

    def __init__(self) -> None:
        self._process = psutil.Process() if psutil is not None else None
        self.usage = ResourceUsage()

    def __enter__(self) -> "ResourceMonitor":
        if self._process is None:
            return self

        self.usage.cpu_percent_start = self._process.cpu_percent(interval=None)
        self.usage.memory_mb_start = self._memory_mb()
        self.usage.memory_mb_peak = self.usage.memory_mb_start
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._process is None:
            return None

        self.usage.cpu_percent_end = self._process.cpu_percent(interval=None)
        self.usage.memory_mb_end = self._memory_mb()
        values = [
            value
            for value in (self.usage.memory_mb_start, self.usage.memory_mb_end)
            if value is not None
        ]
        self.usage.memory_mb_peak = max(values) if values else None
        return None

    def _memory_mb(self) -> float:
        if self._process is None:
            return 0.0
        return self._process.memory_info().rss / (1024 * 1024)
