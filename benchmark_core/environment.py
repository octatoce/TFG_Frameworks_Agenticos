"""Environment metadata helpers for experiment traceability."""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from benchmark_core.schemas import EnvironmentInfo


def get_package_version(package_name: str) -> str | None:
    """Return an installed package version, or None if unavailable."""

    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def get_git_commit(repo_root: Path | None = None) -> str | None:
    """Return the current git commit when the project is inside a git repo."""

    root = repo_root or Path.cwd()
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    return completed.stdout.strip() or None


def build_environment_info(
    package_names: list[str] | None = None,
    repo_root: Path | None = None,
) -> EnvironmentInfo:
    """Build reproducibility metadata for one run."""

    versions = {
        package_name: version
        for package_name in (package_names or [])
        if (version := get_package_version(package_name)) is not None
    }
    benchmark_core_version = get_package_version("tfg-agent-frameworks")
    return EnvironmentInfo(
        python_version=sys.version.split()[0],
        os=f"{platform.system()} {platform.release()}",
        git_commit=get_git_commit(repo_root=repo_root),
        benchmark_core_version=benchmark_core_version,
        package_versions=versions,
    )
