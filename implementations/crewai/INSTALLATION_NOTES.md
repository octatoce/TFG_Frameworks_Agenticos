# CrewAI - Installation Notes

## Version used

- Package: `crewai`
- Installed version: `1.14.6`
- Installation command used:

```powershell
python -m pip install crewai
```

The dependency is also recorded in:

- `implementations/crewai/requirements.txt`
- `pyproject.toml` optional dependency group `crewai`

## Environment notes

- Python used during validation: `3.13.6`
- Operating system reported by the benchmark result: `Windows 10`
- `pip` installed packages in the user site because the global site-packages directory was not writable.
- CrewAI pulls a large dependency tree, including `chromadb`, `lancedb`, `openai`, telemetry-related packages, and CLI tooling.
- Several installed executable scripts, including `crewai.exe`, were placed under `C:\Users\octat\AppData\Roaming\Python\Python313\Scripts`, which was not on `PATH`.

## Important runtime side effects

Importing CrewAI attempted to create user-level application directories:

```text
C:\Users\octat\AppData\Local\CrewAI\...
C:\Users\octat\AppData\Local\crewai\credentials
```

Inside the managed workspace sandbox this raised `PermissionError`.

Final mitigation in `ARCH_01_SINGLE_REACT/run.py`:

- Set `LOCALAPPDATA` to a repository-local directory before importing CrewAI.
- Monkeypatch `appdirs.user_data_dir` so CrewAI app data is redirected to `.crewai_data`.
- Set telemetry/tracking-related environment variables:
  - `CREWAI_DISABLE_TELEMETRY=true`
  - `CREWAI_DISABLE_TRACKING=true`
  - `CREWAI_DISABLE_VERSION_CHECK=true`
  - `OTEL_SDK_DISABLED=true`
  - `CREWAI_TESTING=true`
- Add `.crewai_data/` to `.gitignore`.

This makes the prototype more reproducible because hidden per-user CrewAI state is not required for benchmark execution.

## Implementation notes for ARCH_01_SINGLE_REACT

`ARCH_01_SINGLE_REACT` is implemented with real CrewAI primitives:

- `Agent`
- `Task`
- `Crew`
- `Process.sequential`
- Custom `BaseLLM` subclass wrapping `benchmark_core.InstrumentedLLM`

The custom LLM avoids external API keys during iteration 1 while still validating:

- CrewAI orchestration overhead,
- common input/output schemas,
- LLM call accounting,
- token accounting proxy,
- latency measurement,
- resource measurement,
- JSON result persistence.

The task prompt is rendered by the same `benchmark_core.render_single_react_prompt` used by LangGraph. CrewAI then wraps that task internally, which produced a higher token proxy count than LangGraph in the smoke run.

## Warnings observed

Tests pass, but CrewAI emits deprecation warnings from its internals:

- `function_calling_llm is deprecated`
- `max_retries` is deprecated in favor of `guardrail_max_retries`
- additional generic deprecation warnings around agent attributes

These warnings do not break the current prototype, but they should be monitored before freezing the final TFG environment.

## Smoke result

The shared smoke case was executed and saved to:

```text
results/raw/crewai/ARCH_01_SINGLE_REACT/iter1-arch01-smoke-001.json
```

Observed smoke metrics:

- Status: `success`
- LLM calls: `1`
- Token proxy total: `202`
- Resource usage: memory start/end captured after installing `psutil`

