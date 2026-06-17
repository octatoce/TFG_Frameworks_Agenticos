# LangGraph - Installation Notes

## Version used

- Package: `langgraph`
- Installed version: `1.2.4`
- Installation command used:

```powershell
python -m pip install langgraph
```

The dependency is also recorded in:

- `implementations/langgraph/requirements.txt`
- `pyproject.toml` optional dependency group `langgraph`

## Environment notes

- Python used during validation: `3.13.6`
- Operating system reported by the benchmark result: `Windows 10`
- `pip` installed packages in the user site because the global site-packages directory was not writable.
- The first combined installation attempt for `langgraph crewai` timed out under the default command timeout. Re-running with a longer timeout completed successfully.

## Implementation notes for ARCH_01_SINGLE_REACT

`ARCH_01_SINGLE_REACT` is implemented with a real LangGraph `StateGraph` containing one node:

- Node: `react_agent`
- Entry point: `react_agent`
- Terminal edge: `react_agent -> END`

The node calls the shared deterministic `InstrumentedLLM` from `benchmark_core`. This avoids external API keys during iteration 1 while still validating:

- framework execution path,
- common input/output schemas,
- LLM call accounting,
- token accounting proxy,
- latency measurement,
- resource measurement,
- JSON result persistence.

## Issues found

LangGraph resolves state type annotations when compiling the graph. A first implementation used `Any`, `AgentStep`, `LLMCallMetrics`, and `ExperimentError` directly in the `TypedDict` state. When the module was loaded through `importlib` in tests, LangGraph raised `NameError` while resolving those annotations.

Final mitigation:

- Keep the LangGraph state schema simple with built-in `object` values for lists crossing the graph boundary.
- Validate full structured objects later through the canonical `ExperimentResult` Pydantic schema.

This keeps the framework-specific state lightweight and avoids coupling LangGraph's type resolution to benchmark schema internals.

## Smoke result

The shared smoke case was executed and saved to:

```text
results/raw/langgraph/ARCH_01_SINGLE_REACT/iter1-arch01-smoke-001.json
```

Observed smoke metrics:

- Status: `success`
- LLM calls: `1`
- Token proxy total: `134`
- Resource usage: memory start/end captured after installing `psutil`

