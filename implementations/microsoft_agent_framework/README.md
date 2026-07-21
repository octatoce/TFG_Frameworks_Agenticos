# Microsoft Agent Framework

Implementaciones para Microsoft Agent Framework.

El baseline ejecutable usa un adaptador determinista local sobre `benchmark_core` para mantener la comparabilidad con los demas frameworks durante los tests.

Cuando `ExperimentConfig.model_provider == "openai"`, la ruta real usa `agent_framework.BaseAgent` como custom agent y llama al SDK oficial de OpenAI Responses desde el agente. Esta ruta evita un fallo de `ContextVar` observado en este entorno con `OpenAIChatClient` y `OpenAIChatCompletionClient`.

Arquitecturas incluidas:

- `ARCH_01_SINGLE_REACT`
- `ARCH_02_SEQUENTIAL_PIPELINE`
- `ARCH_03_ROUTER_SPECIALISTS`
