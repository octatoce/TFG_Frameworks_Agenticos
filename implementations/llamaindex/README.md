# LlamaIndex

Implementaciones para LlamaIndex Workflows / AgentWorkflow.

El baseline ejecutable usa un adaptador determinista local sobre `benchmark_core` para mantener la comparabilidad con los demas frameworks durante los tests.

Cuando `ExperimentConfig.model_provider == "openai"`, la ruta real usa `llama_index.core.agent.workflow.FunctionAgent` con `llama_index.llms.openai.OpenAI`.

Arquitecturas incluidas:

- `ARCH_01_SINGLE_REACT`
- `ARCH_02_SEQUENTIAL_PIPELINE`
- `ARCH_03_SUPERVISOR_WORKERS`

