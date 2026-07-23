# Pydantic AI

Implementaciones para Pydantic AI + pydantic-graph.

El baseline ejecutable usa un adaptador determinista local sobre `benchmark_core` y modelos Pydantic para validar salidas intermedias sin alterar el contrato comun.

Cuando `ExperimentConfig.model_provider == "openai"`, la ruta real usa `pydantic_ai.Agent` con modelo `openai-chat:<model_name>`. El prefijo explicito conserva Chat Completions cuando Pydantic AI cambie el significado por defecto de `openai:` en v2. Las salidas comunes se validan despues con modelos Pydantic locales.

Arquitecturas incluidas:

- `ARCH_01_SINGLE_REACT`
- `ARCH_02_SEQUENTIAL_PIPELINE`
- `ARCH_03_ROUTER_SPECIALISTS`
