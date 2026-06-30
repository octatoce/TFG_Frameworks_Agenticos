# ARCH_04_SUPERVISOR_WORKERS - LangGraph

Implementacion como `StateGraph` con supervisor centralizado, workers condicionales y retorno al supervisor tras cada worker.

Nodos principales:

- `supervisor_plan`
- `supervisor_decision`
- `data_worker`
- `reasoning_worker`
- `validation_worker`
- `synthesis_worker`
- `supervisor_finalize`

El estado registra plan, salidas de workers, iteraciones, revisiones, workers ejecutados, aceptados/rechazados, warnings y razon de parada. No se usa checkpointing.
