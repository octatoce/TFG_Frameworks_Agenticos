# LangGraph - ARCH_01_SINGLE_REACT

Arquitectura minima con un unico nodo LangGraph.

## Flujo

1. Se renderiza el prompt comun de `ARCH_01_SINGLE_REACT`.
2. El nodo `react_agent` llama al LLM instrumentado.
3. El grafo termina directamente en `END`.

## Resultado

Devuelve un `ExperimentResult` con:

- respuesta final;
- una llamada LLM;
- un unico `AgentStep`;
- metricas y uso de recursos.

## Objetivo

Servir como caso base: la forma mas simple de ejecutar una tarea con LangGraph dentro del benchmark.
