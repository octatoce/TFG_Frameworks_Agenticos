# LangGraph - ARCH_03_SUPERVISOR_WORKERS

Arquitectura con supervisor, workers y rutas condicionales.

## Roles

- `supervisor_planning`: decide que workers ejecutar.
- `data_worker`: recupera y resume evidencia.
- `reasoning_worker`: razona sobre la evidencia.
- `validation_worker`: valida consistencia y confianza.
- `supervisor_synthesis`: genera la respuesta final.

## Flujo

Primero se ejecuta `supervisor_planning`. Segun los workers seleccionados, el grafo salta solo por los nodos necesarios y termina en `supervisor_synthesis`.

El orden canonico de workers es:

```text
data_worker -> reasoning_worker -> validation_worker
```

## Resultado

Devuelve un `ExperimentResult` con:

- respuesta final;
- workers seleccionados y omitidos;
- un `AgentStep` por fase ejecutada;
- una llamada LLM por fase ejecutada.

## Objetivo

Comparar una estructura multi-agente dinamica usando edges condicionales de LangGraph sin perder trazabilidad.
