# LangGraph - ARCH_03_ROUTER_SPECIALISTS

Arquitectura con router, specialists y rutas condicionales.

## Roles

- `router_routing`: decide que specialists ejecutar.
- `data_specialist`: recupera y resume evidencia.
- `reasoning_specialist`: razona sobre la evidencia.
- `validation_specialist`: valida consistencia y confianza.
- `router_synthesis`: genera la respuesta final.

## Flujo

Primero se ejecuta `router_routing`. Segun los specialists seleccionados, el grafo salta solo por los nodos necesarios y termina en `router_synthesis`.

El orden canonico de specialists es:

```text
data_specialist -> reasoning_specialist -> validation_specialist
```

## Resultado

Devuelve un `ExperimentResult` con:

- respuesta final;
- specialists seleccionados y omitidos;
- un `AgentStep` por fase ejecutada;
- una llamada LLM por fase ejecutada.

## Objetivo

Comparar una estructura multi-agente dinamica usando edges condicionales de LangGraph sin perder trazabilidad.

