# CrewAI - ARCH_03_ROUTER_SPECIALISTS

Arquitectura con un router y varios specialists.

## Roles

- `router_routing`: decide que specialists ejecutar.
- `data_specialist`: recupera y resume evidencia.
- `reasoning_specialist`: razona sobre la evidencia.
- `validation_specialist`: valida consistencia y confianza.
- `router_synthesis`: genera la respuesta final.

## Flujo

Primero se ejecuta el router para elegir specialists. Despues se ejecutan los specialists seleccionados y, al final, el router sintetiza la respuesta.

La ejecucion usa `Process.sequential` para que el orden sea claro y comparable con el resto del benchmark.

## Resultado

Devuelve un `ExperimentResult` con:

- respuesta final;
- specialists seleccionados y omitidos;
- un `AgentStep` por fase ejecutada;
- una llamada LLM por fase ejecutada.

## Objetivo

Comparar una estructura multi-agente mas dinamica sin perder trazabilidad ni control experimental.

