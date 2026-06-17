# CrewAI - ARCH_03_SUPERVISOR_WORKERS

Arquitectura con un supervisor y varios workers.

## Roles

- `supervisor_planning`: decide que workers ejecutar.
- `data_worker`: recupera y resume evidencia.
- `reasoning_worker`: razona sobre la evidencia.
- `validation_worker`: valida consistencia y confianza.
- `supervisor_synthesis`: genera la respuesta final.

## Flujo

Primero se ejecuta el supervisor para elegir workers. Despues se ejecutan los workers seleccionados y, al final, el supervisor sintetiza la respuesta.

La ejecucion usa `Process.sequential` para que el orden sea claro y comparable con el resto del benchmark.

## Resultado

Devuelve un `ExperimentResult` con:

- respuesta final;
- workers seleccionados y omitidos;
- un `AgentStep` por fase ejecutada;
- una llamada LLM por fase ejecutada.

## Objetivo

Comparar una estructura multi-agente mas dinamica sin perder trazabilidad ni control experimental.
