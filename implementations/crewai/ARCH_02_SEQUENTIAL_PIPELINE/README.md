# CrewAI - ARCH_02_SEQUENTIAL_PIPELINE

Arquitectura de pipeline secuencial con cuatro agentes especializados.

## Fases

1. `planner`: crea un plan breve.
2. `retriever`: selecciona evidencia.
3. `analyst`: analiza la evidencia.
4. `writer`: produce la respuesta final.

## Flujo

Cada fase es una `Task` de CrewAI y usa como contexto la task anterior. Todas se ejecutan en un `Crew` con `Process.sequential`.

## Resultado

Devuelve un `ExperimentResult` con:

- respuesta final extraida de la fase `writer`;
- cuatro llamadas LLM;
- cuatro `AgentStep`;
- salidas intermedias: plan, evidencia y analisis.

## Objetivo

Medir el coste y la trazabilidad de dividir una misma pregunta en pasos secuenciales.
