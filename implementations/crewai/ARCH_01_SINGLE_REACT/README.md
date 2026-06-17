# CrewAI - ARCH_01_SINGLE_REACT

Arquitectura minima con un unico agente CrewAI.

## Flujo

1. Se renderiza el prompt comun de `ARCH_01_SINGLE_REACT`.
2. Se crea un `Agent` con el LLM instrumentado.
3. Se crea una `Task` asociada a ese agente.
4. Se ejecuta un `Crew` secuencial con una sola task.

## Resultado

Devuelve un `ExperimentResult` con:

- respuesta final;
- una llamada LLM;
- un unico `AgentStep`;
- metricas y uso de recursos.

## Objetivo

Servir como caso base: la forma mas simple de ejecutar una tarea con CrewAI dentro del benchmark.
