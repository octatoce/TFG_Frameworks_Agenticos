# LangGraph - ARCH_02_SEQUENTIAL_PIPELINE

Arquitectura de pipeline secuencial con cuatro nodos.

## Fases

1. `planner`: crea un plan breve.
2. `retriever`: selecciona evidencia.
3. `analyst`: analiza la evidencia.
4. `writer`: produce la respuesta final.

## Flujo

Cada fase es un nodo del `StateGraph`. El estado se va actualizando en orden:

```text
planner -> retriever -> analyst -> writer -> END
```

## Resultado

Devuelve un `ExperimentResult` con:

- respuesta final extraida del nodo `writer`;
- cuatro llamadas LLM;
- cuatro `AgentStep`;
- salidas intermedias: plan, evidencia y analisis.

## Objetivo

Medir el coste y la trazabilidad de dividir una misma pregunta en pasos secuenciales.
