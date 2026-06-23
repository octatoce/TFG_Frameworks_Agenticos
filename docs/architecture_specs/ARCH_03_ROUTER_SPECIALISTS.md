# ARCH_03_ROUTER_SPECIALISTS

## Objetivo

Evaluar una arquitectura multi-componente donde un router decide que especialistas deben intervenir y despues se construye una respuesta final a partir de sus salidas.

La pregunta experimental asociada es:

```text
Un router condicional con especialistas mejora la calidad o trazabilidad frente a un pipeline fijo, sin introducir todavia supervision iterativa?
```

Esta arquitectura no representa un supervisor real. El router no revisa, no corrige y no pide nuevas iteraciones a los especialistas. Solo selecciona especialistas, ejecuta los seleccionados una vez en orden canonico y sintetiza la respuesta final.

## Componentes Logicos

| Componente | Responsabilidad |
| --- | --- |
| Router | Decide que especialistas ejecutar y registra seleccionados/omitidos. |
| Data Specialist | Recupera y resume evidencia documental si hay documentos. |
| Reasoning Specialist | Razona sobre la query y la evidencia disponible. |
| Validation Specialist | Valida consistencia, limitaciones y confianza cuando el caso lo requiere. |
| Router Synthesis | Integra las salidas disponibles en una respuesta final. |

## Restricciones

- Debe existir un router explicito.
- Deben existir tres especialistas logicos: Data, Reasoning y Validation.
- El router decide una sola vez que especialistas ejecutar.
- Los especialistas seleccionados se ejecutan como maximo una vez por run.
- No hay bucles, replanificacion, feedback ni correcciones.
- No hay comunicacion libre entre especialistas.
- El orden de ejecucion es canonico: `data_specialist`, `reasoning_specialist`, `validation_specialist`.
- La sintesis final no puede pedir trabajo adicional; solo agrega las salidas ya generadas.
- Cada intervencion ejecutada debe registrarse como `AgentStep`.
- Cada llamada al modelo debe registrarse como `LLMCallMetrics`.

## Flujo Logico

1. Recibir `ExperimentInput` y `ExperimentConfig`.
2. `router_routing` analiza la query, documentos y metadata.
3. El router devuelve `selected_specialists`, `skipped_specialists` y una justificacion breve.
4. Se ejecutan, una sola vez, los especialistas seleccionados en orden canonico.
5. `router_synthesis` integra las salidas generadas.
6. Se construye `structured_output` con respuesta, especialistas seleccionados/omitidos y trazas clave.
7. Se devuelven metricas, pasos, llamadas LLM y `ExperimentResult`.

## Estado Compartido Minimo

```python
state = {
    "query": input_data.query,
    "documents": input_data.documents,
    "selected_specialists": [],
    "skipped_specialists": [],
    "router_plan": None,
    "evidence": [],
    "preliminary_decision": None,
    "validation_report": None,
    "final_output": None,
}
```

## Pseudocodigo Canonico

```python
def run_architecture(input_data: ExperimentInput, config: ExperimentConfig) -> ExperimentResult:
    started_at = utc_now()
    state = initialize_router_state(input_data)
    steps = []
    llm_calls = []
    errors = []

    router_plan, router_call = router.route(
        query=state["query"],
        documents=state["documents"],
        available_specialists=["data_specialist", "reasoning_specialist", "validation_specialist"],
    )
    state["selected_specialists"] = router_plan["selected_specialists"]
    state["skipped_specialists"] = router_plan["skipped_specialists"]
    steps.append(make_step("router_routing", router_call))
    llm_calls.append(router_call)

    if "data_specialist" in state["selected_specialists"]:
        state["evidence"], data_call = data_specialist.run(state)
        steps.append(make_step("data_specialist", data_call))
        llm_calls.append(data_call)

    if "reasoning_specialist" in state["selected_specialists"]:
        state["preliminary_decision"], reasoning_call = reasoning_specialist.run(state)
        steps.append(make_step("reasoning_specialist", reasoning_call))
        llm_calls.append(reasoning_call)

    if "validation_specialist" in state["selected_specialists"]:
        state["validation_report"], validation_call = validation_specialist.run(state)
        steps.append(make_step("validation_specialist", validation_call))
        llm_calls.append(validation_call)

    state["final_output"], synthesis_call = router.synthesize(state)
    steps.append(make_step("router_synthesis", synthesis_call))
    llm_calls.append(synthesis_call)

    return build_experiment_result(...)
```

## Equivalencia Entre Frameworks

| Framework | Implementacion equivalente |
| --- | --- |
| LangGraph | Grafo con nodo router, nodos especialistas y edges condicionales hacia la siguiente fase seleccionada. |
| CrewAI | Crew secuencial controlada: una task de routing, tasks de especialistas seleccionados y una task final de sintesis. |
| Microsoft Agent Framework | Agente/workflow controlado con fases equivalentes y seleccion explicita de especialistas. |
| LlamaIndex | `FunctionAgent`/workflow con routing controlado y sin handoffs libres. |
| Pydantic AI | Agente tipado y salida validada con modelos Pydantic, manteniendo el flujo lineal seleccionado por el router. |

La equivalencia exige:

- Mismos especialistas logicos.
- Mismo formato de seleccion: `SELECTED_SPECIALISTS=...`, `SKIPPED_SPECIALISTS=...`.
- Mismo orden canonico de especialistas.
- Sin reintentos semanticos ni revision iterativa.
- Misma salida `ExperimentResult`.

## Metricas Especialmente Relevantes

| Metrica | Motivo |
| --- | --- |
| Numero de especialistas ejecutados | Mide la complejidad real elegida por el router. |
| Latencia total | Evalua coste del routing frente al pipeline fijo. |
| Latencia por especialista | Permite identificar roles caros. |
| Llamadas LLM | Debe ser `1 + especialistas_ejecutados + 1`. |
| Tokens por rol | Permite comparar coste del router, especialistas y sintesis. |
| Tasa de errores de seleccion | Evalua si el router devuelve formato parseable. |

## Estado de Implementacion

Implementada para:

- `implementations/langgraph/ARCH_03_ROUTER_SPECIALISTS/run.py`
- `implementations/crewai/ARCH_03_ROUTER_SPECIALISTS/run.py`
- `implementations/microsoft_agent_framework/ARCH_03_ROUTER_SPECIALISTS/run.py`
- `implementations/llamaindex/ARCH_03_ROUTER_SPECIALISTS/run.py`
- `implementations/pydantic_ai/ARCH_03_ROUTER_SPECIALISTS/run.py`

Estado actual:

- Intervenciones posibles: `router_routing`, `data_specialist`, `reasoning_specialist`, `validation_specialist`, `router_synthesis`.
- `reasoning_specialist` se mantiene como especialista minimo.
- `data_specialist` se ejecuta cuando existen documentos de entrada.
- `validation_specialist` se ejecuta cuando la query o el tipo de tarea sugieren validacion, evaluacion, riesgo, confianza, errores o comparacion.
- Los especialistas omitidos quedan registrados en `structured_output.skipped_specialists`.
- No hay supervision real ni correcciones iterativas.

## Riesgos y Decisiones Pendientes

- Si un modelo real no respeta `SELECTED_SPECIALISTS=...` y `SKIPPED_SPECIALISTS=...`, el parser aplica una politica conservadora.
- La arquitectura de supervisor real queda separada como `ARCH_04_SUPERVISOR_WORKERS`, con revision, feedback, posibles revisiones y limite efectivo de iteraciones.
