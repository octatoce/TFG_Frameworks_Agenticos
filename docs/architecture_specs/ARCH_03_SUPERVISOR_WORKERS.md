# ARCH_03_SUPERVISOR_WORKERS

## Objetivo

Evaluar una arquitectura multi-agente con un supervisor central que decide que trabajadores intervienen, coordina el flujo y sintetiza la respuesta final.

Esta es la primera arquitectura claramente multi-agente de la iteracion. La pregunta experimental asociada es:

```text
Un supervisor central mejora la calidad y modularidad de la decision respecto a un pipeline fijo, aunque aumente coste y latencia?
```

## Componentes logicos

| Componente | Responsabilidad |
| --- | --- |
| Supervisor | Decide que workers ejecutar, coordina el flujo y sintetiza o valida la respuesta final. |
| Data Worker | Recupera y resume evidencia documental. |
| Reasoning Worker | Analiza la consulta y propone una decision preliminar. |
| Validation Worker | Revisa inconsistencias, limitaciones y confianza de la respuesta. |
| Final Writer | Genera la salida estructurada final si no lo hace directamente el supervisor. |

## Restricciones

- Debe existir un supervisor explicito.
- Deben existir al menos tres workers logicos: Data, Reasoning y Validation.
- Los workers no pueden comunicarse libremente entre si.
- Toda comunicacion debe pasar por el supervisor o por un estado comun controlado.
- El supervisor debe registrar que workers se han ejecutado y cuales se han omitido.
- El numero de iteraciones debe estar limitado por `ExperimentConfig.max_agent_iterations`.
- Los reintentos deben estar controlados por `ExperimentConfig.retry_count`.
- Cada intervencion relevante debe registrarse como `AgentStep`.
- Cada llamada al modelo debe registrarse como `LLMCallMetrics`.

## Flujo logico

1. Recibir `ExperimentInput` y `ExperimentConfig`.
2. Supervisor analiza query, metadata y workers disponibles.
3. Supervisor selecciona workers necesarios.
4. Data Worker recupera evidencia si ha sido seleccionado.
5. Reasoning Worker genera analisis o decision preliminar si ha sido seleccionado.
6. Validation Worker revisa la decision preliminar si ha sido seleccionado.
7. Supervisor integra resultados.
8. Supervisor o Final Writer genera respuesta estructurada.
9. Validar campos obligatorios.
10. Registrar metricas, pasos, llamadas LLM, errores y recursos.
11. Devolver `ExperimentResult`.

## Estado compartido minimo

```python
state = {
    "query": input_data.query,
    "documents": input_data.documents,
    "selected_workers": [],
    "skipped_workers": [],
    "evidence": [],
    "preliminary_decision": None,
    "validation_report": None,
    "final_output": None,
}
```

## Pseudocodigo canonico

```python
def run_architecture(input_data: ExperimentInput, config: ExperimentConfig) -> ExperimentResult:
    started_at = utc_now()
    state = initialize_supervisor_state(input_data)
    steps = []
    llm_calls = []
    errors = []

    supervisor_plan, supervisor_call = supervisor.plan(
        query=state["query"],
        metadata=input_data.metadata,
        available_workers=["data_worker", "reasoning_worker", "validation_worker"],
    )
    state["selected_workers"] = supervisor_plan["selected_workers"]
    state["skipped_workers"] = supervisor_plan["skipped_workers"]
    steps.append(make_step("supervisor_planning", supervisor_call))
    llm_calls.append(supervisor_call)

    if "data_worker" in state["selected_workers"]:
        state["evidence"], data_call = data_worker.run(state["query"], state["documents"])
        steps.append(make_step("data_worker", data_call))
        llm_calls.append(data_call)

    if "reasoning_worker" in state["selected_workers"]:
        state["preliminary_decision"], reasoning_call = reasoning_worker.run(
            query=state["query"],
            evidence=state["evidence"],
        )
        steps.append(make_step("reasoning_worker", reasoning_call))
        llm_calls.append(reasoning_call)

    if "validation_worker" in state["selected_workers"]:
        state["validation_report"], validation_call = validation_worker.run(
            query=state["query"],
            evidence=state["evidence"],
            preliminary_decision=state["preliminary_decision"],
        )
        steps.append(make_step("validation_worker", validation_call))
        llm_calls.append(validation_call)

    state["final_output"], synthesis_call = supervisor.synthesize(
        query=state["query"],
        evidence=state["evidence"],
        preliminary_decision=state["preliminary_decision"],
        validation_report=state["validation_report"],
    )
    steps.append(make_step("supervisor_synthesis", synthesis_call))
    llm_calls.append(synthesis_call)

    structured_output = parse_or_repair_output(state["final_output"])

    return build_experiment_result(
        input_data=input_data,
        config=config,
        status=RunStatus.SUCCESS,
        final_answer=structured_output["answer"],
        structured_output=structured_output,
        steps=steps,
        llm_calls=llm_calls,
        errors=errors,
        started_at=started_at,
        finished_at=utc_now(),
    )
```

## Equivalencia entre frameworks

| Framework | Implementacion equivalente |
| --- | --- |
| LangGraph | Grafo con nodo supervisor, nodos worker y nodo de sintesis. Puede usar rutas condicionales controladas. |
| CrewAI | Equipo con manager/supervisor y agentes especialistas. |

Para que la comparacion sea valida:

- Debe haber un supervisor identificable.
- Deben existir los mismos workers logicos.
- Los workers no deben hablar entre si directamente.
- El supervisor debe registrar workers ejecutados y omitidos.
- La respuesta final debe seguir el mismo schema.
- Las diferencias internas del framework deben quedar reflejadas en `steps`, `llm_calls` y `metadata`.

## Metricas especialmente relevantes

| Metrica | Motivo |
| --- | --- |
| Latencia total | Deberia aumentar respecto a `ARCH_01` y posiblemente `ARCH_02`. |
| Latencia por worker | Permite identificar roles costosos. |
| Numero de workers ejecutados | Mide complejidad real de la ejecucion. |
| Llamadas LLM | Normalmente sera mayor que en arquitecturas anteriores. |
| Tokens por rol | Permite comparar coste del supervisor frente a workers. |
| Calidad de validacion | Evalua si el worker critico aporta valor. |
| Tasa de errores | Mas componentes pueden introducir mas fallos. |
| Mantenibilidad | Permite valorar si separar roles mejora claridad y evolucion. |

## Estado de implementacion

Implementada en iteracion 1 para:

- `implementations/langgraph/ARCH_03_SUPERVISOR_WORKERS/run.py`
- `implementations/crewai/ARCH_03_SUPERVISOR_WORKERS/run.py`

Estado actual:

- Intervenciones logicas posibles: `supervisor_planning`, `data_worker`, `reasoning_worker`, `validation_worker`, `supervisor_synthesis`.
- El supervisor decide dinamicamente que workers ejecutar.
- `reasoning_worker` se mantiene como worker minimo.
- `data_worker` se ejecuta cuando existen documentos de entrada.
- `validation_worker` se ejecuta cuando la query o el tipo de tarea sugieren validacion, evaluacion, riesgo, confianza, errores o comparacion.
- Los workers omitidos quedan registrados en `structured_output.skipped_workers`.
- El numero de `AgentStep` y llamadas LLM depende de los workers seleccionados: `1 + workers_ejecutados + 1`.
- Modo local determinista mediante `InstrumentedLLM`.
- Modo OpenAI real disponible mediante `OpenAIInstrumentedLLM` cuando `config.model_provider == "openai"`.
- En CrewAI se usa un supervisor explicito como agente, pero `Process.sequential` para evitar delegacion o planificacion jerarquica no controlada.

## Riesgos y decisiones pendientes

- Si se usa un proveedor real, el output de `supervisor_planning` debe respetar el formato `SELECTED_WORKERS=...` y `SKIPPED_WORKERS=...`; si no lo respeta, el parser aplica una politica conservadora.
- Documentar cualquier diferencia entre manager nativo de CrewAI y supervisor explicito implementado manualmente si se migra a `Process.hierarchical`.
- Controlar memoria, delegacion y herramientas implicitas para evitar ventajas no metodologicas.
