# ARCH_02_SEQUENTIAL_PIPELINE

## Objetivo

Evaluar una arquitectura secuencial y controlada donde cada fase tiene una responsabilidad concreta y la salida de una fase alimenta a la siguiente.

Esta arquitectura representa un punto intermedio entre el agente unico de `ARCH_01_SINGLE_REACT` y sistemas multi-agente mas complejos. La pregunta experimental asociada es:

```text
Un pipeline explicito y simple mejora la trazabilidad y la calidad sin anadir demasiada latencia?
```

## Componentes logicos

| Componente | Responsabilidad |
| --- | --- |
| Planner | Interpreta la consulta y produce un plan breve de resolucion. |
| Retriever | Selecciona documentos o fragmentos relevantes a partir del plan. |
| Analyst | Analiza la query y la evidencia recuperada. |
| Writer | Produce la respuesta final en formato estructurado comun. |

## Restricciones

- El flujo debe ser estrictamente secuencial.
- Deben existir exactamente cuatro fases logicas: Planner, Retriever, Analyst y Writer.
- No puede haber ramas condicionales complejas.
- No puede haber comunicacion libre entre componentes.
- No puede haber delegacion dinamica.
- Cada fase debe quedar registrada como `AgentStep`.
- Las llamadas LLM deben registrarse como `LLMCallMetrics`.
- Los reintentos deben estar controlados por `ExperimentConfig.retry_count`.
- El limite operativo debe respetar `ExperimentConfig.timeout_seconds` y, cuando aplique, `ExperimentConfig.max_agent_iterations`.

## Flujo logico

1. Recibir `ExperimentInput` y `ExperimentConfig`.
2. Inicializar estado comun del pipeline.
3. Planner genera un plan breve.
4. Retriever selecciona evidencias relevantes.
5. Analyst analiza query, plan y evidencias.
6. Writer genera la respuesta final.
7. Validar y normalizar `structured_output`.
8. Registrar pasos, llamadas LLM, errores y recursos.
9. Calcular metricas con `benchmark_core`.
10. Devolver `ExperimentResult`.

## Estado compartido minimo

```python
state = {
    "query": input_data.query,
    "documents": input_data.documents,
    "plan": None,
    "evidence": [],
    "analysis": None,
    "final_output": None,
}
```

## Pseudocodigo canonico

```python
def run_architecture(input_data: ExperimentInput, config: ExperimentConfig) -> ExperimentResult:
    started_at = utc_now()
    state = initialize_pipeline_state(input_data)
    steps = []
    llm_calls = []
    errors = []

    state["plan"], planner_call = planner.run(state["query"], input_data.metadata)
    steps.append(make_step("planner", planner_call))
    llm_calls.append(planner_call)

    state["evidence"], retriever_call = retriever.run(
        query=state["query"],
        documents=state["documents"],
        plan=state["plan"],
    )
    steps.append(make_step("retriever", retriever_call))
    llm_calls.append(retriever_call)

    state["analysis"], analyst_call = analyst.run(
        query=state["query"],
        evidence=state["evidence"],
        plan=state["plan"],
    )
    steps.append(make_step("analyst", analyst_call))
    llm_calls.append(analyst_call)

    state["final_output"], writer_call = writer.run(
        query=state["query"],
        analysis=state["analysis"],
        evidence=state["evidence"],
    )
    steps.append(make_step("writer", writer_call))
    llm_calls.append(writer_call)

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
| LangGraph | Grafo lineal: `Planner -> Retriever -> Analyst -> Writer -> END`. |
| CrewAI | Secuencia de cuatro tareas: planning, retrieval, analysis y writing. |

Para que la comparacion sea valida:

- El orden de fases debe ser siempre el mismo.
- Cada fase debe recibir informacion equivalente.
- Cada fase debe tener un prompt base comun o funcionalmente equivalente.
- No deben existir optimizaciones especificas por framework.
- Cada fase debe producir una salida trazable.

## Metricas especialmente relevantes

| Metrica | Motivo |
| --- | --- |
| Latencia total | Evalua el coste de encadenar fases. |
| Latencia por fase | Permite identificar el componente mas caro. |
| Llamadas LLM | Normalmente debe ser mayor que en `ARCH_01_SINGLE_REACT`. |
| Tokens por fase | Ayuda a detectar prompts o fases demasiado costosas. |
| Errores acumulados | Un error temprano puede afectar a todo el pipeline. |
| Trazabilidad | Deberia mejorar respecto al agente unico. |

## Estado de implementacion

Implementada en iteracion 1 para:

- `implementations/langgraph/ARCH_02_SEQUENTIAL_PIPELINE/run.py`
- `implementations/crewai/ARCH_02_SEQUENTIAL_PIPELINE/run.py`

Estado actual:

- Cuatro fases logicas: `planner`, `retriever`, `analyst`, `writer`.
- Cuatro `AgentStep` por ejecucion.
- Cuatro llamadas LLM por ejecucion en el flujo normal.
- Modo local determinista mediante `InstrumentedLLM`.
- Modo OpenAI real disponible mediante `OpenAIInstrumentedLLM` cuando `config.model_provider == "openai"`.
- El retriever se implementa como fase LLM instrumentada, no como herramienta determinista independiente.

## Riesgos y decisiones pendientes

- Evaluar en una iteracion posterior si `Retriever` debe pasar a ser una herramienta determinista. Cambiar esa decision afectaria directamente a llamadas, tokens y coste, por lo que debe aplicarse de forma equivalente en ambos frameworks.
- Evitar que CrewAI active delegacion o memoria implicita.
- Evitar que LangGraph introduzca rutas condicionales que conviertan el pipeline en supervisor encubierto.
