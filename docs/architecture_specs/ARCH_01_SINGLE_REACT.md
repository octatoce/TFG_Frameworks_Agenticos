# ARCH_01_SINGLE_REACT

## Objetivo

Evaluar el comportamiento de una arquitectura agentica minima, formada por un unico agente decisor que recibe la consulta, razona sobre el caso y produce una respuesta final comparable.

Esta arquitectura actua como baseline principal de la iteracion. La pregunta experimental asociada es:

```text
Realmente mejora algo usar varios agentes o workflows complejos frente a un unico agente bien definido?
```

## Componentes logicos

| Componente | Responsabilidad |
| --- | --- |
| Single Agent | Recibe la consulta, analiza el contexto disponible y genera la respuesta final. |
| Document context | Proporciona los documentos de `ExperimentInput.documents` al agente de forma equivalente entre frameworks. |
| LLM interface | Ejecuta la llamada al modelo mediante `benchmark_core`, en modo local determinista o proveedor real. |
| Output formatter | Normaliza la respuesta final al `structured_output` comun. |

## Herramientas permitidas

La arquitectura objetivo puede incluir herramientas, siempre que esten disponibles de forma equivalente en todos los frameworks comparados:

| Herramienta | Uso previsto |
| --- | --- |
| `document_search` | Recuperar fragmentos relevantes de los documentos. |
| `calculator` / `basic_stats` | Calcular metricas simples cuando el caso lo requiera. |
| `output_formatter` | Reparar o normalizar la salida estructurada. |

En la implementacion actual de iteracion 1 no se han introducido herramientas externas. Los documentos se inyectan directamente en el prompt comun mediante `render_single_react_prompt`. Esta decision se acepta como variante minima valida para validar el contrato, la ejecucion y la instrumentacion antes de introducir herramientas reales.

## Restricciones

- Debe existir un unico agente decisor.
- No puede haber supervisor, workers, debate entre agentes ni delegacion dinamica.
- El maximo de iteraciones debe venir de `ExperimentConfig.max_agent_iterations`.
- El prompt base debe ser comun o funcionalmente equivalente entre frameworks.
- La instrumentacion de llamadas LLM, tokens, latencia, coste estimado, errores y recursos debe pasar por `benchmark_core`.
- La salida debe ser un `ExperimentResult` valido segun el schema comun.

## Flujo logico

1. Recibir `ExperimentInput` y `ExperimentConfig`.
2. Construir el LLM instrumentado con `build_llm_from_config`.
3. Renderizar el prompt comun de `ARCH_01_SINGLE_REACT`.
4. Ejecutar un unico agente o nodo agente.
5. Registrar la llamada LLM como `LLMCallMetrics`.
6. Extraer la respuesta final.
7. Construir `structured_output`.
8. Registrar un `AgentStep` asociado al agente unico.
9. Calcular metricas con `benchmark_core`.
10. Devolver `ExperimentResult`.

## Pseudocodigo canonico

```python
def run_architecture(input_data: ExperimentInput, config: ExperimentConfig) -> ExperimentResult:
    started_at = utc_now()
    llm = build_llm_from_config(config)

    prompt = render_single_react_prompt(input_data)

    call_record = llm.complete(
        prompt=prompt,
        input_data=input_data,
        call_id=f"{config.run_id}-llm-001",
        step_id=1,
    )

    final_answer = extract_final_answer(call_record.response)

    structured_output = {
        "answer": final_answer,
        "mode": f"{config.model_provider}_react",
        "document_ids": [document.document_id for document in input_data.documents],
    }

    step = AgentStep(
        step_id=1,
        name="single_react_agent",
        step_type="agent_llm_call",
        llm_call_ids=[call_record.metrics.call_id],
        input_data={"prompt": prompt},
        output_data=structured_output,
    )

    return build_experiment_result(
        input_data=input_data,
        config=config,
        status=RunStatus.SUCCESS,
        final_answer=final_answer,
        structured_output=structured_output,
        steps=[step],
        llm_calls=[call_record.metrics],
        errors=[],
        started_at=started_at,
        finished_at=utc_now(),
    )
```

## Equivalencia entre frameworks

| Framework | Implementacion equivalente |
| --- | --- |
| LangGraph | Grafo con un unico nodo `react_agent` y transicion a `END`. |
| CrewAI | `Crew` secuencial con un unico `Agent` y una unica `Task`. |

La equivalencia no exige que las estructuras internas sean identicas. Exige que:

- Solo haya un agente decisor.
- La entrada experimental sea la misma.
- El prompt base sea el mismo antes del envoltorio propio de cada framework.
- El modelo y parametros de generacion sean los mismos.
- La salida use el mismo schema.
- Las metricas se recojan con `benchmark_core`.

## Metricas especialmente relevantes

| Metrica | Motivo |
| --- | --- |
| Latencia total | Deberia ser una de las arquitecturas mas rapidas. |
| Latencia LLM | Separa coste del modelo de overhead del framework. |
| Numero de llamadas LLM | Baseline de coste frente a arquitecturas multi-paso. |
| Tokens totales | Permite comparar contra pipeline y multi-agente. |
| Coste estimado | Facilita comparacion economica. |
| Errores de formato | Evalua robustez del output estructurado. |
| Uso de CPU/RAM | Ayuda a detectar overhead del framework. |

## Estado de implementacion

Implementada en iteracion 1 para:

- `implementations/langgraph/ARCH_01_SINGLE_REACT/run.py`
- `implementations/crewai/ARCH_01_SINGLE_REACT/run.py`

Estado actual:

- Modo local determinista mediante `InstrumentedLLM`.
- Modo OpenAI real mediante `OpenAIInstrumentedLLM` cuando `config.model_provider == "openai"`.
- Sin herramientas externas reales todavia.
- Un `AgentStep` por ejecucion.
- Una llamada LLM por ejecucion en el flujo normal.

## Riesgos y decisiones pendientes

- Si se anade `document_search`, debe introducirse de forma equivalente en LangGraph y CrewAI.
- Si CrewAI anade prompt wrapping propio, la diferencia de tokens debe conservarse como parte del overhead real del framework.
- Si se usan modelos OpenAI reales, los resultados pueden variar entre ejecuciones; deben registrarse `model_name`, `temperature`, tokens reales y `response_id`.
