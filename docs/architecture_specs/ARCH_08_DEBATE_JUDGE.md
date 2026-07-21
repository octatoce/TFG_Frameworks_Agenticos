# ARCH_08_DEBATE_JUDGE

## 1. Objetivo

Evaluar si tres propuestas independientes, una critica cruzada explicita y un
juez final mejoran la robustez de respuestas ambiguas lo suficiente para
compensar cinco llamadas LLM, mayor contexto, coste y latencia.

## 2. Descripcion general

ARCH_08 entrega el mismo `ExperimentInput` completo a tres debaters con
perspectivas distintas. Las propuestas se sincronizan antes de una unica ronda
de debate. El juez recibe el caso original, las tres propuestas y esa ronda, y
es el unico componente autorizado para producir la decision final.

Las propuestas pueden ejecutarse concurrentemente porque no dependen entre si.
El debate y el juez siempre se ejecutan despues mediante dos barreras causales
explicitas.

## 3. Componentes

### debater_a

- Propone la solucion directa mas fuerte.
- Fundamenta argumentos en `query`, documentos y metadata.
- Explicita evidencia, supuestos y riesgos.
- No conoce las otras propuestas.

### debater_b

- Propone una interpretacion o solucion genuinamente alternativa.
- Evita repetir la posicion directa.
- Usa el mismo input, modelo, temperatura y herramientas que los demas.

### debater_c

- Adopta una perspectiva critica, conservadora y pragmatica.
- Destaca incertidumbre, restricciones y modos de fallo.
- Sigue produciendo una propuesta accionable, no solo objeciones.

### debate_round

- Recibe exactamente tres propuestas estructuradas.
- Ejecuta una sola llamada LLM.
- Emite una critica por propuesta con fortaleza, debilidad y desacuerdo.
- Registra consenso, desacuerdos y puntos fuertes/debiles.
- No reescribe propuestas ni decide la respuesta final.

### judge

- Recibe input original, propuestas y `debate_round`.
- Elige una propuesta, combina varias o rechaza todas.
- Justifica brevemente la decision, declara confianza y cuestiones abiertas.
- Es el unico responsable del `final_answer` y `structured_output` final.

## 4. Flujo de ejecucion

1. Validar `ExperimentInput` y `ExperimentConfig` mediante schemas comunes.
2. Ejecutar `debater_a`, `debater_b` y `debater_c` independientemente.
3. Sincronizar y normalizar las tres `DebateProposal`.
4. Ejecutar una vez `debate_round` con las propuestas completas.
5. Ejecutar una vez `judge` con propuestas y critica cruzada.
6. Construir pasos, llamadas, metricas y `ExperimentResult` con
   `benchmark_core`.
7. Guardar el JSON raw en la ruta canonica y devolver el resultado.

## 5. Pseudocodigo

```text
function run_architecture(input_data, config):
    proposals = fan_out_independent(
        debater_a(input_data),
        debater_b(input_data),
        debater_c(input_data)
    )

    debate = debate_round(input_data, proposals)  # exactly once
    decision = judge(input_data, proposals, debate)  # exactly once

    result = build_experiment_result(
        final_answer=decision.answer,
        structured_output={proposals, debate, decision},
        steps=[debater_a, debater_b, debater_c, debate_round, judge]
    )
    save_result_json(result)
    return result
```

## 6. Estructuras intermedias

`DebateProposal` contiene:

- debater y perspectiva;
- propuesta;
- argumentos y evidencia;
- supuestos y riesgos;
- salida raw y error opcional.

`DebateRoundOutput` contiene:

- `round_number=1`;
- tres `CrossCritique`;
- consenso, desacuerdos y puntos fuertes/debiles;
- salida raw.

`JudgeDecision` contiene:

- respuesta final;
- `decision_mode`: `select`, `combine` o `reject`;
- propuestas seleccionadas y rechazadas;
- justificacion, confianza y cuestiones no resueltas;
- salida raw.

Estas estructuras se validan con Pydantic dentro de `benchmark_core` sin
modificar los schemas persistidos comunes.

## 7. Trazas y metricas

El orden canonico de `AgentStep` es:

1. `debater_a` (`debate_proposal_llm_call`)
2. `debater_b` (`debate_proposal_llm_call`)
3. `debater_c` (`debate_proposal_llm_call`)
4. `debate_round` (`debate_round_llm_call`)
5. `judge` (`debate_judge_llm_call`)

`structured_output.debate_execution` y
`ExperimentMetrics.metadata.debate_execution` registran:

- latencia, llamadas, tokens y error por componente;
- numero de propuestas, rondas, criticas y desacuerdos;
- modo de decision y propuestas seleccionadas/rechazadas;
- si el juez combina o selecciona una propuesta;
- primitiva nativa y paralelismo de propuestas.

`latency_total_ms`, tokens totales, llamadas totales, coste y recursos siguen
siendo calculados por `benchmark_core`.

## 8. Equivalencia entre frameworks

| Framework | Propuestas | Debate y juicio |
| --- | --- | --- |
| LangGraph | Tres nodos desde `START` en `StateGraph`. | Arista de origen multiple a `debate_round`, seguida de `judge`. |
| Microsoft Agent Framework | `WorkflowBuilder.add_fan_out_edges` a tres `Executor`. | Fan-in explicito, `debate_round` y arista secuencial a `judge`. |
| CrewAI | Tres `Task(async_execution=True, context=[])`. | Tareas secuenciales con `context` explicito; sin manager. |
| LlamaIndex | Tres eventos y pasos de `Workflow`. | `Context.collect_events`, paso de debate y paso judge. |
| Pydantic AI + pydantic-graph | Fork `GraphBuilder` con salidas `DebateProposal`. | `Join`, `DebateRoundOutput` y `JudgeDecision` tipados. |

## 9. Restricciones

- Una sola ronda de debate.
- Exactamente un juez y una decision final.
- Sin routing, handoffs, Map-Reduce o supervisor iterativo.
- Sin memoria persistente, planning implicito ni delegacion.
- Sin herramientas diferentes entre frameworks.
- Sin bucle generador-critico-corrector.
- Las propuestas no reciben resultados de otros debaters.
- El debate no puede finalizar y el juez no puede abrir nuevas rondas.

## 10. Diferencias con otras arquitecturas

- A diferencia de ARCH_06, existe critica cruzada explicita antes de decidir.
- A diferencia de ARCH_07, todos trabajan sobre el mismo caso completo y no
  hay particionado documental.
- A diferencia de ARCH_04, el juez no planifica, delega ni solicita revisiones.
- A diferencia de ARCH_09, ninguna propuesta se corrige iterativamente.

## 11. Criterios de aceptacion

- Existen cinco runners con el contrato comun.
- Hay tres propuestas estructuradas y diversas.
- `debate_round` depende de las tres y contiene criticas explicitas.
- `judge` aparece despues del debate y decide en solitario.
- Se registran cinco pasos y cinco llamadas en el baseline normal.
- Metricas y persistencia usan `benchmark_core`.
- No se rompen ARCH_01 a ARCH_07 y pasa `pytest` completo.

## 12. Desviaciones conocidas

La critica cruzada se implementa como una unica llamada de moderacion que
compara simultaneamente las tres propuestas, en vez de tres llamadas de replica
entre pares. Es una ronda explicita y trazable, limita el coste a cinco llamadas
y evita que el orden de replicas introduzca diferencias entre frameworks.

CrewAI materializa antes del `kickoff` descripciones con referencias de contexto;
las propuestas y la critica reales son inyectadas por `Task.context`. En las
otras implementaciones, los valores estructurados viajan directamente por el
estado o los eventos nativos. Las trazas normalizadas siempre conservan los
valores reales.
