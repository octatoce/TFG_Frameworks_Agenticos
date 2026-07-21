# ARCH_06_PARALLEL_FANOUT_FANIN

## 1. Objetivo

Evaluar como cada framework expresa concurrencia real, fan-out/fan-in, sincronizacion y agregacion trazable cuando cuatro agentes analizan el mismo caso desde perspectivas independientes.

La arquitectura conserva el contrato comun:

```python
run_architecture(input_data: ExperimentInput, config: ExperimentConfig) -> ExperimentResult
```

## 2. Descripcion general

`ExperimentInput` entra una sola vez y se difunde, sin particionarlo, a cuatro ramas. Las ramas pueden ejecutarse simultaneamente porque ninguna consume resultados de otra. Cuando las cuatro terminan o el timeout interrumpe la ejecucion, un unico agregador recibe todas las salidas parciales y produce el `structured_output` final.

```text
                         -> factual_analysis_branch ---------
                         -> technical_reasoning_branch ------
common ExperimentInput   -> risk_constraints_branch --------- > aggregator -> ExperimentResult
                         -> alternative_solution_branch -----
```

No existe router selectivo, supervisor iterativo, handoff, debate ni particionado documental.

## 3. Componentes

| Componente | Responsabilidad | Dependencias previas |
| --- | --- | --- |
| `factual_analysis_branch` | Extraer hechos, evidencias, citas internas y fragmentos objetivos. | Ninguna. |
| `technical_reasoning_branch` | Proponer una solucion o decision preliminar desde el razonamiento tecnico. | Ninguna. |
| `risk_constraints_branch` | Identificar riesgos, limitaciones, contradicciones, incertidumbre y bordes. | Ninguna. |
| `alternative_solution_branch` | Proponer una interpretacion distinta o enfoque complementario. | Ninguna. |
| `aggregator` | Fusionar las cuatro salidas, deduplicar, resolver contradicciones e integrar riesgos y alternativas. | Las cuatro ramas. |

Cada rama produce una salida parcial normalizada con `analysis`, `key_points`, `evidence`, `risks`, `alternatives`, `raw_output` y `error`. El agregador produce `answer`, contradicciones resueltas, riesgos integrados y alternativas consideradas.

## 4. Flujo de ejecucion

1. Se crea un estado o mensaje con el mismo `ExperimentInput` comun.
2. El framework activa las cuatro ramas independientes.
3. Cada rama realiza una llamada LLM instrumentada y registra un `AgentStep`.
4. El fan-in espera las cuatro salidas; una rama no puede desbloquear el agregador por si sola.
5. El agregador recibe un diccionario que referencia las cuatro salidas y realiza una unica llamada LLM.
6. Se construye `ExperimentResult`, se calculan metricas comunes y se guarda el JSON canonico.

El orden canonico de trazas es el de la tabla anterior y despues `aggregator`, aunque el orden real de finalizacion de las ramas pueda variar.

## 5. Pseudocodigo

```python
function run_architecture(input_data, config):
    initialize state from the single common input

    launch independently:
        factual_analysis_branch(input_data)
        technical_reasoning_branch(input_data)
        risk_constraints_branch(input_data)
        alternative_solution_branch(input_data)

    wait for all branches or timeout
    collect the four partial outputs

    aggregator_output = aggregator(
        factual_analysis_branch_output,
        technical_reasoning_branch_output,
        risk_constraints_branch_output,
        alternative_solution_branch_output,
    )

    build structured_output
    build ExperimentResult with benchmark_core
    save results/raw/{framework}/{architecture}/{run_id}.json
    return ExperimentResult
```

## 6. Restricciones

- Las ramas reciben el mismo input base y tienen `depends_on=[]`.
- No se transfieren mensajes entre ramas.
- El agregador es el unico fan-in y ejecuta una sola ronda.
- No se permite supervisor, manager jerarquico, delegacion, handoff o debate.
- No se divide la coleccion documental en chunks o batches.
- No se habilitan memoria persistente, checkpoints, cache o observabilidad externa.
- Se mantienen modelo, temperatura, timeout, maximo de iteraciones, herramientas y prompts funcionales comunes.
- Cada componente usa una llamada LLM comparable; el total esperado es cinco.

## 7. Diferencias con otras arquitecturas

- `ARCH_02_SEQUENTIAL_PIPELINE`: sus fases dependen de resultados previos; ARCH_06 usa perspectivas simultaneas sin dependencias.
- `ARCH_03_ROUTER_SPECIALISTS`: selecciona una ruta o subconjunto; ARCH_06 ejecuta siempre las cuatro ramas.
- `ARCH_04_SUPERVISOR_WORKERS`: planifica, revisa y puede iterar; ARCH_06 solo sincroniza y agrega una vez.
- `ARCH_05_HANDOFF_SWARM`: transfiere control entre agentes; ARCH_06 no transfiere control.
- `ARCH_07_MAP_REDUCE_AGENTIC`: divide documentos en lotes equivalentes; ARCH_06 no particiona el input.

## 8. Equivalencia entre frameworks

| Framework | Fan-out | Fan-in | Agentes/componentes |
| --- | --- | --- | --- |
| LangGraph | Aristas desde `START` a cuatro nodos de `StateGraph`. | Arista de origen multiple hacia `aggregator`. | Cuatro nodos de rama y un nodo agregador. |
| Microsoft Agent Framework | `WorkflowBuilder.add_fan_out_edges`. | `WorkflowBuilder.add_fan_in_edges`/`FanInEdgeGroup`. | Cuatro `Executor` que invocan agentes y un `AggregatorExecutor`. |
| CrewAI | Cuatro `Task(async_execution=True, context=[])`. | Tarea agregadora sincronica con `context` igual a las cuatro tareas. | Cinco `Agent`, delegacion desactivada. |
| LlamaIndex | Un paso `Workflow` emite cuatro eventos especializados. | `Context.collect_events` espera cuatro `BranchResultEvent`. | Cuatro pasos con `FunctionAgent`/adaptador y un paso agregador. |
| Pydantic AI + pydantic-graph | Broadcast fork de `GraphBuilder` hacia cuatro `Step`. | `Join` tipado con reducer y paso `aggregator`. | Agentes Pydantic AI dentro de pasos tipados. |

## 9. Metricas relevantes

Las metricas comunes se conservan en `ExperimentMetrics`, `AgentStep` y `LLMCallMetrics`. ARCH_06 añade metadata compatible hacia atras:

- `latency_total_ms` en `ExperimentMetrics.metadata`;
- latencia, llamadas LLM, tokens de entrada/salida y error por rama;
- metricas equivalentes del agregador;
- `branches_completed` y `branches_failed`;
- `parallelism_used` y `fallback_sequential`;
- primitiva nativa usada;
- `structured_output` final y las cuatro salidas parciales.

No se añaden campos obligatorios ni se modifica `schema_version`.

## 10. Criterios de aceptacion

- Existen cinco implementaciones y todas exponen el contrato comun.
- Todas usan `benchmark_core`, devuelven `ExperimentResult` y persisten el JSON canonico.
- Las cuatro ramas aparecen antes del agregador en la traza normalizada.
- Cada rama referencia solo el input comun y ninguna salida hermana.
- El agregador referencia exactamente las cuatro salidas parciales.
- Se registran cinco pasos y cinco llamadas LLM en el baseline local correcto.
- Los limites configurados y el timeout se propagan a las primitivas disponibles.
- Los tests contractuales, estructurales y previos pasan y ARCH_07 permanece como arquitectura separada.

## 11. Desviaciones conocidas

No existe fallback secuencial en las versiones instaladas y verificadas.

CrewAI materializa la concurrencia mediante su executor de tareas asincromas, que usa futuros internos, y bloquea antes de la tarea sincronica agregadora. El prompt del agregador referencia las cuatro claves canonicas y CrewAI inyecta los valores reales mediante `Task.context`.

Los adaptadores OpenAI de Microsoft, LlamaIndex y Pydantic AI se invocan desde threads gestionados por los runtimes asincronos para evitar bloquear sus event loops, manteniendo los agentes/componentes nativos y la instrumentacion comun. Esto no introduce ramas, herramientas ni llamadas adicionales.
