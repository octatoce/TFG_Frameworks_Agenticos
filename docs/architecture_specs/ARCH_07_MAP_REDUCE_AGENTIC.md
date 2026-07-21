# ARCH_07_MAP_REDUCE_AGENTIC

## 1. Objetivo

Evaluar la capacidad de cada framework para escalar sobre carga documental
mediante particionado determinista, mappers equivalentes concurrentes y una
reduccion global trazable.

## 2. Descripcion general

ARCH_07 recibe un `ExperimentInput`, conserva el orden original de sus
documentos, los divide en batches contiguos y aplica exactamente el mismo
mapper a cada batch. Un unico reducer consume las salidas parciales y genera la
respuesta final.

El fan-out se realiza por volumen documental, no por perspectivas o roles.

## 3. Componentes

### document_partitioner

- Lee `input_data.documents` una sola vez.
- Usa `config.metadata["map_reduce_batch_size"]`; acepta `batch_size` como alias
  compatible y usa `3` por defecto.
- Preserva el orden documental y crea identificadores `batch_001`,
  `batch_002`, etc.
- Registra documentos totales, tamano de batch, numero de batches y sus IDs.
- No realiza ninguna llamada LLM.

### mapper

- Todos los mappers comparten instrucciones, prompt y schema de salida.
- Cada ejecucion recibe solo un `DocumentBatch`.
- Produce respuesta parcial, hallazgos, evidencias y limitaciones.
- No accede a otros batches ni se comunica con otros mappers.

### reducer

- Recibe todas las salidas parciales ordenadas por indice de batch.
- Deduplica hallazgos, resuelve contradicciones y prioriza evidencias.
- No recibe de nuevo el contenido completo de los documentos.
- Realiza una unica llamada LLM y genera el `structured_output` final.

## 4. Flujo de ejecucion

1. Validar el `batch_size`.
2. Particionar los documentos de forma estable y contigua.
3. Emitir una unidad mapper equivalente por batch.
4. Ejecutar los mappers concurrentemente cuando hay mas de un batch.
5. Esperar todas las salidas o el timeout comun.
6. Ordenar las salidas por `batch_index`.
7. Ejecutar una vez el reducer.
8. Construir metricas, `ExperimentResult` y resultado raw.

## 5. Pseudocodigo

```text
function run_architecture(input_data, config):
    batch_size = config.metadata.map_reduce_batch_size or 3
    partition_step, batches = deterministic_partition(input_data.documents, batch_size)

    mapper_results = map_concurrently(
        same_mapper,
        each batch in batches
    )

    ordered_results = sort_by_batch_index(mapper_results)
    reducer_result = reducer(ordered_results)

    steps = [partition_step, mapper_steps..., reducer_step]
    structured_output = build_common_map_reduce_output(...)
    result = build_experiment_result(...)
    save_result_json(result)
    return result
```

## 6. Restricciones

- No existe mapper global sobre todos los documentos.
- Los mappers no tienen roles especializados diferentes.
- No hay routing, debate, handoffs ni supervisor iterativo.
- El reducer no inicia nuevas rondas y no relee documentos completos.
- El particionado no depende del framework ni del modelo.
- Se mantienen modelo, temperatura, timeout, herramientas y schemas comunes.

Si no hay documentos, se crea un batch vacio trazable para conservar el
contrato mapper-reducer y producir una limitacion explicita.

## 7. Diferencias con otras arquitecturas

- ARCH_02 ejecuta fases funcionalmente distintas y dependientes; ARCH_07 aplica
  la misma funcion a particiones independientes.
- ARCH_03 selecciona rutas; ARCH_07 procesa todos los batches.
- ARCH_04 revisa e itera; ARCH_07 reduce una sola vez.
- ARCH_05 transfiere control; ARCH_07 no usa handoffs.
- ARCH_06 envia el mismo input a cuatro perspectivas distintas; ARCH_07 envia
  documentos distintos a mappers equivalentes.

## 8. Equivalencia entre frameworks

| Framework | Map | Fan-in / Reduce | Paralelismo |
| --- | --- | --- | --- |
| LangGraph | `StateGraph` y `Send` dinamico por batch. | Acumulador de estado y nodo `reducer`. | Nativo por superstep. |
| Microsoft Agent Framework | `WorkflowBuilder.add_fan_out_edges` hacia `MapperExecutor` equivalentes. | `add_fan_in_edges` hacia `ReducerExecutor`. | Nativo en workflow. |
| CrewAI | `Task(async_execution=True, context=[])` por batch. | Tarea reducer con todos los mappers como `context`. | Futuros async nativos. |
| LlamaIndex | `MapperEvent` dinamico y un paso mapper con multiples workers. | `Context.collect_events` y paso reducer. | Workers concurrentes de Workflow. |
| Pydantic AI + pydantic-graph | `GraphBuilder.map()` sobre `DocumentBatch` tipado. | `Join` tipado y `Step` reducer. | Map fork nativo. |

No se necesita fallback secuencial con las versiones instaladas. En una
ejecucion con un solo batch, `parallelism_used=false` porque no existe trabajo
simultaneo, aunque la primitiva soporte concurrencia.

## 9. Metricas relevantes

- `latency_total_ms`.
- `total_documents`, `batch_size`, `batch_count` y `mapper_count`.
- Latencia, llamadas, tokens, error, documentos y tamano de salida por mapper.
- Latencia, llamadas y tokens del reducer.
- `batches_completed` y `batches_failed`.
- `throughput_docs_per_second`.
- `parallelism_used` y `fallback_sequential`.

Estas metricas se almacenan en `structured_output.map_reduce_execution` y se
copian a `ExperimentMetrics.metadata.map_reduce_execution`, sin cambiar schemas.

## 10. Criterios de aceptacion

- Existen cinco implementaciones con el contrato comun.
- El particionado de 7 documentos con `batch_size=3` produce 3, 3 y 1
  documentos en orden.
- Existe un paso `document_partitioner`, un mapper por batch y un reducer final.
- Cada documento aparece en exactamente un mapper.
- El reducer depende de todos los mappers y no incluye documentos originales.
- Las trazas y metricas se construyen con `benchmark_core`.
- El JSON raw se guarda en la ruta comun y todos los tests pasan.

## 11. Desviaciones conocidas

No hay fallback secuencial en las versiones auditadas. CrewAI construye las
tareas despues de calcular los batches porque su grafo de tareas se materializa
antes de `kickoff`; la ejecucion mapper sigue siendo concurrente y nativa.

En ejecuciones OpenAI, los cinco adaptadores normalizan el objeto `usage` real
del proveedor como `openai_usage`. Si un SDK no expone esos datos, la ejecucion
falla de forma explicita en vez de introducir una estimacion no comparable. El
proxy por palabras queda limitado al baseline local determinista.
