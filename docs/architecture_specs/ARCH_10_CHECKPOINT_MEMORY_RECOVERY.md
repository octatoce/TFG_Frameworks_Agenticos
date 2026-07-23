# ARCH_10_CHECKPOINT_MEMORY_RECOVERY

## Objetivo

Esta arquitectura mide la capacidad de conservar estado útil, interrumpir un workflow de forma controlada y continuar desde un checkpoint verificable. No evalúa mejora iterativa, debate, routing, handoffs, supervisión, fan-out ni Map-Reduce.

## Contrato y flujo común

Las cinco implementaciones exponen el contrato común:

```python
run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
) -> ExperimentResult
```

El flujo normalizado contiene siete `AgentStep` en este orden:

```text
state_initializer
  -> planning_or_analysis_step
  -> checkpoint_writer
  -> failure_injector
  -> recovery_loader
  -> continuation_step
  -> finalizer
```

`planning_or_analysis_step` y `continuation_step` son las dos únicas llamadas LLM. El resto es lógica determinista, de persistencia o de validación. El resultado se construye con `benchmark_core` y se guarda en `results/raw/{framework}/ARCH_10_CHECKPOINT_MEMORY_RECOVERY/{run_id}.json`.

## Estado y checkpoint

`RecoveryWorkflowState` conserva:

- consulta, documentos y metadata del input;
- análisis preliminar estructurado;
- etapa actual;
- `checkpoint_id`, etapa y timestamp;
- digest SHA-256 del estado serializado;
- flags y motivo de recuperación;
- resultado de continuación y evidencia de finalización posterior a la recuperación.

La frontera lógica común se sitúa después de `planning_or_analysis_step`. El identificador lógico es estable por `run_id` y termina en `checkpoint-001`. Antes de guardar, el estado se serializa de forma canónica y se sella con SHA-256. `recovery_loader` exige análisis, identidad, etapa y digest válidos; una alteración del JSON impide continuar.

## Fallo controlado

El fallo nominal está habilitado por defecto y puede desactivarse con:

```python
config.metadata["checkpoint_inject_failure"] = False
```

Cuando está habilitado, `failure_injector` lanza `ControlledFailure` exactamente una vez después del checkpoint. El runner captura esa excepción, recupera el estado y continúa. El error esperado queda en `AgentStep.error`, `controlled_error_count=1` y `failure_stage=failure_injector`. Como la recuperación es correcta, el `ExperimentResult` termina con estado `success`; no se registra como `ExperimentError` no recuperable.

## Structured output y métricas

`structured_output.recovery_execution` se copia a `ExperimentMetrics.metadata.recovery_execution` e incluye:

- backend y naturaleza nativa o fallback del checkpoint;
- checkpoint lógico y, cuando existe, identificador nativo;
- número lógico y número nativo de snapshots;
- etapa, timestamp y digest verificado;
- fallo inyectado, errores controlados y no controlados;
- intento, resultado, fuente y motivo de recuperación;
- pasos antes del fallo y después de recuperar;
- latencia previa, escritura, carga y fase posterior;
- llamadas y tokens por componente;
- confirmación de que el resultado se generó después de `recovery_loader`.

El conteo `checkpoints_created=1` representa la frontera metodológica común. LangGraph y Microsoft pueden crear snapshots técnicos adicionales por superstep; se informan separadamente como `native_checkpoints_created`.

## Equivalencia por framework

| Framework | Workflow | Checkpoint | Recuperación |
| --- | --- | --- | --- |
| LangGraph | `StateGraph` lineal | `SqliteSaver` nativo y durable local | Se cierra la conexión SQLite, se abre una nueva, se recompila el grafo y se reanuda desde el snapshot persistido. |
| Microsoft Agent Framework | `WorkflowBuilder` con siete executors | `FileCheckpointStorage` nativo | `workflow.run(checkpoint_id=...)` restaura mensajes, estado y posición del workflow. |
| CrewAI | dos `Agent`/`Task`/`Crew` secuenciales y aislados | JSON local tipado de `benchmark_core` | Carga y valida el estado antes de crear la tarea de continuación. |
| LlamaIndex | dos fases `Workflow` enlazadas | serialización nativa `Context.to_dict()` guardada en JSON | `Context.from_dict()` recupera el payload del store; una segunda fase continúa desde él. |
| Pydantic AI + pydantic-graph | dos grafos tipados `GraphBuilder` | `PortableCheckpoint` Pydantic en JSON | El segundo grafo valida y carga el modelo antes de `continuation_step`. |

## Desviaciones y alcance de persistencia

- LangGraph instala `langgraph-checkpoint-sqlite==3.1.0`. El runner usa un fichero SQLite local, cierra la conexión tras el fallo y recupera con otra conexión, por lo que ya no depende del estado Python en memoria. SQLite es adecuado para esta prueba local, no se presenta como backend de producción; una validación productiva requeriría PostgreSQL u otro backend durable y concurrente.
- Microsoft usa persistencia nativa en fichero y es la única variante de esta matriz que restaura directamente el cursor completo del mismo workflow desde almacenamiento durable.
- LlamaIndex serializa de forma nativa el `Context`, pero la reanudación se expresa en dos fases para que el fallo determinista no se vuelva a consumir desde la cola serializada.
- CrewAI dispone en la versión instalada de checkpointing genérico del runtime, pero serializa la entidad completa y se dispara por eventos. Para no persistir agentes, clientes LLM o automatismos distintos de los otros frameworks, ARCH_10 usa un fallback JSON explícito alrededor de las tareas.
- `pydantic-graph` no ofrece un backend durable equivalente; se usa estado y checkpoint Pydantic validados, con dos grafos explícitos. Pydantic AI sí integra ejecución durable mediante plataformas externas como Temporal, DBOS o Prefect, pero medirlas aquí exigiría introducir un orquestador y una infraestructura que los otros runners no usan. Por ello no se etiqueta el fallback como checkpointing nativo completo.

## Smoke OpenAI

`scripts/run_arch10_openai_smoke.py` ejecuta el mismo caso sintético con los cinco frameworks y exige:

- siete pasos normalizados y dos llamadas reales a OpenAI;
- un checkpoint lógico, un fallo controlado y recuperación exitosa;
- tokens obtenidos de `openai_usage`;
- digest del estado verificado y finalización posterior a `recovery_loader`;
- persistencia SQLite y reapertura de la base en la variante LangGraph.

El script imprime una tabla comparable con backend, naturaleza nativa, latencia total, escritura, recuperación, fase posterior, llamadas y tokens.

No se depende de LangSmith, CrewAI Cloud, observabilidad externa ni memoria conversacional persistente.

## Baseline determinista

Los tests nominales habilitan el fallo. Cada framework produce siete pasos, dos llamadas LLM, un checkpoint lógico, un error controlado, recuperación exitosa y una respuesta final generada después de la recuperación. También se prueba el round-trip del checkpoint portable y el rechazo de un estado manipulado.
