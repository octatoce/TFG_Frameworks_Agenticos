# ARCH_04_SUPERVISOR_WORKERS

## Objetivo

Evaluar una arquitectura multi-agente centralizada con un supervisor real. El supervisor planifica, decide que workers intervienen, revisa sus salidas, puede pedir revisiones acotadas y finaliza cuando la calidad es suficiente o se alcanza el limite de iteraciones.

La pregunta experimental asociada es:

```text
Que sobrecarga, coste y trazabilidad introduce un supervisor central activo frente a un router simple o a un pipeline fijo?
```

Esta arquitectura no es un pipeline ni un router simple. La decision se reevalua despues de cada salida de worker y queda registrada.

## Componentes Logicos

| Componente | Responsabilidad |
| --- | --- |
| Supervisor | Crea el plan, decide el siguiente worker, revisa salidas, solicita revisiones y finaliza. |
| DataWorker | Extrae datos, evidencias y fragmentos relevantes de los documentos. |
| ReasoningWorker | Analiza la informacion y construye la explicacion tecnica. |
| ValidationWorker | Busca errores, contradicciones, riesgos, falta de evidencia o debilidades. |
| SynthesisWorker | Construye una salida final clara a partir del material aprobado. |

## Restricciones

- Debe existir un supervisor explicito y activo.
- El supervisor genera un plan inicial con workers, tareas, salidas esperadas y criterios de calidad.
- El supervisor revisa al menos una salida de worker cuando existe una salida previa.
- El supervisor puede pedir revisiones, pero siempre con limite de iteraciones.
- El limite por defecto es `max_supervisor_iterations = 3`.
- Si no existe campo especifico en `ExperimentConfig`, el limite se lee de `config.metadata["max_supervisor_iterations"]` o de `MAX_SUPERVISOR_ITERATIONS`.
- No hay memoria persistente ni memoria conversacional global.
- No se usan checkpoints.
- No se usa fan-out/fan-in paralelo.
- No se usan handoffs descentralizados ni debate con juez.
- No se cambia el contrato comun ni la metodologia experimental.

## Flujo Logico

1. Recibir `ExperimentInput` y `ExperimentConfig`.
2. `supervisor_plan` genera `workers_to_run`, `task_assignments`, `expected_outputs` y `quality_criteria`.
3. `supervisor_decision` decide `run_worker`, `request_revision` o `finalize`.
4. El worker indicado ejecuta una tarea concreta.
5. El control vuelve al supervisor, que revisa la salida anterior.
6. El ciclo continua hasta decision de finalizacion, limite de iteraciones o error recuperable.
7. `supervisor_finalize` sintetiza o aprueba la respuesta final.
8. Se devuelve `ExperimentResult` con trazabilidad comun y metadata especifica.

## Estado Compartido Minimo

```python
state = {
    "plan": None,
    "worker_outputs": [],
    "iterations": 0,
    "max_supervisor_iterations": 3,
    "workers_executed": [],
    "revisions_requested": 0,
    "accepted_worker_outputs": [],
    "rejected_worker_outputs": [],
    "stop_reason": None,
    "warnings": [],
}
```

## Pseudocodigo Canonico

```python
def run_architecture(input_data: ExperimentInput, config: ExperimentConfig) -> ExperimentResult:
    state = initialize_supervisor_state(input_data, config)
    plan = supervisor.create_plan(input_data)
    state["plan"] = plan

    while state["iterations"] < max_supervisor_iterations:
        decision = supervisor.decide_next_action(
            input_data=input_data,
            plan=state["plan"],
            worker_outputs=state["worker_outputs"],
            current_iteration=state["iterations"],
        )

        if decision.action == "finalize":
            state["stop_reason"] = decision.stop_reason
            break

        if decision.action == "run_worker":
            output = run_worker(decision.worker_name, decision.task)
            state["worker_outputs"].append(output)

        elif decision.action == "request_revision":
            output = revise_worker(decision.worker_name, decision.revision_instructions)
            state["worker_outputs"].append(output)
            state["revisions_requested"] += 1

        else:
            state["warnings"].append("Invalid supervisor action.")
            state["stop_reason"] = "invalid_supervisor_action"
            break

        state["workers_executed"].append(decision.worker_name)
        state["iterations"] += 1

    if state["stop_reason"] is None:
        state["stop_reason"] = "max_supervisor_iterations_reached"

    final_output = supervisor.finalize(input_data, state)
    return build_experiment_result(...)
```

## Equivalencia Entre Frameworks

| Framework | Implementacion equivalente |
| --- | --- |
| LangGraph | `StateGraph` con nodos de plan, decision, workers y finalizacion; los workers vuelven al supervisor por conditional edges. |
| CrewAI | `Process.hierarchical` con `manager_agent` nativo, workers como agentes de la crew y reconstruccion de trazas desde llamadas/roles ejecutados. |
| Microsoft Agent Framework | Workflow/orquestacion centralizada con supervisor que mantiene estado y ejecuta fases medibles. |
| LlamaIndex | Workflow supervisado con steps equivalentes y estado acotado. |
| Pydantic AI | Flujo graph-shaped con modelos Pydantic para plan, accion, worker output, revision y estado. |

La equivalencia exige:

- Mismos workers logicos.
- Mismo limite de iteraciones.
- Misma estructura de plan y decision.
- Mismo registro de workers ejecutados, revisiones, aceptados/rechazados y razon de parada.
- Sin paralelismo ni memoria persistente.
- Misma salida `ExperimentResult`.

## Metricas Especialmente Relevantes

| Metrica | Motivo |
| --- | --- |
| Numero de workers ejecutados | Mide la cantidad real de delegacion. |
| Iteraciones del supervisor | Mide sobrecarga de coordinacion. |
| Revisiones solicitadas | Mide coste de control de calidad. |
| Workers usados y no usados | Permite distinguir supervisor activo de pipeline fijo. |
| Razon de parada | Explica si finaliza por calidad, limite o error recuperable. |
| Llamadas LLM y tokens | Cuantifica el coste adicional del supervisor central. |
| Latencia por step | Permite separar coste de supervisor y workers. |

## Estado de Implementacion

Implementada para:

- `implementations/langgraph/ARCH_04_SUPERVISOR_WORKERS/run.py`
- `implementations/crewai/ARCH_04_SUPERVISOR_WORKERS/run.py`
- `implementations/microsoft_agent_framework/ARCH_04_SUPERVISOR_WORKERS/run.py`
- `implementations/llamaindex/ARCH_04_SUPERVISOR_WORKERS/run.py`
- `implementations/pydantic_ai/ARCH_04_SUPERVISOR_WORKERS/run.py`

Estado actual:

- El supervisor planifica antes de ejecutar workers.
- Cada worker ejecutado vuelve al supervisor.
- Las decisiones se registran como steps `supervisor_decision`.
- Las revisiones estan soportadas y contabilizadas.
- El modo determinista local no fuerza revisiones por defecto; registra `revisions_requested = 0` cuando no son necesarias.
- La parada por defecto puede ocurrir por `max_supervisor_iterations_reached` si el plan requiere todos los ciclos disponibles.

## Riesgos y Decisiones Pendientes

- Si un modelo real no respeta el formato parseable de plan o decision, los parsers aplican una politica conservadora para evitar ejecuciones vacias.
- El limite de iteraciones se mantiene fuera del schema comun para no modificar el contrato experimental.
- CrewAI se ejecuta con `Process.hierarchical`, porque es la implementacion nativa de manager/supervisor del framework. La trazabilidad se reconstruye desde llamadas LLM, roles ejecutados y outputs de tasks.
