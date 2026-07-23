# ARCH_09_REFLECTION_CRITIC_LOOP

## Objetivo

Esta arquitectura mide si mejorar iterativamente una única respuesta mediante crítica y revisión compensa el coste adicional en llamadas, tokens y latencia. No genera propuestas independientes, no incorpora un juez y no delega la planificación a un supervisor.

## Contrato

Las cinco implementaciones exponen:

```python
run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
) -> ExperimentResult
```

El resultado se construye con `benchmark_core`, conserva los schemas comunes y se guarda en `results/raw/{framework}/ARCH_09_REFLECTION_CRITIC_LOOP/{run_id}.json`.

## Componentes y flujo

1. `generator` crea la versión inicial a partir de la consulta, documentos y metadata.
2. `critic` evalúa la versión actual y produce una crítica estructurada.
3. `stop_controller` aplica una política determinista y trazable.
4. Si se debe continuar, `reviser` crea una versión corregida y devuelve el control a `critic`.
5. Si se debe parar, la versión actual se convierte en el `structured_output` final.

El ciclo efectivo es:

```text
generator -> critic -> stop_controller --stop--> final
                  ^          |
                  |          +--continue--> reviser
                  +-----------------------------+
```

No se usa `while True`; el número de críticas nunca puede superar el límite configurado. El generator se ejecuta una vez. Cada iteración ejecuta exactamente un critic y un stop_controller, y solo ejecuta reviser cuando el controlador decide continuar.

## Configuración y límite

Se reutiliza `ExperimentConfig.max_agent_iterations` como límite superior compatible. El límite específico se resuelve, por orden, desde:

1. `config.metadata["reflection_max_iterations"]`;
2. el alias compatible `config.metadata["max_iterations"]`;
3. el valor por defecto `2`.

El valor se valida como entero positivo y se recorta a `config.max_agent_iterations`. El umbral se lee desde `config.metadata["reflection_quality_threshold"]` y vale `0.85` por defecto.

## Política de parada

El `stop_controller` no llama al LLM. Aplica esta prioridad estable:

1. `no_critical_issues=true` -> `quality_sufficient`;
2. `critic.score >= quality_threshold` -> `quality_score_threshold_reached`;
3. severidad `minor` o `none` -> `minor_issues_only`;
4. iteración actual igual al máximo -> `max_iterations_reached`;
5. en cualquier otro caso -> `continue_revision`.

Evaluar primero la calidad permite distinguir una respuesta que alcanza el criterio justo en la última iteración de una ejecución agotada sin calidad suficiente.

## Estructuras intermedias

`benchmark_core.reflection_critic_loop` valida:

- `ReflectionVersion`: respuesta, evidencia, confianza, limitaciones y cambios aplicados;
- `CritiqueEvaluation`: score, severidad, defectos, mejoras y huecos de evidencia;
- `StopDecision`: decisión, motivo, umbral y flags de terminación;
- `ReflectionSettings`: máximo de iteraciones y umbral de calidad.

El resultado incluye versión inicial, versión final, historial completo, resúmenes de versiones, críticas, decisiones de parada y `reflection_execution`.

## Trazas y métricas

Los pasos canónicos son:

- `generator`;
- `critic_NNN`;
- `stop_controller_NNN`;
- `reviser_NNN`, solo cuando continúa el ciclo.

`reflection_execution` registra:

- iteraciones, revisiones y versiones;
- motivo de parada y flags de calidad/límite;
- latencias, llamadas, tokens y errores por componente e iteración;
- progresión y delta de scores del critic;
- versión final seleccionada;
- primitiva de orquestación del framework.

El resumen se copia también a `ExperimentMetrics.metadata`. El coste total, tokens totales, latencia total y número global de llamadas siguen calculándose mediante los builders comunes.

## Equivalencia por framework

- **LangGraph:** `StateGraph` con arista condicional desde `stop_controller` y retorno explícito `reviser -> critic`.
- **Microsoft Agent Framework:** `WorkflowBuilder`, executors diferenciados y `add_switch_case_edge_group` para terminar o revisar.
- **CrewAI:** agentes y tareas nativas de una sola llamada por componente; un `for` externo acotado materializa el ciclo porque la API de Crew no ofrece una transición cíclica limpia sin introducir manager o planificación.
- **LlamaIndex:** `Workflow` con eventos de estado, crítica y solicitud de revisión; `StopEvent` termina el flujo.
- **Pydantic AI + pydantic-graph:** `GraphBuilder`, `Step` y `Decision` tipados, con retorno de reviser al paso critic.

## Restricciones

No se usan debate, juez, supervisor, routing, handoffs, Map-Reduce, fan-out, memoria persistente, planificación implícita ni herramientas específicas de un framework. Los reintentos automáticos no forman parte del controlador de reflexión y toda llamada relevante queda asociada a un `AgentStep`.

## Baseline determinista de tests

Con el límite por defecto de dos iteraciones, el modelo local produce una crítica mayor en la primera iteración, una revisión y una segunda crítica suficiente. El resultado esperado es cuatro llamadas LLM, seis pasos, dos versiones y parada `quality_sufficient`. Los tests también fijan el límite a uno para verificar la ruta `max_iterations_reached` sin ejecutar reviser.
