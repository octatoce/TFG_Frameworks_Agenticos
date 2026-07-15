# ARCH_05_HANDOFF_SWARM

## Nombre de arquitectura

`ARCH_05_HANDOFF_SWARM`

## Objetivo

Evaluar una arquitectura multi-agente descentralizada donde el agente especialista activo decide si finaliza o transfiere el control a otro especialista mediante handoff.

## Entrada

La entrada es el contrato comun:

```python
run_architecture(input_data: ExperimentInput, config: ExperimentConfig) -> ExperimentResult
```

Se usan los mismos documentos, modelo, temperatura, timeout y herramientas funcionales comunes del benchmark.

## Salida esperada

La salida debe incluir `ExperimentResult` con `structured_output` que contenga, como minimo:

- `answer`
- `decision`
- `confidence`
- `evidence`
- `limitations`
- `initial_agent`
- `active_agent_history`
- `handoff_history`
- `number_of_handoffs`
- `number_of_agent_invocations`
- `finalizing_agent`
- `stop_reason`

## Componentes/agentes

| Agente | Responsabilidad |
| --- | --- |
| DataSpecialist | Extrae evidencia documental y decide a quien transferir. |
| ReasoningSpecialist | Razona sobre evidencia/contexto y decide si validar, sintetizar o finalizar. |
| ValidationSpecialist | Busca riesgos, contradicciones y limitaciones. |
| SynthesisSpecialist | Produce la respuesta final cuando recibe contexto suficiente. |

Puede existir una seleccion inicial del primer agente, pero no planifica la secuencia completa ni mantiene control central.

## Semántica común

Cada agente activo devuelve una `HandoffDecision`:

```python
{
    "action": "handoff" | "finalize",
    "target_agent": "agent_id_or_none",
    "reason": "...",
    "task": "...",
    "context_summary": "...",
    "final_output": "...",
    "confidence": 0.0,
}
```

Si `action == "handoff"`, el destino debe ser valido y se registra `source_agent`, `target_agent`, razon, tarea y contexto transferido.

Si `action == "finalize"`, el agente activo queda registrado como `finalizing_agent`.

## Pseudocódigo

```python
active_agent = choose_initial_agent(input_data)
state = SwarmState(active_agent=active_agent)

while not stopped:
    if limits_reached(state):
        finalize_with_fallback()
        break

    decision = active_agent.run(input_data, state)
    state.active_agent_history.append(active_agent)

    if invalid(decision):
        apply_fallback_policy()
    elif decision.action == "finalize":
        final_output = decision.final_output
        break
    else:
        record_handoff(active_agent, decision.target_agent)
        active_agent = decision.target_agent
```

## Herramientas permitidas

Solo las herramientas comunes ya disponibles en el benchmark. No se anaden capacidades funcionales exclusivas para un framework.

## Memoria permitida

Solo estado interno durante una ejecucion:

- agente activo;
- historial de agentes;
- historial de handoffs;
- resultados parciales;
- contexto transferido;
- contadores;
- warnings;
- razon de parada.

No hay memoria persistente, checkpoints ni recuperacion propia de arquitecturas posteriores.

## Paralelismo permitido

No se permite paralelismo. ARCH_05 representa transferencia secuencial de control.

## Criterios de parada

- `action == finalize`.
- Se alcanza `max_handoffs` (por defecto `4`).
- Se alcanza `max_agent_invocations` (por defecto `6`).
- Se supera `max_consecutive_visits_per_agent` (por defecto `2`).
- Decision invalida con fallback.
- Error no recuperable.
- Ciclo detectado con limite/fallback.

## Métricas a recoger

- `initial_agent`
- `active_agent_history`
- `handoff_history`
- `number_of_handoffs`
- `max_handoffs`
- `number_of_agent_invocations`
- `unique_agents_executed`
- `finalizing_agent`
- `repeated_agent_visits`
- `cycle_detected`
- `fallback_used`
- `stop_reason`
- `framework_native_primitives`
- `native_automatic_behaviors`
- `warnings`

Cada llamada LLM de agente activo se registra como `AgentStep` y `LLMCallMetrics`.

## Qué NO puede hacer la implementación

- No puede tener supervisor central.
- No puede usar manager de ARCH_04.
- No puede planificar toda la secuencia al inicio.
- No puede ser pipeline fijo.
- No puede ser router simple.
- No puede ejecutar en paralelo.
- No puede usar checkpoints ni memoria persistente.
- No puede ocultar la orquestacion completa en utils.

## Implementación en LangGraph

Usa `StateGraph` con nodos especialistas y `conditional_edges` directos entre agentes. Cada nodo decide el siguiente nodo o `END`.

## Implementación en Microsoft Agent Framework

Usa `HandoffBuilder` de `agent-framework-orchestrations` con agentes `Agent` reales, `with_start_agent`, `add_handoff`, herramientas nativas `handoff_to_*` y eventos `handoff_sent`. No usa supervisor central ni group chat.

Auditoria nativa: se instala `agent-framework-orchestrations==1.0.0` y se verifica que el workflow emite `handoff_sent` cuando un agente invoca la herramienta `handoff_to_<target>`. El cliente benchmark conserva el esquema comun `HandoffDecision` y lo adapta a tool call nativa para no perder medicion de llamadas, tokens, coste y latencia.

## Implementación en CrewAI

Usa `CrewAI Flow` con estado tipado, `@start`, `@listen` y `@router` para materializar las transiciones. Cada listener ejecuta una `Task` de un unico `Agent` activo mediante CrewAI; el router solo propaga el destino decidido por el output del especialista. No usa `Process.hierarchical` ni `manager_agent`, reservados para ARCH_04.

## Implementación en LlamaIndex

Usa `FunctionAgent` especialistas cuando hay proveedor real y un adaptador determinista equivalente para los tests locales. La transferencia se mantiene en un workflow explicito instrumentado para conservar control de `max_handoffs`, `max_agent_invocations`, tokens, coste y pasos.

Auditoria nativa: `AgentWorkflow` y `can_handoff_to` estan disponibles en `llama-index-core==0.14.22`. No se usa como implementacion principal porque el modo local del benchmark no proporciona un `BaseWorkflowAgent`/LLM function-calling compatible con el handoff tool de AgentWorkflow, y ejecutarlo solo para OpenAI introduciria una capacidad distinta y saltaria la instrumentacion comun de llamadas/tokens. Esta implementacion no se documenta como handoff nativo de `AgentWorkflow`.

## Implementación en Pydantic AI y pydantic-graph

Usa un grafo ejecutable real de `pydantic-graph` con `BaseNode`, `GraphRunContext`, `Graph` y `End`. Cada especialista es un nodo registrado que devuelve otro nodo especialista o `End` con la salida final. La validacion de `HandoffDecision` se mantiene con modelos Pydantic.

## Primitivas nativas utilizadas

| Framework | Primitivas |
| --- | --- |
| LangGraph | `StateGraph`, nodos especialistas, `conditional_edges`. |
| Microsoft Agent Framework | `HandoffBuilder`, `HandoffConfiguration`, `with_start_agent`, `add_handoff`, herramientas `handoff_to_*`, eventos `handoff_sent`. |
| CrewAI | `Flow`, `@start`, `@listen`, `@router`, `Agent`, `Task`. |
| LlamaIndex | `FunctionAgent`/adaptador local y workflow explicito; `AgentWorkflow.can_handoff_to` evaluado pero no usado por bloqueo de instrumentacion/local baseline. |
| Pydantic AI | `pydantic_graph.Graph`, `BaseNode`, `GraphRunContext`, `End`, modelos Pydantic. |

## Diferencia con ARCH_03_ROUTER_SPECIALISTS

ARCH_03 tiene router central que selecciona especialistas al inicio. ARCH_05 no tiene router persistente: el agente activo decide localmente el siguiente handoff.

## Diferencia con ARCH_04_SUPERVISOR_WORKERS

ARCH_04 tiene supervisor central que planifica, revisa y controla la calidad. ARCH_05 no tiene aprobacion central obligatoria; la responsabilidad se desplaza entre especialistas mediante handoffs.
