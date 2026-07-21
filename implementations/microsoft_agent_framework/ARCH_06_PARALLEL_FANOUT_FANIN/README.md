# ARCH_06_PARALLEL_FANOUT_FANIN - Microsoft Agent Framework

Implementacion con `WorkflowBuilder`, un executor de entrada comun, cuatro executors de rama y un executor agregador. `add_fan_out_edges` difunde el mismo `ExperimentInput` y ejecuta las ramas concurrentemente; `add_fan_in_edges` sincroniza y entrega una lista con las cuatro salidas al agregador.

No existe fallback secuencial en la version soportada (`agent-framework-core` con `FanOutEdgeGroup` y `FanInEdgeGroup`). Cada executor de rama invoca un agente real/adaptador instrumentado del framework.
