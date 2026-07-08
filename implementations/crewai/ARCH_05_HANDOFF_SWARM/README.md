# ARCH_05_HANDOFF_SWARM - CrewAI

Implementacion con `CrewAI Flow`, listeners y routers para representar las transiciones de handoff.

Cada listener ejecuta la `Task` del agente activo y el router propaga el destino decidido por ese agente. No usa `Process.hierarchical` ni `manager_agent`; esas primitivas pertenecen a `ARCH_04_SUPERVISOR_WORKERS`.
