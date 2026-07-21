# ARCH_07_MAP_REDUCE_AGENTIC - CrewAI

Cada batch crea una `Task(async_execution=True, context=[])` con agentes de
configuracion identica. La tarea reducer es sincronica y declara todas las
tareas mapper como `context`, por lo que CrewAI espera sus futuros antes del
fan-in.

No se usan manager, delegacion, routing, handoffs ni fallback secuencial.
