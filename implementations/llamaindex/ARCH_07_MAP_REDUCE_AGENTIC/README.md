# ARCH_07_MAP_REDUCE_AGENTIC - LlamaIndex

Un `Workflow` emite un `MapperEvent` por batch. El unico paso mapper dispone de
workers concurrentes y aplica la misma logica a todos los eventos.
`Context.collect_events` constituye el fan-in antes del reducer.

No se usan routing, handoffs ni fallback secuencial.
