# ARCH_07_MAP_REDUCE_AGENTIC - Microsoft Agent Framework

Un `DocumentPartitionerExecutor` distribuye el input mediante
`WorkflowBuilder.add_fan_out_edges` hacia executors mapper equivalentes, uno por
batch. `add_fan_in_edges` sincroniza sus resultados antes del reducer unico.

No existe fallback secuencial en la version instalada.
