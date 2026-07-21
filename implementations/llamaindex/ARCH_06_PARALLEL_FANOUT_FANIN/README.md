# ARCH_06_PARALLEL_FANOUT_FANIN - LlamaIndex

Implementacion con `Workflow` y eventos. El paso inicial emite cuatro tipos de evento independientes; cuatro pasos especializados los procesan concurrentemente y devuelven `BranchResultEvent`. `Context.collect_events` sincroniza exactamente cuatro resultados antes de ejecutar `aggregator`.

Cada rama usa un `FunctionAgent` real cuando el proveedor es OpenAI y el adaptador instrumentado equivalente en el baseline local. No hay `AgentWorkflow` de handoffs, supervisor ni pipeline lineal.
