# ARCH_04_SUPERVISOR_WORKERS - LlamaIndex

Implementacion como workflow supervisado con estado explicito. El supervisor ejecuta steps equivalentes a planificacion, decision/revision, worker y finalizacion.

No crea un RAG externo ni usa memoria persistente. Los documentos proceden de `ExperimentInput` y las iteraciones se limitan con `max_supervisor_iterations`.
