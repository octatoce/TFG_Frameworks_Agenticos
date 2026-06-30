# ARCH_04_SUPERVISOR_WORKERS - CrewAI

Implementacion con `Process.hierarchical`, la primitiva nativa de CrewAI para manager/supervisor.

La crew usa un `manager_agent` Supervisor y workers `DataWorker`, `ReasoningWorker`, `ValidationWorker` y `SynthesisWorker`. El benchmark reconstruye plan, decisiones, workers ejecutados, revisiones, iteraciones y razon de parada a partir de las llamadas y roles ejecutados.
