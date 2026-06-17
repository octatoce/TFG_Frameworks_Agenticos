# LlamaIndex

Implementaciones de la iteracion 2 para LlamaIndex Workflows / AgentWorkflow.

El baseline ejecutable usa un adaptador determinista local sobre `benchmark_core` para mantener la comparabilidad con LangGraph y CrewAI sin requerir credenciales ni dependencias externas durante los tests. La dependencia nativa queda declarada en `requirements.txt` para ejecuciones futuras con el framework instalado.

Arquitecturas incluidas:

- `ARCH_01_SINGLE_REACT`
- `ARCH_02_SEQUENTIAL_PIPELINE`
- `ARCH_03_SUPERVISOR_WORKERS`

