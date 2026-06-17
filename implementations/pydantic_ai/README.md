# Pydantic AI

Implementaciones de la iteracion 2 para Pydantic AI + pydantic-graph.

El baseline ejecutable usa un adaptador determinista local sobre `benchmark_core` y modelos Pydantic para validar salidas intermedias sin alterar el contrato comun. Las dependencias nativas quedan declaradas en `requirements.txt` para ejecuciones futuras con el framework instalado.

Arquitecturas incluidas:

- `ARCH_01_SINGLE_REACT`
- `ARCH_02_SEQUENTIAL_PIPELINE`
- `ARCH_03_SUPERVISOR_WORKERS`

