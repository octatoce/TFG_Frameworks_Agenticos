# ARCH_06_PARALLEL_FANOUT_FANIN - LangGraph

Implementacion con `StateGraph`: `START` distribuye el mismo estado a cuatro nodos independientes y una arista de origen multiple sincroniza las cuatro ramas antes de `aggregator`.

Los nodos de rama escriben en claves de estado aisladas, por lo que no existe dependencia ni comunicacion entre ellos. `aggregator` es el unico punto de fan-in y no inicia iteraciones adicionales.
