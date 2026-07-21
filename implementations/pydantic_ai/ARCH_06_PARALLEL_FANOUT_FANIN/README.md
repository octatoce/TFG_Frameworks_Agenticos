# ARCH_06_PARALLEL_FANOUT_FANIN - Pydantic AI + pydantic-graph

Implementacion tipada con la API moderna `pydantic_graph.GraphBuilder`. El nodo inicial abre un broadcast fork hacia cuatro `Step` especializados; un `Join` con reducer espera las cuatro salidas `BranchExecution`; `aggregator` valida y produce `ParallelGraphOutput`.

Las ramas invocan agentes Pydantic AI reales/adaptadores tipados y se ejecutan como tareas concurrentes del runtime del grafo. No se usa el `Graph` legacy, un bucle Python, handoffs ni supervision iterativa.
