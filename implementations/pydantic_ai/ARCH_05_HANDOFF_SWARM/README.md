# ARCH_05_HANDOFF_SWARM - Pydantic AI

Implementacion con la API moderna `pydantic_graph.GraphBuilder`, cuatro `Step`
especialistas y un `Decision` compartido para los handoffs. Las decisiones de los
agentes siguen validadas como `HandoffDecision` con Pydantic.

El grafo conserva la ejecucion secuencial propia de ARCH_05, los limites de
handoffs y visitas, y la instrumentacion comun. No usa el `Graph` legacy ni un
bucle controlador en Python.

Cada especialista es un paso real del grafo y devuelve una transicion tipada al
`Decision`.
