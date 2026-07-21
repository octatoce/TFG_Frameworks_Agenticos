# ARCH_07_MAP_REDUCE_AGENTIC - LangGraph

`StateGraph` ejecuta `document_partitioner`, crea un `Send` dinamico por batch,
acumula las salidas equivalentes de `mapper` y activa un unico `reducer`.

El particionado es determinista y el reducer recibe solo salidas parciales, no
el contenido completo original.
