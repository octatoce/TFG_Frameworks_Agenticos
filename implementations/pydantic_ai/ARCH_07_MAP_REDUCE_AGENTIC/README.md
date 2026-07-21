# ARCH_07_MAP_REDUCE_AGENTIC - Pydantic AI + pydantic-graph

`GraphBuilder` ejecuta un paso particionador cuya lista de batches se expande
con `.map()`. Un unico `Step` mapper tipado procesa cada elemento en paralelo,
un `Join` reune `MapperExecution` y el reducer genera una salida validada.

No se usa `asyncio.gather`, `Graph` legacy ni fallback secuencial.
