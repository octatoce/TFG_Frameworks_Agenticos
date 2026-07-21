# LangGraph

Implementaciones del benchmark usando grafos reales de LangGraph.

Cada arquitectura se modela con:

- `StateGraph`
- nodos;
- edges;
- estado compartido;
- `compile().invoke(...)`

Cada arquitectura expone la misma funcion publica:

```python
run_architecture(input_data, config)
```

## Codigo compartido

La logica comun esta en `utils_langgraph.py`.

Ese archivo crea el LLM instrumentado, mide recursos, centraliza la construccion del `ExperimentResult` y aporta utilidades pequenas para llamadas LLM, ids de documentos y extraccion de respuesta final.

Asi, cada `run.py` mantiene visible la parte importante de LangGraph: nodos, edges y estado.

## Arquitecturas

- `ARCH_01_SINGLE_REACT`: un nodo unico.
- `ARCH_02_SEQUENTIAL_PIPELINE`: pipeline lineal de cuatro nodos.
- `ARCH_03_ROUTER_SPECIALISTS`: grafo con router, specialists y edges condicionales.

## Notas

La refactorizacion mantiene el contrato comun del benchmark. Desde fuera, cada runner devuelve siempre un `ExperimentResult`.
