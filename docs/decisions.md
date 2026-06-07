# Decisiones y desviaciones

Este documento registra decisiones tecnicas relevantes y desviaciones necesarias respecto a la metodologia comun.

## Iteracion 1

- Se crea un monorepo unico.
- Se define `benchmark_core` como paquete comun para contratos, metricas y persistencia.
- `ARCH_01_SINGLE_REACT` queda implementada en LangGraph y CrewAI con un agente/nodo decisor unico.
- `ARCH_02_SEQUENTIAL_PIPELINE` queda implementada en LangGraph y CrewAI como pipeline secuencial de cuatro fases: planner, retriever, analyst y writer.
- `ARCH_03_SUPERVISOR_WORKERS` queda implementada en LangGraph y CrewAI con supervisor explicito, tres workers logicos y sintesis final.
- `psutil` queda como dependencia opcional para monitorizacion de recursos; el codigo no falla si no esta instalado.
- En `ARCH_01_SINGLE_REACT` no se introducen herramientas externas reales en la primera version. Los documentos se pasan como contexto en el prompt comun mediante `benchmark_core.render_single_react_prompt`.
- Se acepta esta variante minima porque valida schemas, runner comun, instrumentacion, persistencia JSON y comparabilidad basica antes de anadir `document_search` u otras herramientas.
- Se incorpora `OpenAIInstrumentedLLM` para pruebas con API real cuando `ExperimentConfig.model_provider == "openai"`, manteniendo el modo local determinista como baseline reproducible.
- En `ARCH_02_SEQUENTIAL_PIPELINE`, `Retriever` se implementa inicialmente como fase LLM instrumentada y no como herramienta determinista. Esto fuerza cuatro llamadas LLM por ejecucion y facilita medir el coste completo de un pipeline secuencial.
- En `ARCH_03_SUPERVISOR_WORKERS`, se introduce una segunda version dinamica: el supervisor decide los workers a ejecutar y los omitidos quedan registrados. El orden de ejecucion sigue siendo canonico para preservar comparabilidad.
- En CrewAI, `ARCH_03_SUPERVISOR_WORKERS` usa `Process.sequential` con un agente supervisor explicito, en lugar de `Process.hierarchical`, para evitar comportamiento de manager implicito dificil de comparar en la primera iteracion.
- La seleccion del supervisor se normaliza mediante un formato parseable (`SELECTED_WORKERS=...`, `SKIPPED_WORKERS=...`). Si un modelo real no respeta el formato, se aplica una politica conservadora para evitar ejecuciones vacias.
