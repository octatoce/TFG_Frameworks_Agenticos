# Evaluacion de Frameworks Agenticos Modernos: Comparativa Tecnica

Este repositorio contiene el entorno de experimentacion para un Trabajo Fin de Grado de Ingenieria Informatica centrado en comparar tecnicamente frameworks modernos de agentes LLM mediante prototipos reales.

El objetivo es evaluar de forma comparable aspectos como latencia, consumo de recursos, numero de llamadas al LLM, tokens, coste estimado, errores, mantenibilidad y facilidad de implementacion.

## Estructura del repositorio

- `benchmark_core/`: paquete comun con schemas, metricas, tracing, monitorizacion de recursos, escritura de resultados y contrato de ejecucion.
- `implementations/`: implementaciones por framework. Cada framework contiene una subcarpeta por arquitectura.
- `configs/`: configuraciones compartidas de modelos, experimentos y datasets.
- `docs/`: metodologia, decisiones y especificaciones academicas de arquitecturas.
- `datasets/`: datos raw y procesados usados por los experimentos.
- `results/`: resultados raw en JSON, resultados procesados y figuras generadas.
- `notebooks/`: analisis exploratorio y visualizaciones.
- `tests/`: tests minimos del contrato comun y de la estructura.

## Desarrollado hasta ahora

Frameworks que ya estan incluidos:

- LangGraph
- CrewAI
- Microsoft Agent Framework
- LlamaIndex
- Pydantic AI

Arquitecturas que ya estan implementadas:

- `ARCH_01_SINGLE_REACT`
- `ARCH_02_SEQUENTIAL_PIPELINE`
- `ARCH_03_ROUTER_SPECIALISTS`
- `ARCH_04_SUPERVISOR_WORKERS`
- `ARCH_05_HANDOFF_SWARM`
- `ARCH_06_PARALLEL_FANOUT_FANIN`
- `ARCH_07_MAP_REDUCE_AGENTIC`
- `ARCH_08_DEBATE_JUDGE`

El estado actual del repositorio deja prototipos funcionales para una matriz de 5 frameworks x 8 arquitecturas. Las ejecuciones pueden usar un LLM local determinista para validar comparabilidad, schemas, metricas y persistencia sin depender de servicios externos, y mantienen rutas de proveedor real mediante las primitivas propias de cada framework.

| Framework | ARCH_01 | ARCH_02 | ARCH_03 | ARCH_04 | ARCH_05 | ARCH_06 | ARCH_07 | ARCH_08 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LangGraph | Implementada | Implementada | Implementada | Implementada | Implementada | Implementada | Implementada | Implementada |
| CrewAI | Implementada | Implementada | Implementada | Implementada | Implementada | Implementada | Implementada | Implementada |
| Microsoft Agent Framework | Implementada | Implementada | Implementada | Implementada | Implementada | Implementada | Implementada | Implementada |
| LlamaIndex | Implementada | Implementada | Implementada | Implementada | Implementada | Implementada | Implementada | Implementada |
| Pydantic AI | Implementada | Implementada | Implementada | Implementada | Implementada | Implementada | Implementada | Implementada |

Total actual:

```text
5 frameworks x 8 arquitecturas = 40 implementaciones
```

## Instalacion base

Se requiere Python 3.11 o superior.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

En Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Para instalar dependencias especificas de un framework:

```powershell
pip install -e ".[langgraph]"
pip install -e ".[crewai]"
pip install -e ".[microsoft_agent_framework]"
pip install -e ".[llamaindex]"
pip install -e ".[pydantic_ai]"
```

Para instalar todos los frameworks declarados:

```powershell
pip install -e ".[dev,all-frameworks]"
```

## Ejecutar tests

```bash
python -m pytest
```

## Ejecutar smoke tests

```powershell
python scripts\run_arch01_smoke.py
python scripts\run_arch02_smoke.py
python scripts\run_arch03_smoke.py
```

Cada script ejecuta la arquitectura correspondiente en los cinco frameworks y guarda resultados raw bajo `results/raw/`.

Para validar llamadas reales a OpenAI en `ARCH_01_SINGLE_REACT` para los cinco frameworks:

```powershell
python scripts\run_arch01_openai_smoke.py
```

Este script requiere `.env` con `OPENAI_API_KEY` y usa `MODEL_NAME` si esta definido.

Para validar el fan-out/fan-in real, las cinco llamadas por framework y las
metricas por rama de `ARCH_06_PARALLEL_FANOUT_FANIN`:

```powershell
python scripts\run_arch06_openai_smoke.py
```

Ademas de guardar los resultados raw, el script comprueba que los cinco
adaptadores usan exclusivamente el conteo real del proveedor (`openai_usage`).
El proxy por palabras se reserva para ejecuciones locales deterministas y no se
debe mezclar con resultados OpenAI.

Para ejecutar una prueba OpenAI basica de `ARCH_07_MAP_REDUCE_AGENTIC` con siete
documentos sinteticos, tres batches y cuatro llamadas por framework:

```powershell
python scripts\run_arch07_openai_smoke.py
```

Para validar las tres propuestas, la ronda de critica, el juez y las cinco
llamadas OpenAI por framework de `ARCH_08_DEBATE_JUDGE`:

```powershell
python scripts\run_arch08_openai_smoke.py
```

## Organizacion de experimentos

Todas las arquitecturas deben exponer una funcion:

```python
run_architecture(input_data: ExperimentInput, config: ExperimentConfig) -> ExperimentResult
```

Todas las entradas y salidas deben usar los schemas de `benchmark_core`. Las metricas se recogen mediante utilidades comunes y los resultados raw se guardan siempre en:

```text
results/raw/{framework}/{architecture}/{run_id}.json
```

Si se introducen prompts, datasets, configuraciones u optimizaciones diferentes por framework, tiene que quedar justificado y documentado en `docs/decisions.md`, para que la comparacion siga siendo trazable y lo mas justa posible.
