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

## Alcance de la iteracion 1

Frameworks iniciales:

- LangGraph
- CrewAI

Arquitecturas iniciales:

- `ARCH_01_SINGLE_REACT`
- `ARCH_02_SEQUENTIAL_PIPELINE`
- `ARCH_03_SUPERVISOR_WORKERS`

Esta iteracion implementa prototipos minimos funcionales para cada combinacion framework-arquitectura. Las ejecuciones pueden usar un LLM local determinista para validar comparabilidad, schemas, metricas y persistencia sin depender de servicios externos. Tambien queda preparado un smoke test opcional contra OpenAI mediante variables locales de entorno.

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

## Ejecutar tests

```bash
python -m pytest
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

No se deben introducir prompts, datasets, configuraciones u optimizaciones distintas por framework salvo que la desviacion este documentada en `docs/decisions.md`.
