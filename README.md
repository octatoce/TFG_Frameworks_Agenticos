# Evaluación de Frameworks Agénticos Modernos: Comparativa Técnica

Este repositorio contiene el código de mi Trabajo Fin de Grado de Ingeniería Informática. El objetivo del proyecto es comparar varios frameworks agénticos modernos implementando las mismas arquitecturas y ejecutándolas bajo unas condiciones experimentales comunes.

La idea no es decidir qué framework es mejor en términos absolutos, sino estudiar qué diferencias aparecen cuando todos resuelven el mismo problema con el mismo modelo, los mismos datos y una arquitectura equivalente. Me interesan especialmente la latencia, el consumo de tokens, el número de llamadas al LLM, los errores, la trazabilidad y la dificultad de expresar cada patrón en el framework.

## Estado actual

Actualmente se comparan cinco frameworks:

- LangGraph
- Microsoft Agent Framework
- CrewAI
- LlamaIndex
- Pydantic AI junto con pydantic-graph

Se han implementado ocho arquitecturas en cada uno de ellos:

1. `ARCH_01_SINGLE_REACT`: un único agente con un ciclo ReAct sencillo.
2. `ARCH_02_SEQUENTIAL_PIPELINE`: pipeline con fases dependientes ejecutadas en orden.
3. `ARCH_03_ROUTER_SPECIALISTS`: un router selecciona los especialistas necesarios para cada caso.
4. `ARCH_04_SUPERVISOR_WORKERS`: un supervisor central planifica, delega, revisa y limita las iteraciones.
5. `ARCH_05_HANDOFF_SWARM`: varios especialistas se transfieren el control mediante handoffs.
6. `ARCH_06_PARALLEL_FANOUT_FANIN`: cuatro perspectivas independientes trabajan en paralelo y un agregador combina sus resultados.
7. `ARCH_07_MAP_REDUCE_AGENTIC`: los documentos se dividen en batches, se procesan con mappers equivalentes y se sintetizan con un reducer.
8. `ARCH_08_DEBATE_JUDGE`: tres propuestas independientes pasan por una ronda de crítica y un juez toma la decisión final.

Esto da una matriz de 40 implementaciones:

| Framework | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LangGraph | Sí | Sí | Sí | Sí | Sí | Sí | Sí | Sí |
| Microsoft Agent Framework | Sí | Sí | Sí | Sí | Sí | Sí | Sí | Sí |
| CrewAI | Sí | Sí | Sí | Sí | Sí | Sí | Sí | Sí |
| LlamaIndex | Sí | Sí | Sí | Sí | Sí | Sí | Sí | Sí |
| Pydantic AI | Sí | Sí | Sí | Sí | Sí | Sí | Sí | Sí |

## Metodología común

Todas las implementaciones respetan el mismo contrato:

```python
run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
) -> ExperimentResult
```

Para mantener la comparación lo más justa posible se utilizan:

- los mismos modelos y parámetros de generación;
- los mismos datasets e inputs;
- los mismos límites de tiempo e iteraciones;
- las mismas herramientas cuando una arquitectura las necesita;
- schemas comunes para entradas, configuración y resultados;
- métricas y trazas construidas desde `benchmark_core`;
- las primitivas propias de cada framework siempre que existe una alternativa estable.

Los resultados de cada ejecución se guardan en:

```text
results/raw/{framework}/{architecture}/{run_id}.json
```

Las decisiones que pueden afectar a la equivalencia entre frameworks están explicadas en [`docs/decisions.md`](docs/decisions.md). Las especificaciones completas de las arquitecturas están en [`docs/architecture_specs/`](docs/architecture_specs/).

## Medición de tokens

En las ejecuciones con OpenAI, los cinco frameworks utilizan el objeto `usage` devuelto por el proveedor. Los distintos formatos de Responses API, Chat Completions y Pydantic AI se normalizan en los mismos campos:

- tokens de entrada;
- tokens de salida;
- tokens de entrada en caché;
- tokens de razonamiento;
- tokens totales.

Cada llamada queda identificada con `token_counting_method=openai_usage`. Si un SDK no devuelve información de uso, la ejecución falla en lugar de sustituirla silenciosamente por una estimación.

El modelo local determinista sí utiliza un proxy por palabras. Su finalidad es comprobar contratos, grafos, trazas y persistencia sin consumir una API externa. Estos resultados no deben mezclarse con los resultados OpenAI al comparar tokens.

## Estructura del repositorio

```text
benchmark_core/     Contratos, schemas, métricas, trazas y escritura de resultados
configs/            Configuración común de los experimentos
docs/               Metodología, decisiones y especificaciones
implementations/    Implementaciones separadas por framework y arquitectura
scripts/            Smokes locales y smokes con OpenAI
tests/              Tests contractuales, estructurales y de métricas
datasets/           Datasets raw y procesados
results/            Resultados raw, resultados procesados y figuras
notebooks/          Análisis y visualizaciones
```

## Instalación

El proyecto requiere Python 3.11 o superior.

En Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,all-frameworks]"
```

En Linux o macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,all-frameworks]"
```

También se puede instalar solo el framework que se quiera probar:

```powershell
pip install -e ".[langgraph]"
pip install -e ".[microsoft_agent_framework]"
pip install -e ".[crewai]"
pip install -e ".[llamaindex]"
pip install -e ".[pydantic_ai]"
```

## Tests

La suite completa se ejecuta con:

```powershell
python -m pytest
```

Los tests comprueban el contrato común, la estructura de las arquitecturas, el orden de los pasos, la instrumentación y el guardado de resultados. Las pruebas normales utilizan el modelo determinista y no necesitan una API externa.

## Smokes locales

Actualmente hay smokes locales para las tres primeras arquitecturas:

```powershell
python scripts\run_arch01_smoke.py
python scripts\run_arch02_smoke.py
python scripts\run_arch03_smoke.py
```

Cada script ejecuta el mismo caso en los cinco frameworks y guarda los JSON bajo `results/raw/`.

## Smokes con OpenAI

Las pruebas con OpenAI requieren un archivo `.env` con, al menos:

```dotenv
OPENAI_API_KEY=...
MODEL_NAME=...
```

Los smokes disponibles son:

```powershell
python scripts\run_arch01_openai_smoke.py
python scripts\run_arch06_openai_smoke.py
python scripts\run_arch07_openai_smoke.py
python scripts\run_arch08_openai_smoke.py
```

- ARCH_01 comprueba la ruta OpenAI básica de los cinco adaptadores.
- ARCH_06 comprueba el fan-out/fan-in y el solapamiento temporal de las ramas.
- ARCH_07 utiliza siete documentos sintéticos, tres batches y un reducer final.
- ARCH_08 comprueba las propuestas independientes, la crítica cruzada y la decisión del juez.

Estos scripts también verifican que todas las llamadas tengan tokens reales `openai_usage` y que el resultado se haya guardado correctamente.

## Situación del proyecto

La parte de implementación de las ocho arquitecturas está completa en los cinco frameworks. Los tests contractuales y los smokes básicos con OpenAI están funcionando.

La siguiente fase es preparar los experimentos sobre datasets públicos versionados, ejecutar suficientes repeticiones y analizar los resultados sin mezclar ejecuciones locales, smokes y benchmarks definitivos.
