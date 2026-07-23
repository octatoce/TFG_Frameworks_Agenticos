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

Se han implementado diez arquitecturas en cada uno de ellos:

1. `ARCH_01_SINGLE_REACT`: un único agente con un ciclo ReAct sencillo.
2. `ARCH_02_SEQUENTIAL_PIPELINE`: pipeline con fases dependientes ejecutadas en orden.
3. `ARCH_03_ROUTER_SPECIALISTS`: un router selecciona los especialistas necesarios para cada caso.
4. `ARCH_04_SUPERVISOR_WORKERS`: un supervisor central planifica, delega, revisa y limita las iteraciones.
5. `ARCH_05_HANDOFF_SWARM`: varios especialistas se transfieren el control mediante handoffs.
6. `ARCH_06_PARALLEL_FANOUT_FANIN`: cuatro perspectivas independientes trabajan en paralelo y un agregador combina sus resultados.
7. `ARCH_07_MAP_REDUCE_AGENTIC`: los documentos se dividen en batches, se procesan con mappers equivalentes y se sintetizan con un reducer.
8. `ARCH_08_DEBATE_JUDGE`: tres propuestas independientes pasan por una ronda de crítica y un juez toma la decisión final.
9. `ARCH_09_REFLECTION_CRITIC_LOOP`: una respuesta única se somete a crítica y revisión en un ciclo explícito y acotado.
10. `ARCH_10_CHECKPOINT_MEMORY_RECOVERY`: un workflow guarda estado, simula un fallo y continúa desde un checkpoint verificado.

Esto da una matriz de 50 implementaciones:

| Framework | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LangGraph | Sí | Sí | Sí | Sí | Sí | Sí | Sí | Sí | Sí | Sí |
| Microsoft Agent Framework | Sí | Sí | Sí | Sí | Sí | Sí | Sí | Sí | Sí | Sí |
| CrewAI | Sí | Sí | Sí | Sí | Sí | Sí | Sí | Sí | Sí | Sí |
| LlamaIndex | Sí | Sí | Sí | Sí | Sí | Sí | Sí | Sí | Sí | Sí |
| Pydantic AI | Sí | Sí | Sí | Sí | Sí | Sí | Sí | Sí | Sí | Sí |

La implementación base de `ARCH_10_CHECKPOINT_MEMORY_RECOVERY` está terminada y validada en los cinco frameworks. Actualmente cubre el modo de recuperación ante fallo controlado: se crea un checkpoint después del análisis inicial, se interrumpe la ejecución, se recupera el estado y se continúa hasta producir el resultado final.

Como ampliación de esta arquitectura queda pendiente un segundo modo Human-in-the-Loop. La idea es que el workflow llegue a una tarea que requiera validación humana, guarde el estado y quede pausado sin exigir una respuesta en tiempo real. Cuando llegue la decisión de la persona, incluso en otra ejecución o sesión, el workflow deberá recuperar el checkpoint y continuar desde ese punto. Esta ampliación no sustituirá el modo de fallo actual; permitirá comparar por separado recuperación técnica y pausa asíncrona por intervención humana.

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

La última validación completa del estado actual terminó con `37 passed`. También se ejecutó el smoke real de ARCH_10 con OpenAI en los cinco frameworks: las cinco ejecuciones crearon checkpoint, simularon el fallo, recuperaron el estado y finalizaron correctamente.

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
python scripts\run_arch09_openai_smoke.py
python scripts\run_arch10_openai_smoke.py
```

- ARCH_01 comprueba la ruta OpenAI básica de los cinco adaptadores.
- ARCH_06 comprueba el fan-out/fan-in y el solapamiento temporal de las ramas.
- ARCH_07 utiliza siete documentos sintéticos, tres batches y un reducer final.
- ARCH_08 comprueba las propuestas independientes, la crítica cruzada y la decisión del juez.
- ARCH_09 comprueba el criterio de parada, las versiones intermedias, los tokens reales y el límite del ciclo de reflexión.
- ARCH_10 inyecta un fallo controlado, recupera el estado y compara backend, latencias, llamadas y tokens. En LangGraph cierra y vuelve a abrir una base SQLite antes de reanudar para verificar persistencia durable local.

Estos scripts también verifican que todas las llamadas tengan tokens reales `openai_usage` y que el resultado se haya guardado correctamente.

## Situación del proyecto

La implementación base de las diez arquitecturas está completa en los cinco frameworks, lo que da las 50 variantes de la matriz comparativa. Los tests contractuales están automatizados y los smokes OpenAI existentes cubren las arquitecturas indicadas en la sección anterior.

ARCH_10 sigue abierta únicamente como línea de ampliación: el modo de recuperación ante fallo ya está implementado y probado, mientras que el modo Human-in-the-Loop asíncrono queda como tarea pendiente. Para esa extensión habrá que definir una entrada de aprobación común, persistir la pausa sin bloquear el proceso y comprobar que cada framework puede reanudar la ejecución cuando la respuesta humana llegue más tarde.

La siguiente fase general es preparar los experimentos sobre datasets públicos versionados, ejecutar suficientes repeticiones y analizar los resultados sin mezclar ejecuciones locales, smokes y benchmarks definitivos.
