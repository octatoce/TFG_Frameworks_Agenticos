# CrewAI

Implementaciones del benchmark usando primitivas reales de CrewAI:

- `Agent`
- `Task`
- `Crew`
- `Process.sequential`

Cada arquitectura expone la misma funcion publica:

```python
run_architecture(input_data, config)
```

## Codigo compartido

La logica comun esta en `utils_crewai.py`.

Ese archivo prepara el entorno de CrewAI, crea el LLM instrumentado, construye agentes/tasks/crews con opciones comunes y envuelve la ejecucion con el decorador `crewai_architecture_runner`.

Asi, cada `run.py` solo contiene lo especifico de su arquitectura.

## Arquitecturas

- `ARCH_01_SINGLE_REACT`: un agente y una task.
- `ARCH_02_SEQUENTIAL_PIPELINE`: cuatro fases secuenciales.
- `ARCH_03_SUPERVISOR_WORKERS`: supervisor, workers seleccionados y sintesis final.

## Notas

CrewAI puede crear estado local y activar telemetria por defecto. La preparacion comun redirige esos efectos a `.crewai_data` dentro del repositorio y desactiva tracking para que las ejecuciones sean reproducibles.
