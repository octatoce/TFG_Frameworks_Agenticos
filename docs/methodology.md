# Metodologia

Este documento describe la metodologia de evaluacion del TFG.

## Principios

- Comparar frameworks mediante prototipos funcionalmente equivalentes.
- Usar el mismo contrato de entrada y salida para todas las arquitecturas.
- Recoger metricas mediante `benchmark_core`.
- Mantener constantes los datasets, prompts y configuraciones cuando sea posible.
- Documentar cualquier modificacion necesaria en `docs/decisions.md`.

## Metricas iniciales

- Latencia total.
- Numero de pasos.
- Numero de llamadas al LLM.
- Tokens de entrada.
- Tokens de salida.
- Coste estimado.
- Numero de errores.
- Uso de CPU y RAM cuando el entorno lo permita.
