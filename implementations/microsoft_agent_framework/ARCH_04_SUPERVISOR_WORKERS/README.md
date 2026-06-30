# ARCH_04_SUPERVISOR_WORKERS - Microsoft Agent Framework

Implementacion como workflow centralizado: un supervisor crea el plan, decide el siguiente worker, revisa salidas, puede pedir revision y finaliza con limite de iteraciones.

No usa AutoGen clasico ni conversacion libre entre agentes. La capa de orquestacion mantiene el estado y registra plan, decisiones, workers ejecutados, revisiones y razon de parada.
