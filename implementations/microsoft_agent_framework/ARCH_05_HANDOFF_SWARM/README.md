# ARCH_05_HANDOFF_SWARM - Microsoft Agent Framework

Implementacion con `agent-framework-orchestrations` y `HandoffBuilder`.

Los agentes son `Agent` reales del Microsoft Agent Framework. El agente activo devuelve una `HandoffDecision`; si decide transferir, el cliente benchmark la convierte en una herramienta nativa `handoff_to_*` y el workflow emite `handoff_sent`.

No usa supervisor central, group chat ni AutoGen clasico.
