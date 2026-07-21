# ARCH_08_DEBATE_JUDGE - LangGraph

Implementacion con `StateGraph`: tres nodos debater parten desde `START`, una
arista de origen multiple actua como barrera hacia `debate_round`, y `judge`
se ejecuta solo despues de esa ronda. No hay routing, handoffs ni bucles.
