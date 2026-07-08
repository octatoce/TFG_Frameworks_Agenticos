# ARCH_05_HANDOFF_SWARM - LangGraph

Implementacion con `StateGraph`, nodos especialistas y `conditional_edges` directos entre agentes.

No existe supervisor central: cada nodo especialista devuelve una decision `handoff` o `finalize`, y el grafo transfiere el control al siguiente nodo o termina.
