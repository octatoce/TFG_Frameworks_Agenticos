# ARCH_09_REFLECTION_CRITIC_LOOP - LangGraph

`StateGraph` representa el ciclo `generator -> critic -> stop_controller ->
reviser -> critic`. Una arista condicional desde `stop_controller` termina en
`END` o permite otra revision. El limite de recursion deriva del maximo de
iteraciones configurado y no se usa persistencia.
