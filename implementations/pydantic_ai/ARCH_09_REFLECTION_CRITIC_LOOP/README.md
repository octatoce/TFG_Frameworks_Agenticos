# ARCH_09_REFLECTION_CRITIC_LOOP - Pydantic AI + pydantic-graph

`GraphBuilder` usa pasos tipados para generator, critic, stop_controller y
reviser. Una `Decision[StopDecision]` dirige al finalizador o devuelve reviser
al critic. Versiones, criticas, decisiones y salida final son modelos Pydantic.
