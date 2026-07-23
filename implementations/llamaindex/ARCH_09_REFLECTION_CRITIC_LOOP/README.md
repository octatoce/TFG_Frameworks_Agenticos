# ARCH_09_REFLECTION_CRITIC_LOOP - LlamaIndex

Un `Workflow` encadena eventos generator, critic y stop_controller. Si la
decision no termina, emite `ReviserRequestEvent`; reviser devuelve un nuevo
`ReflectionStateEvent` al critic. El evento de parada produce el resultado sin
handoffs ni swarm.
