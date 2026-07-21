# ARCH_08_DEBATE_JUDGE - CrewAI

Tres tareas debater asincronas e independientes alimentan mediante `context`
una unica tarea `debate_round`; una tarea `judge` posterior recibe propuestas y
criticas. Se usa proceso secuencial sin manager, delegacion, memoria ni planning.
