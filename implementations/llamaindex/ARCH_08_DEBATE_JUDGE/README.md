# ARCH_08_DEBATE_JUDGE - LlamaIndex

Un `Workflow` emite tres eventos debater, los sincroniza con
`Context.collect_events`, ejecuta `debate_round` y despues `judge`. No usa
`AgentWorkflow` con handoffs ni swarm.
