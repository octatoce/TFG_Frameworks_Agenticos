# ARCH_09_REFLECTION_CRITIC_LOOP - CrewAI

Tres agentes y tareas CrewAI de proposito unico se ejecutan bajo un `for`
externo acotado. Cada critic y reviser se materializa como una tarea/crew
medible; el stop_controller es determinista. No se usa manager, Flow router,
delegacion, memoria, planning ni reintentos internos no trazados.
