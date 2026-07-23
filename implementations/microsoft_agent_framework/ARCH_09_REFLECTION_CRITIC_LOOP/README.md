# ARCH_09_REFLECTION_CRITIC_LOOP - Microsoft Agent Framework

El workflow usa executors diferenciados para generator, critic,
stop_controller y reviser. `add_switch_case_edge_group` termina o dirige a
reviser; el edge `reviser -> critic` materializa el ciclo limitado.
