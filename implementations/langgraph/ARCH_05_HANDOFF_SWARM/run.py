"""LangGraph implementation for ARCH_05_HANDOFF_SWARM."""

from __future__ import annotations

from typing import TypedDict

from benchmark_core.handoff_swarm import (
    HANDOFF_AGENTS,
    build_fallback_answer,
    choose_initial_handoff_agent,
    document_ids,
    extract_final_answer,
    get_handoff_limits,
    is_valid_handoff_decision,
    parse_handoff_decision,
    render_handoff_swarm_prompt,
)
from benchmark_core.schemas import AgentStep, ExperimentConfig, ExperimentInput
from benchmark_core.tracing import utc_now
from implementations.langgraph.utils_langgraph import (
    LangGraphRunContext,
    LangGraphRunOutput,
    complete_llm_step,
    invoke_with_resource_monitor,
    langgraph_architecture_runner,
    next_step_id,
)


class SwarmState(TypedDict, total=False):
    active_agent: str
    initial_agent: str
    active_agent_history: list[str]
    handoff_history: list[dict[str, object]]
    partial_results: list[dict[str, object]]
    context_transferred: str
    number_of_handoffs: int
    number_of_agent_invocations: int
    repeated_agent_visits: dict[str, int]
    cycle_detected: bool
    fallback_used: bool
    finalizing_agent: str | None
    stop_reason: str | None
    warnings: list[str]
    last_decision: dict[str, object] | None
    final_answer: str
    max_handoffs: int
    max_agent_invocations: int
    max_consecutive_visits_per_agent: int
    steps: list[object]
    llm_calls: list[object]


@langgraph_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: LangGraphRunContext,
) -> LangGraphRunOutput:
    """Execute decentralized handoffs as direct LangGraph node transitions."""

    from langgraph.graph import END, StateGraph

    limits = get_handoff_limits(config)

    def append_step(
        state: SwarmState,
        *,
        agent_name: str,
        prompt: str,
        decision: dict[str, object],
        call_record: object,
    ) -> SwarmState:
        step_started_at = utc_now()
        step = AgentStep(
            step_id=len(state["steps"]) + 1,
            name=agent_name,
            step_type="handoff_agent_llm_call",
            actor=f"langgraph.{agent_name}_node",
            input_data={"prompt": prompt, "active_agent": agent_name},
            output_data={"decision": decision},
            llm_call_ids=[call_record.metrics.call_id],
            started_at=step_started_at,
            finished_at=utc_now(),
            metadata={
                "architecture": "ARCH_05_HANDOFF_SWARM",
                "native_primitive": "StateGraph.conditional_edges",
            },
        )
        return {
            **state,
            "steps": [*state["steps"], step],
            "llm_calls": [*state["llm_calls"], call_record.metrics],
        }

    def run_agent(state: SwarmState, agent_name: str) -> SwarmState:
        if state["number_of_agent_invocations"] >= state["max_agent_invocations"]:
            return {
                **state,
                "stop_reason": "max_agent_invocations_reached",
                "finalizing_agent": agent_name,
                "final_answer": build_fallback_answer(input_data, state["partial_results"]),
            }

        visits = dict(state["repeated_agent_visits"])
        visits[agent_name] = visits.get(agent_name, 0) + 1
        if visits[agent_name] > state["max_consecutive_visits_per_agent"]:
            return {
                **state,
                "repeated_agent_visits": visits,
                "cycle_detected": True,
                "fallback_used": True,
                "stop_reason": "max_consecutive_visits_per_agent_reached",
                "finalizing_agent": agent_name,
                "final_answer": build_fallback_answer(input_data, state["partial_results"]),
                "warnings": [*state["warnings"], f"Visit limit reached for {agent_name}."],
            }

        prompt_state = {
            **state,
            "repeated_agent_visits": visits,
        }
        prompt = render_handoff_swarm_prompt(input_data, agent_name, prompt_state)
        step_id = next_step_id(state)
        call_record = complete_llm_step(
            llm=context.llm,
            input_data=input_data,
            config=config,
            prompt=prompt,
            step_id=step_id,
        )
        decision = parse_handoff_decision(call_record.response.strip())
        updated = append_step(
            {
                **state,
                "active_agent_history": [*state["active_agent_history"], agent_name],
                "number_of_agent_invocations": state["number_of_agent_invocations"] + 1,
                "repeated_agent_visits": visits,
                "last_decision": decision,
                "partial_results": [
                    *state["partial_results"],
                    {"agent": agent_name, "decision": decision},
                ],
            },
            agent_name=agent_name,
            prompt=prompt,
            decision=decision,
            call_record=call_record,
        )

        if not is_valid_handoff_decision(decision):
            warnings = [*updated["warnings"], f"Invalid handoff decision from {agent_name}."]
            if agent_name != "synthesis_specialist" and updated["number_of_handoffs"] < updated["max_handoffs"]:
                handoff = {
                    "sequence_number": updated["number_of_handoffs"] + 1,
                    "source_agent": agent_name,
                    "target_agent": "synthesis_specialist",
                    "reason": "Fallback after invalid decision.",
                    "task": "Finalize from available context.",
                    "context_summary": "Fallback context after invalid decision.",
                    "timestamp": utc_now().isoformat(),
                }
                return {
                    **updated,
                    "active_agent": "synthesis_specialist",
                    "context_transferred": "Fallback context after invalid decision.",
                    "handoff_history": [*updated["handoff_history"], handoff],
                    "number_of_handoffs": updated["number_of_handoffs"] + 1,
                    "fallback_used": True,
                    "warnings": warnings,
                }
            return {
                **updated,
                "fallback_used": True,
                "warnings": warnings,
                "stop_reason": "invalid_decision_fallback_finalized",
                "finalizing_agent": agent_name,
                "final_answer": build_fallback_answer(input_data, updated["partial_results"]),
            }

        if decision["action"] == "finalize":
            return {
                **updated,
                "stop_reason": "agent_finalized",
                "finalizing_agent": agent_name,
                "final_answer": extract_final_answer(str(decision["final_output"])),
            }

        target_agent = str(decision["target_agent"])
        cycle_detected = updated["cycle_detected"] or (
            len(updated["active_agent_history"]) >= 2
            and target_agent == updated["active_agent_history"][-2]
        )
        if updated["number_of_handoffs"] >= updated["max_handoffs"]:
            return {
                **updated,
                "cycle_detected": cycle_detected,
                "stop_reason": "max_handoffs_reached",
                "finalizing_agent": agent_name,
                "final_answer": build_fallback_answer(input_data, updated["partial_results"]),
            }

        handoff = {
            "sequence_number": updated["number_of_handoffs"] + 1,
            "source_agent": agent_name,
            "target_agent": target_agent,
            "reason": decision["reason"],
            "task": decision["task"],
            "context_summary": decision["context_summary"],
            "timestamp": utc_now().isoformat(),
        }
        return {
            **updated,
            "active_agent": target_agent,
            "context_transferred": str(decision["context_summary"]),
            "handoff_history": [*updated["handoff_history"], handoff],
            "number_of_handoffs": updated["number_of_handoffs"] + 1,
            "cycle_detected": cycle_detected,
        }

    def route_next(state: SwarmState) -> str:
        if state.get("stop_reason"):
            return END
        return state["active_agent"]

    graph = StateGraph(SwarmState)
    for agent in HANDOFF_AGENTS:
        graph.add_node(agent, lambda state, agent=agent: run_agent(state, agent))
    graph.set_entry_point(choose_initial_handoff_agent(input_data))
    for agent in HANDOFF_AGENTS:
        graph.add_conditional_edges(agent, route_next, {**{target: target for target in HANDOFF_AGENTS}, END: END})

    initial_agent = choose_initial_handoff_agent(input_data)
    initial_state: SwarmState = {
        "active_agent": initial_agent,
        "initial_agent": initial_agent,
        "active_agent_history": [],
        "handoff_history": [],
        "partial_results": [],
        "context_transferred": "",
        "number_of_handoffs": 0,
        "number_of_agent_invocations": 0,
        "repeated_agent_visits": {},
        "cycle_detected": False,
        "fallback_used": False,
        "finalizing_agent": None,
        "stop_reason": None,
        "warnings": [],
        "last_decision": None,
        "final_answer": "",
        "max_handoffs": limits["max_handoffs"],
        "max_agent_invocations": limits["max_agent_invocations"],
        "max_consecutive_visits_per_agent": limits["max_consecutive_visits_per_agent"],
        "steps": [],
        "llm_calls": [],
    }
    state, resource_usage = invoke_with_resource_monitor(graph.compile(), initial_state)

    final_answer = state["final_answer"] or build_fallback_answer(input_data, state["partial_results"])
    last_decision = state["last_decision"] or {}
    structured_output = {
        "answer": final_answer,
        "decision": last_decision,
        "confidence": last_decision.get("confidence", 0.0),
        "evidence": last_decision.get("evidence", "none"),
        "limitations": last_decision.get("limitations", "none"),
        "initial_agent": state["initial_agent"],
        "active_agent_history": state["active_agent_history"],
        "handoff_history": state["handoff_history"],
        "number_of_handoffs": state["number_of_handoffs"],
        "max_handoffs": state["max_handoffs"],
        "number_of_agent_invocations": state["number_of_agent_invocations"],
        "max_agent_invocations": state["max_agent_invocations"],
        "unique_agents_executed": sorted(set(state["active_agent_history"]), key=HANDOFF_AGENTS.index),
        "finalizing_agent": state["finalizing_agent"],
        "repeated_agent_visits": state["repeated_agent_visits"],
        "cycle_detected": state["cycle_detected"],
        "fallback_used": state["fallback_used"],
        "stop_reason": state["stop_reason"],
        "framework_native_primitives": ["StateGraph", "conditional_edges"],
        "native_automatic_behaviors": [],
        "parallelism_used": False,
        "warnings": state["warnings"],
        "document_ids": document_ids(input_data),
        "framework_execution": "langgraph_handoff_swarm_state_graph",
    }
    return LangGraphRunOutput(
        final_answer=final_answer,
        structured_output=structured_output,
        steps=state["steps"],
        llm_calls=state["llm_calls"],
        resource_usage=resource_usage,
    )
