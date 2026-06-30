"""CrewAI implementation for ARCH_04_SUPERVISOR_WORKERS."""

from __future__ import annotations

from benchmark_core.llm_wrapper import SUPERVISOR_WORKERS
from benchmark_core.resource_monitor import ResourceMonitor
from benchmark_core.schemas import ExperimentConfig, ExperimentInput
from benchmark_core.schemas import AgentStep
from benchmark_core.tracing import utc_now
from implementations.crewai.utils_crewai import (
    CrewAIRunContext,
    CrewAIRunOutput,
    create_agent,
    create_hierarchical_crew,
    create_task,
    crewai_architecture_runner,
)
from implementations.supervisor_workers_common import (
    document_ids,
    extract_final_answer,
    get_max_supervisor_iterations,
)


WORKER_DEFINITIONS = {
    "data_worker": {
        "role": "DataWorker",
        "goal": "Extract evidence and relevant document fragments.",
        "backstory": "A controlled worker in a centralized supervisor benchmark.",
    },
    "reasoning_worker": {
        "role": "ReasoningWorker",
        "goal": "Analyze evidence and build the technical explanation.",
        "backstory": "A controlled worker in a centralized supervisor benchmark.",
    },
    "validation_worker": {
        "role": "ValidationWorker",
        "goal": "Find errors, contradictions, risks, and missing evidence.",
        "backstory": "A controlled worker in a centralized supervisor benchmark.",
    },
    "synthesis_worker": {
        "role": "SynthesisWorker",
        "goal": "Build a final clear answer from approved material.",
        "backstory": "A controlled worker in a centralized supervisor benchmark.",
    },
}


@crewai_architecture_runner
def run_architecture(
    input_data: ExperimentInput,
    config: ExperimentConfig,
    context: CrewAIRunContext,
) -> CrewAIRunOutput:
    """Execute CrewAI through its native hierarchical process."""

    supervisor = create_agent(
        role="Supervisor",
        goal="Plan, coordinate, review, request bounded revisions, and approve the final answer.",
        backstory="A centralized supervisor for ARCH_04_SUPERVISOR_WORKERS.",
        crewai_llm=context.crewai_llm,
        config=config,
        allow_delegation=True,
    )
    workers = [
        create_agent(
            **definition,
            crewai_llm=context.crewai_llm,
            config=config,
        )
        for definition in WORKER_DEFINITIONS.values()
    ]
    max_supervisor_iterations = get_max_supervisor_iterations(config)
    document_blocks = "\n".join(
        f"[{document.document_id}] {document.content}" for document in input_data.documents
    ) or "No documents provided."

    task = create_task(
        description=(
            "You are executing ARCH_04_SUPERVISOR_WORKERS using CrewAI's native hierarchical process.\n"
            "The manager/supervisor must delegate only the workers needed, validate outcomes before proceeding, "
            "and stop when quality is sufficient or the iteration limit is reached.\n\n"
            f"Maximum supervisor worker delegations: {max_supervisor_iterations}\n"
            "Available workers: DataWorker, ReasoningWorker, ValidationWorker, SynthesisWorker.\n"
            "Use DataWorker only when document evidence is useful. Use ValidationWorker when the task needs "
            "risk, contradiction, confidence, comparison, or error checking. Use SynthesisWorker only when a "
            "final answer draft is needed.\n\n"
            f"Task type: {input_data.task_type}\n"
            f"Question: {input_data.query}\n\n"
            f"Documents:\n{document_blocks}\n"
        ),
        expected_output=(
            "A final answer approved by the hierarchical manager, based on delegated worker outputs. "
            "The manager should not use persistent memory, external observability, checkpointing, or parallel fan-out."
        ),
        agent=None,
        config=config,
    )

    with ResourceMonitor() as monitor:
        crew = create_hierarchical_crew(
            agents=workers,
            tasks=[task],
            manager_agent=supervisor,
        )
        crew_output = crew.kickoff()
        resource_usage = monitor.usage

    role_to_worker = {
        "DataWorker": "data_worker",
        "ReasoningWorker": "reasoning_worker",
        "ValidationWorker": "validation_worker",
        "SynthesisWorker": "synthesis_worker",
    }
    worker_outputs = []
    workers_executed = []
    steps = []
    supervisor_seen = 0
    records = list(context.crewai_llm.call_records)
    def record_role(record):
        role = record.metrics.metadata.get("crewai_from_agent_role")
        if role:
            return role
        for candidate_role in ["Supervisor", "DataWorker", "ReasoningWorker", "ValidationWorker", "SynthesisWorker"]:
            if f"You are {candidate_role}." in record.prompt:
                return candidate_role
        return None

    last_supervisor_index = max(
        (
            index
            for index, record in enumerate(records)
            if record_role(record) == "Supervisor"
        ),
        default=-1,
    )

    for index, record in enumerate(records):
        role = record_role(record)
        step_started_at = utc_now()
        if role in role_to_worker:
            worker_name = role_to_worker[role]
            workers_executed.append(worker_name)
            worker_output = {
                "worker_name": worker_name,
                "task": f"Native CrewAI hierarchical delegation to {role}.",
                "output": record.response.strip(),
                "revision": False,
                "iteration": len(workers_executed) - 1,
            }
            worker_outputs.append(worker_output)
            steps.append(
                AgentStep(
                    step_id=len(steps) + 1,
                    name=worker_name,
                    step_type="worker_llm_call",
                    actor=f"crewai.{role}",
                    input_data={"prompt": record.prompt},
                    output_data={"worker_output": worker_output},
                    llm_call_ids=[record.metrics.call_id],
                    started_at=step_started_at,
                    finished_at=utc_now(),
                    metadata={
                        "architecture": "ARCH_04_SUPERVISOR_WORKERS",
                        "crew_process": "hierarchical",
                        "worker_role": worker_name,
                        "native_hierarchical": True,
                    },
                )
            )
        elif role == "Supervisor":
            supervisor_seen += 1
            is_final = index == last_supervisor_index
            name = "supervisor_finalize" if is_final else "supervisor_plan" if supervisor_seen == 1 else "supervisor_decision"
            step_type = (
                "supervisor_finalize_llm_call"
                if is_final
                else "supervisor_plan_llm_call"
                if supervisor_seen == 1
                else "supervisor_review_llm_call"
            )
            steps.append(
                AgentStep(
                    step_id=len(steps) + 1,
                    name=name,
                    step_type=step_type,
                    actor="crewai.native_hierarchical_manager",
                    input_data={"prompt": record.prompt},
                    output_data={
                        "manager_output": record.response.strip(),
                        "workers_executed_so_far": list(workers_executed),
                    },
                    llm_call_ids=[record.metrics.call_id],
                    started_at=step_started_at,
                    finished_at=utc_now(),
                    metadata={
                        "architecture": "ARCH_04_SUPERVISOR_WORKERS",
                        "crew_process": "hierarchical",
                        "native_hierarchical": True,
                    },
                )
            )

    workers_used = sorted(set(workers_executed), key=SUPERVISOR_WORKERS.index)
    workers_not_used = [worker for worker in SUPERVISOR_WORKERS if worker not in workers_used]
    final_answer = extract_final_answer(str(crew_output))
    stop_reason = (
        "max_supervisor_iterations_reached"
        if len(workers_executed) >= max_supervisor_iterations
        else "native_hierarchical_completed"
    )
    task_assignments = {
        "data_worker": "Extract evidence and source references.",
        "reasoning_worker": "Analyze evidence and answer requirements.",
        "validation_worker": "Check contradictions, missing evidence, and risk.",
        "synthesis_worker": "Build the final structured answer.",
    }
    supervisor_plan = {
        "workers_to_run": workers_used,
        "task_assignments": {
            worker: task_assignments[worker]
            for worker in workers_used
        },
        "expected_outputs": {
            worker: f"Native CrewAI hierarchical output for {worker}."
            for worker in workers_used
        },
        "quality_criteria": ["evidence", "consistency", "completeness", "clarity"],
        "raw_plan": "Derived from CrewAI native hierarchical manager delegations.",
    }
    structured_output = {
        "answer": final_answer,
        "mode": f"{config.model_provider}_supervisor_workers",
        "supervisor_plan": supervisor_plan,
        "worker_outputs": worker_outputs,
        "workers_executed": workers_executed,
        "number_of_workers_executed": len(workers_executed),
        "workers_used": workers_used,
        "workers_not_used": workers_not_used,
        "supervisor_iterations": min(len(workers_executed), max_supervisor_iterations),
        "max_supervisor_iterations": max_supervisor_iterations,
        "revisions_requested": 0,
        "accepted_worker_outputs": workers_used,
        "rejected_worker_outputs": [],
        "stop_reason": stop_reason,
        "warnings": [],
        "document_ids": document_ids(input_data),
        "framework_execution": "crewai_native_hierarchical_process",
    }

    return CrewAIRunOutput(
        final_answer=final_answer,
        structured_output=structured_output,
        steps=steps,
        resource_usage=resource_usage,
    )
