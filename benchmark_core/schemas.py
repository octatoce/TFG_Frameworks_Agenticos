from datetime import datetime
from enum import StrEnum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


class DocumentInput(StrictModel):
    document_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExperimentInput(StrictModel):
    case_id: str
    dataset_id: str
    case_version: str = "1.0"
    task_type: str
    query: str
    documents: list[DocumentInput] = Field(default_factory=list)
    expected_output: dict[str, Any] | None = None
    evaluation_criteria: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(StrictModel):
    experiment_id: str
    run_id: str
    run_index: int = 0
    framework: str
    framework_version: str | None = None
    architecture: str
    model_provider: str
    model_name: str
    temperature: float = 0.0
    top_p: float | None = None
    max_tokens: int | None = None
    max_agent_iterations: int = 10
    timeout_seconds: int = 120
    retry_count: int = 0
    random_seed: int | None = None
    collect_resource_usage: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class TokenUsage(StrictModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0


class LLMCallMetrics(StrictModel):
    call_id: str
    step_id: int | None = None
    model_provider: str
    model_name: str
    latency_seconds: float = 0.0
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    estimated_cost_usd: float = 0.0
    finish_reason: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentStep(StrictModel):
    step_id: int
    name: str
    step_type: str
    actor: str | None = None
    input_data: dict[str, Any] = Field(default_factory=dict)
    output_data: dict[str, Any] = Field(default_factory=dict)
    llm_call_ids: list[str] = Field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResourceUsage(StrictModel):
    cpu_percent_start: float | None = None
    cpu_percent_end: float | None = None
    cpu_percent_avg: float | None = None
    cpu_percent_peak: float | None = None
    memory_mb_start: float | None = None
    memory_mb_end: float | None = None
    memory_mb_peak: float | None = None


class ExperimentError(StrictModel):
    error_type: str
    message: str
    step_id: int | None = None
    recoverable: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EnvironmentInfo(StrictModel):
    python_version: str | None = None
    os: str | None = None
    git_commit: str | None = None
    benchmark_core_version: str | None = None
    package_versions: dict[str, str] = Field(default_factory=dict)


class ExperimentMetrics(StrictModel):
    total_latency_seconds: float = 0.0
    llm_latency_seconds: float = 0.0
    step_count: int = 0
    llm_call_count: int = 0
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    estimated_cost_usd: float = 0.0
    error_count: int = 0
    resource_usage: ResourceUsage = Field(default_factory=ResourceUsage)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExperimentResult(StrictModel):
    schema_version: str = "1.0"
    case_id: str
    dataset_id: str
    framework: str
    architecture: str
    run_id: str
    status: RunStatus
    final_answer: str
    structured_output: dict[str, Any] = Field(default_factory=dict)
    input_snapshot: ExperimentInput
    config_snapshot: ExperimentConfig
    metrics: ExperimentMetrics = Field(default_factory=ExperimentMetrics)
    steps: list[AgentStep] = Field(default_factory=list)
    llm_calls: list[LLMCallMetrics] = Field(default_factory=list)
    errors: list[ExperimentError] = Field(default_factory=list)
    environment: EnvironmentInfo = Field(default_factory=EnvironmentInfo)
    started_at: datetime
    finished_at: datetime