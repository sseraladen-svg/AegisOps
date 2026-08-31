"""Pydantic response models for the HTTP surface.

These exist for the OpenAPI document as much as for validation: without
declared response models FastAPI documents every endpoint as returning an
untyped object, which makes /docs useless as a contract for the frontend. Each
model here is the shape the React client actually destructures.
"""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

DetectorName = Literal["naive", "isolation_forest", "lstm_autoencoder"]
LogLevel = Literal["DEBUG", "INFO", "WARN", "ERROR"]
Severity = Literal["low", "medium", "high"]
HealthStatus = Literal["healthy", "degraded", "critical", "unknown"]


class Health(BaseModel):
    status: Literal["ok"]
    results_available: bool = Field(description="whether eval/results.json exists yet")
    incident_count: int
    investigation_count: int


class ServiceOut(BaseModel):
    id: int
    name: str


class Point(BaseModel):
    ts: str
    value: float


class MetricSeries(BaseModel):
    service_id: int
    service_name: str
    metric_name: str
    ts_start: str
    ts_end: str
    count: int = Field(description="points returned after downsampling")
    total_available: int = Field(description="points in the range before downsampling")
    downsampled_by: int = Field(description="stride applied; 1 means no downsampling")
    unit: Optional[str] = None
    points: List[Point]


class LogLine(BaseModel):
    id: int
    service_id: int
    ts: str
    level: LogLevel
    message: str
    request_id: Optional[str] = None


class LogPage(BaseModel):
    service_id: Optional[int] = None
    ts_start: Optional[str] = None
    ts_end: Optional[str] = None
    level: Optional[LogLevel] = None
    total_matching: int = Field(description="rows matching the filter, ignoring paging")
    level_counts: Dict[str, int]
    offset: int
    limit: int
    logs: List[LogLine]


class IncidentOut(BaseModel):
    id: int
    service_id: int
    service_name: Optional[str] = None
    metric_name: str
    ts_start: str
    ts_end: str
    detector_source: str
    anomaly_score: float
    status: str
    duration_minutes: float
    has_investigation: bool = False


class GroundTruthOut(BaseModel):
    id: int
    service_id: int
    metric_name: str
    ts_start: str
    ts_end: str
    difficulty_tier: str


class EvidenceOut(BaseModel):
    claim: str
    source_tool: str
    detail: str


class InvestigationOut(BaseModel):
    incident_id: int
    hypothesis: Optional[str] = None
    confidence: Optional[float] = None
    severity: Optional[Severity] = None
    evidence: List[EvidenceOut] = []
    ruled_out: List[str] = []
    recommended_action: Optional[str] = None
    tools_used: List[str] = []
    stop_state: Optional[str] = None
    validation_warnings: List[str] = Field(
        default=[],
        description="claims the report made that its own tool trace does not support")
    github_issue_url: Optional[str] = None
    trace: List[Dict[str, Any]] = Field(
        default=[], description="ordered model turns and tool results")
    created_at: Optional[str] = None


class MetricSummary(BaseModel):
    metric_name: str
    current: Optional[float] = None
    mean: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    pct_change_vs_earlier: Optional[float] = Field(
        None, description="mean of the most recent quarter vs the rest of the lookback")
    sparkline: List[float] = Field(default=[], description="downsampled, oldest first")


class ServiceHealth(BaseModel):
    service_id: int
    name: str
    status: HealthStatus
    incident_count: int
    worst_anomaly_score: Optional[float] = None
    latest_incident_id: Optional[int] = None
    metrics: List[MetricSummary]


class HealthOverview(BaseModel):
    as_of: str
    lookback_hours: float
    detector: DetectorName
    thresholds: Dict[str, int] = Field(
        description="incident counts at which a service is called degraded / critical")
    services: List[ServiceHealth]


class DetectorScores(BaseModel):
    """One detector's row of the frozen comparison.

    precision / recall / f1 are positional: index i corresponds to
    EvalResults.tiers[i].
    """
    precision: List[float]
    recall: List[float]
    f1: List[float]
    overall: Dict[str, float]
    incidents: int = Field(description="windows this detector raised in total")
    false_positives: int = Field(description="of those, how many matched no ground truth")
    tuning: str = Field(
        description="how this detector's operating point was chosen — the detectors are "
                    "NOT tuned the same way, and the comparison is misleading without it")


class EvalResults(BaseModel):
    generated_at: str
    tiers: List[str]
    ground_truth_counts: Dict[str, int]
    detectors: Dict[str, DetectorScores]


class DemoIncident(BaseModel):
    tier: str
    label: str
    why_it_matters: str
    incident_id: Optional[int] = None
    service_id: int
    service_name: str
    metric_name: str
    ts_start: str
    ts_end: str
    caught_by: List[str]
    missed_by: List[str]


class DemoTour(BaseModel):
    generated_at: str
    incidents: List[DemoIncident]
