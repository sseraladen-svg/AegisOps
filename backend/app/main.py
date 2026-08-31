"""FastAPI surface over the telemetry database, the frozen eval results, the
four agent tools, and the agent's investigations.

docker-compose points uvicorn at `app.main:app`, so this module has to exist
for the backend container to boot at all.

The /api/tools/* endpoints call the same functions the investigation agent
calls in-process (app/agent_tools.py) — exposed over HTTP so the tool surface
can be exercised, demoed, and tested without running a model. There is exactly
one implementation of each tool; this is a second door onto it, not a copy.

Every endpoint declares a response model. That is what makes /docs a usable
contract for the frontend rather than a list of endpoints returning `object`.
"""
import json
from collections import Counter
from datetime import datetime, timedelta, timezone

from fastapi import Depends, FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import func

from app import agent_tools
from app.agent_tools import TOOL_SCHEMAS, ToolError
from app.config import DETECTORS, METRICS, RESULTS_PATH, TIERS
from app.db import get_session
from app.detector_utils import windows_overlap
from app.models import GroundTruthAnomaly, Incident, Investigation, Log, Metric, Service
from app.schemas import (
    DemoIncident, DemoTour, EvalResults, EvidenceOut, GroundTruthOut, Health,
    HealthOverview, IncidentOut, InvestigationOut, LogPage, MetricSeries,
    MetricSummary, ServiceHealth, ServiceOut,
)

app = FastAPI(
    title="agentic-copilot",
    version="0.5.0",
    summary="Anomaly detection over service telemetry, with an investigation agent.",
    description=(
        "Three services emit four metrics every five minutes for fourteen days. "
        "Three detectors compete to find three injected anomalies of increasing "
        "subtlety, and an LLM agent investigates what they flag.\n\n"
        "**Units.** `cpu_usage` is percent, `latency_ms` milliseconds, "
        "`error_rate` percent, `request_rate` requests per minute.\n\n"
        "**Timestamps.** All timestamps are naive ISO-8601 in UTC. The dataset is "
        "anchored to a fixed start so runs are reproducible; 'now' in this API means "
        "the last sample in the database, not wall-clock time."
    ),
    openapi_tags=[
        {"name": "meta", "description": "Health and service inventory."},
        {"name": "telemetry", "description": "Raw metrics and logs."},
        {"name": "incidents", "description": "What the detectors flagged, and what the agent concluded."},
        {"name": "eval", "description": "The frozen detector comparison and its ground truth."},
        {"name": "tools", "description": "The four agent tools, over HTTP."},
    ],
)

# The Vite dev server runs on a different origin than uvicorn, so the browser
# needs these headers to let the dashboard fetch anything at all.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:4173", "http://127.0.0.1:4173",   # vite preview
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

METRIC_UNITS = {
    'cpu_usage': 'percent',
    'latency_ms': 'ms',
    'error_rate': 'percent',
    'request_rate': 'req/min',
}

# A service is called degraded at this many incidents in the lookback window,
# critical at this many. Deliberately crude and stated in the response rather
# than hidden: with four real anomalies in fourteen days, any status rule is a
# presentation choice, not a measurement.
DEGRADED_AT = 1
CRITICAL_AT = 3

DEFAULT_DETECTOR = 'lstm_autoencoder'
SPARKLINE_POINTS = 48


def db_session():
    """Request-scoped session. FastAPI closes it after the response is sent."""
    session = get_session()
    try:
        yield session
    finally:
        session.close()


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------

def _parse_ts(value, field):
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace('Z', '+00:00'))
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"{field}={value!r} is not a valid ISO-8601 timestamp: {exc}") from exc
    return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed


def _validate_range(ts_start, ts_end):
    start, end = _parse_ts(ts_start, 'ts_start'), _parse_ts(ts_end, 'ts_end')
    if start and end and end < start:
        raise HTTPException(
            status_code=422,
            detail=f"ts_end ({end.isoformat()}) is before ts_start ({start.isoformat()})")
    return start, end


def _bucket_mean(values, buckets):
    """Downsample to `buckets` points by averaging, not striding.

    Striding is right for the timeline chart (it preserves real observed values,
    so a spike survives). It is wrong here: this data has a 24-hour cycle, and
    taking every Nth sample of it aliases the cycle into a sawtooth that looks
    like noise. A sparkline is a trend cue, so averaging each bucket is what
    shows the trend it is claiming to show.
    """
    if not values:
        return []
    if len(values) <= buckets:
        return [round(v, 3) for v in values]
    size = len(values) / buckets
    out = []
    for i in range(buckets):
        chunk = values[int(i * size):max(int((i + 1) * size), int(i * size) + 1)]
        out.append(round(sum(chunk) / len(chunk), 3))
    return out


def _service_names(session):
    return {s.id: s.name for s in session.query(Service).all()}


def _require_service(session, service_id):
    service = session.get(Service, service_id)
    if service is None:
        known = sorted(_service_names(session).items())
        raise HTTPException(
            status_code=404, detail=f"no service with id {service_id}; known: {known}")
    return service


def _dataset_end(session):
    """The last sample in the database — this dataset's notion of 'now'."""
    latest = session.query(func.max(Metric.ts)).scalar()
    if latest is None:
        raise HTTPException(
            status_code=503,
            detail="the metrics table is empty — run `make seed` (or `make pipeline`) first")
    return latest


def _incident_out(incident, names, investigated_ids=frozenset()):
    return IncidentOut(
        id=incident.id,
        service_id=incident.service_id,
        service_name=names.get(incident.service_id),
        metric_name=incident.metric_name,
        ts_start=incident.ts_start.isoformat(),
        ts_end=incident.ts_end.isoformat(),
        detector_source=incident.detector_source,
        anomaly_score=incident.anomaly_score,
        status=incident.status,
        duration_minutes=round((incident.ts_end - incident.ts_start).total_seconds() / 60.0, 1),
        has_investigation=incident.id in investigated_ids,
    )


# --------------------------------------------------------------------------
# meta
# --------------------------------------------------------------------------

@app.get("/health", response_model=Health, tags=["meta"],
         summary="Liveness plus whether the pipeline has been run")
def health(session=Depends(db_session)):
    return Health(
        status="ok",
        results_available=RESULTS_PATH.exists(),
        incident_count=session.query(func.count(Incident.id)).scalar() or 0,
        investigation_count=session.query(func.count(Investigation.id)).scalar() or 0,
    )


@app.get("/api/services", response_model=list[ServiceOut], tags=["meta"])
def list_services(session=Depends(db_session)):
    return [ServiceOut(id=s.id, name=s.name)
            for s in session.query(Service).order_by(Service.id).all()]


@app.get("/api/services/health", response_model=HealthOverview, tags=["meta"],
         summary="Per-service status pill, sparklines, and incident counts")
def services_health(
    detector: str = Query(DEFAULT_DETECTOR, description=f"one of {DETECTORS}"),
    lookback_hours: float = Query(
        24.0, gt=0, le=24 * 365,
        description="window ending at the last sample in the database; 336 covers the whole dataset"),
    session=Depends(db_session),
):
    """Everything the overview page's tiles need, in one request.

    Status is derived from this detector's incident count in the window, and the
    thresholds are returned alongside it — a caller should be able to see the
    rule, not just its verdict.
    """
    if detector not in DETECTORS:
        raise HTTPException(status_code=422,
                            detail=f"unknown detector '{detector}'; expected one of {DETECTORS}")

    as_of = _dataset_end(session)
    window_start = as_of - timedelta(hours=lookback_hours)

    incidents = session.query(Incident).filter(
        Incident.detector_source == detector,
        Incident.ts_end >= window_start,
        Incident.ts_start <= as_of,
    ).all()

    by_service = {}
    for incident in incidents:
        by_service.setdefault(incident.service_id, []).append(incident)

    services = []
    for service in session.query(Service).order_by(Service.id).all():
        service_incidents = by_service.get(service.id, [])
        count = len(service_incidents)
        status = ("critical" if count >= CRITICAL_AT
                  else "degraded" if count >= DEGRADED_AT
                  else "healthy")

        summaries = []
        for metric_name in METRICS:
            values = [v for (v,) in session.query(Metric.value).filter(
                Metric.service_id == service.id,
                Metric.metric_name == metric_name,
                Metric.ts >= window_start, Metric.ts <= as_of,
            ).order_by(Metric.ts).all()]

            if not values:
                summaries.append(MetricSummary(metric_name=metric_name))
                continue

            # Compare the most recent quarter of the window against the rest, so
            # the tile can say which way a metric is trending without the caller
            # making a second ranged request per metric.
            split = max(1, len(values) * 3 // 4)
            earlier, recent = values[:split], values[split:]
            pct = None
            if earlier and recent:
                earlier_mean = sum(earlier) / len(earlier)
                if abs(earlier_mean) > 1e-9:
                    pct = round(100.0 * ((sum(recent) / len(recent)) - earlier_mean)
                                / abs(earlier_mean), 2)

            summaries.append(MetricSummary(
                metric_name=metric_name,
                current=round(values[-1], 3),
                mean=round(sum(values) / len(values), 3),
                min=round(min(values), 3),
                max=round(max(values), 3),
                pct_change_vs_earlier=pct,
                sparkline=_bucket_mean(values, SPARKLINE_POINTS),
            ))

        worst = max(service_incidents, key=lambda i: i.anomaly_score, default=None)
        latest = max(service_incidents, key=lambda i: i.ts_start, default=None)
        services.append(ServiceHealth(
            service_id=service.id,
            name=service.name,
            status=status,
            incident_count=count,
            worst_anomaly_score=round(worst.anomaly_score, 4) if worst else None,
            latest_incident_id=latest.id if latest else None,
            metrics=summaries,
        ))

    return HealthOverview(
        as_of=as_of.isoformat(),
        lookback_hours=lookback_hours,
        detector=detector,
        thresholds={"degraded_at": DEGRADED_AT, "critical_at": CRITICAL_AT},
        services=services,
    )


# --------------------------------------------------------------------------
# telemetry
# --------------------------------------------------------------------------

@app.get("/api/metrics", response_model=MetricSeries, tags=["telemetry"],
         summary="One metric series for one service, downsampled to fit a chart")
def get_metrics(
    service_id: int = Query(..., ge=1),
    metric_name: str = Query(..., description=f"one of {METRICS}"),
    ts_start: str | None = Query(None, description="ISO-8601; defaults to the start of the data"),
    ts_end: str | None = Query(None, description="ISO-8601; defaults to the end of the data"),
    max_points: int = Query(
        2000, ge=10, le=20000,
        description="the series is strided down to at most this many points"),
    session=Depends(db_session),
):
    """Downsamples by striding rather than averaging.

    A timeline chart is being asked "did this spike", and averaging buckets is
    exactly the operation that hides a spike. Striding preserves real observed
    values; the response reports the stride so a caller knows what it got.
    """
    service = _require_service(session, service_id)
    if metric_name not in METRICS:
        raise HTTPException(status_code=422,
                            detail=f"unknown metric '{metric_name}'; expected one of {METRICS}")
    start, end = _validate_range(ts_start, ts_end)

    query = session.query(Metric.ts, Metric.value).filter(
        Metric.service_id == service.id, Metric.metric_name == metric_name)
    if start:
        query = query.filter(Metric.ts >= start)
    if end:
        query = query.filter(Metric.ts <= end)
    rows = query.order_by(Metric.ts).all()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"no {metric_name} samples for service {service.id} in that range")

    stride = max(1, len(rows) // max_points)
    sampled = rows[::stride]
    return MetricSeries(
        service_id=service.id,
        service_name=service.name,
        metric_name=metric_name,
        ts_start=rows[0][0].isoformat(),
        ts_end=rows[-1][0].isoformat(),
        count=len(sampled),
        total_available=len(rows),
        downsampled_by=stride,
        unit=METRIC_UNITS.get(metric_name),
        points=[{"ts": ts.isoformat(), "value": round(value, 4)} for ts, value in sampled],
    )


@app.get("/api/logs", response_model=LogPage, tags=["telemetry"],
         summary="Paged log lines, with level counts for the whole match")
def get_logs(
    service_id: int | None = Query(None, ge=1),
    ts_start: str | None = Query(None),
    ts_end: str | None = Query(None),
    level: str | None = Query(None, description=f"one of {agent_tools.LOG_LEVELS}"),
    search: str | None = Query(None, min_length=2, description="case-insensitive substring of the message"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    session=Depends(db_session),
):
    if level is not None:
        level = level.upper()
        if level not in agent_tools.LOG_LEVELS:
            raise HTTPException(
                status_code=422,
                detail=f"level must be one of {agent_tools.LOG_LEVELS}, got {level!r}")
    start, end = _validate_range(ts_start, ts_end)
    if service_id is not None:
        _require_service(session, service_id)

    def apply_filters(query, include_level=True):
        if service_id is not None:
            query = query.filter(Log.service_id == service_id)
        if start:
            query = query.filter(Log.ts >= start)
        if end:
            query = query.filter(Log.ts <= end)
        if search:
            query = query.filter(Log.message.ilike(f"%{search}%"))
        if include_level and level:
            query = query.filter(Log.level == level)
        return query

    total = apply_filters(session.query(func.count(Log.id))).scalar() or 0
    # Level counts ignore the level filter on purpose: a UI showing only errors
    # still needs to say "3 of 40 lines", which the filtered count cannot give.
    level_counts = dict(
        apply_filters(session.query(Log.level, func.count()), include_level=False)
        .group_by(Log.level).all())

    rows = apply_filters(session.query(Log)).order_by(Log.ts).offset(offset).limit(limit).all()
    return LogPage(
        service_id=service_id,
        ts_start=start.isoformat() if start else None,
        ts_end=end.isoformat() if end else None,
        level=level,
        total_matching=total,
        level_counts=level_counts,
        offset=offset,
        limit=limit,
        logs=[{
            "id": row.id, "service_id": row.service_id, "ts": row.ts.isoformat(),
            "level": row.level, "message": row.message, "request_id": row.request_id,
        } for row in rows],
    )


# --------------------------------------------------------------------------
# incidents
# --------------------------------------------------------------------------

@app.get("/api/incidents", response_model=list[IncidentOut], tags=["incidents"])
def list_incidents(
    detector: str | None = Query(None, description=f"one of {DETECTORS}"),
    service_id: int | None = Query(None, ge=1),
    metric_name: str | None = Query(None),
    investigated: bool | None = Query(None, description="filter to incidents with/without a report"),
    ts_start: str | None = Query(None),
    ts_end: str | None = Query(None),
    order_by: str = Query("ts_start", pattern="^(ts_start|anomaly_score)$"),
    limit: int = Query(500, ge=1, le=5000),
    session=Depends(db_session),
):
    if detector is not None and detector not in DETECTORS:
        raise HTTPException(status_code=422,
                            detail=f"unknown detector '{detector}'; expected one of {DETECTORS}")
    if metric_name is not None and metric_name not in METRICS:
        raise HTTPException(status_code=422,
                            detail=f"unknown metric '{metric_name}'; expected one of {METRICS}")
    start, end = _validate_range(ts_start, ts_end)

    query = session.query(Incident)
    if detector is not None:
        query = query.filter(Incident.detector_source == detector)
    if service_id is not None:
        query = query.filter(Incident.service_id == service_id)
    if metric_name is not None:
        query = query.filter(Incident.metric_name == metric_name)
    if start:
        query = query.filter(Incident.ts_end >= start)
    if end:
        query = query.filter(Incident.ts_start <= end)

    order = Incident.anomaly_score.desc() if order_by == "anomaly_score" else Incident.ts_start
    rows = query.order_by(order).limit(limit).all()

    investigated_ids = {
        i for (i,) in session.query(Investigation.incident_id).distinct().all()}
    if investigated is not None:
        rows = [r for r in rows if (r.id in investigated_ids) == investigated]

    names = _service_names(session)
    return [_incident_out(row, names, investigated_ids) for row in rows]


@app.get("/api/incidents/{incident_id}", response_model=IncidentOut, tags=["incidents"])
def get_incident(incident_id: int = Path(..., ge=1), session=Depends(db_session)):
    incident = session.get(Incident, incident_id)
    if incident is None:
        raise HTTPException(status_code=404, detail=f"no incident with id {incident_id}")
    investigated = {i for (i,) in session.query(Investigation.incident_id).distinct().all()}
    return _incident_out(incident, _service_names(session), investigated)


@app.get("/api/incidents/{incident_id}/investigation", response_model=InvestigationOut,
         tags=["incidents"], summary="The agent's report and its full tool trace")
def get_investigation(incident_id: int = Path(..., ge=1), session=Depends(db_session)):
    if session.get(Incident, incident_id) is None:
        raise HTTPException(status_code=404, detail=f"no incident with id {incident_id}")

    investigation = session.query(Investigation).filter(
        Investigation.incident_id == incident_id).order_by(
        Investigation.created_at.desc()).first()
    if investigation is None:
        raise HTTPException(
            status_code=404,
            detail=f"incident {incident_id} has not been investigated yet — run "
                   f"`python -m app.investigator --incident-id {incident_id}`")

    evidence = investigation.evidence_json or {}
    return InvestigationOut(
        incident_id=incident_id,
        hypothesis=investigation.hypothesis,
        confidence=investigation.confidence,
        severity=evidence.get("severity"),
        evidence=[EvidenceOut(**e) for e in evidence.get("evidence", [])],
        ruled_out=evidence.get("ruled_out", []),
        recommended_action=evidence.get("recommended_action"),
        tools_used=evidence.get("tools_used", []),
        stop_state=evidence.get("stop_state"),
        validation_warnings=evidence.get("validation_warnings", []),
        github_issue_url=investigation.github_issue_url,
        trace=investigation.tool_calls_json or [],
        created_at=investigation.created_at.isoformat() if investigation.created_at else None,
    )


# --------------------------------------------------------------------------
# eval
# --------------------------------------------------------------------------

@app.get("/api/eval/results", response_model=EvalResults, tags=["eval"],
         summary="The frozen detector comparison the evaluation page renders")
def eval_results():
    if not RESULTS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="eval/results.json not found — run `make eval` to generate it")
    try:
        return json.loads(RESULTS_PATH.read_text())
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500, detail=f"eval/results.json is not valid JSON: {exc}") from exc


@app.get("/api/ground-truth", response_model=list[GroundTruthOut], tags=["eval"])
def list_ground_truth(session=Depends(db_session)):
    rows = session.query(GroundTruthAnomaly).order_by(GroundTruthAnomaly.ts_start).all()
    return [GroundTruthOut(
        id=gt.id, service_id=gt.service_id, metric_name=gt.metric_name,
        ts_start=gt.ts_start.isoformat(), ts_end=gt.ts_end.isoformat(),
        difficulty_tier=gt.difficulty_tier,
    ) for gt in rows]


@app.get("/api/eval/demo-tour", response_model=DemoTour, tags=["eval"],
         summary="One curated incident per difficulty tier, with who caught it")
def demo_tour(session=Depends(db_session)):
    """The three-stop tour: an obvious spike, a gradual drift, and the subtle
    correlated case. Each stop reports which detectors actually caught it,
    computed from the incidents table rather than asserted."""
    ground_truth = session.query(GroundTruthAnomaly).order_by(
        GroundTruthAnomaly.ts_start).all()
    if not ground_truth:
        raise HTTPException(status_code=503, detail="no ground truth — run `make seed` first")

    names = _service_names(session)
    incidents = session.query(Incident).all()
    stops = []

    for tier in TIERS:
        tier_gt = [gt for gt in ground_truth if gt.difficulty_tier == tier]
        if not tier_gt:
            continue
        gt = tier_gt[0]

        matches = [
            inc for inc in incidents
            if (inc.service_id, inc.metric_name) == (gt.service_id, gt.metric_name)
            and windows_overlap(inc.ts_start, inc.ts_end, gt.ts_start, gt.ts_end)
        ]
        # Ordered by DETECTORS, not alphabetically: the three chips always
        # appear in the same order so a reader can compare tiers by position.
        found = {inc.detector_source for inc in matches}
        caught = [d for d in DETECTORS if d in found]
        best = max(matches, key=lambda i: i.anomaly_score, default=None)

        stops.append(DemoIncident(
            tier=tier,
            label=DEMO_LABELS[tier]["label"],
            why_it_matters=DEMO_LABELS[tier]["why"],
            incident_id=best.id if best else None,
            service_id=gt.service_id,
            service_name=names.get(gt.service_id, "unknown"),
            metric_name=gt.metric_name,
            ts_start=gt.ts_start.isoformat(),
            ts_end=gt.ts_end.isoformat(),
            caught_by=caught,
            missed_by=[d for d in DETECTORS if d not in caught],
        ))

    return DemoTour(
        generated_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        incidents=stops,
    )


DEMO_LABELS = {
    "obvious_spike": {
        "label": "A CPU step change every detector catches",
        "why": "The control case. If a detector misses this, nothing else it reports is trustworthy.",
    },
    "gradual_drift": {
        "label": "Latency ramping over twelve hours",
        "why": "No single sample is anomalous, so a fixed z-score threshold is structurally unable "
               "to see it. This is where the naive detector fails.",
    },
    "subtle_correlated": {
        "label": "A small simultaneous rise on two services",
        "why": "Each service alone looks like noise. Only a model that sees services together has "
               "the information to catch it — which is the case for the multivariate autoencoder.",
    },
}


# --------------------------------------------------------------------------
# the four agent tools, over HTTP
# --------------------------------------------------------------------------

class QueryLogsRequest(BaseModel):
    service_id: int = Field(ge=1)
    ts_start: str
    ts_end: str
    level: str | None = None
    limit: int = Field(50, ge=1, le=agent_tools.MAX_LOG_LIMIT)


class QueryMetricsRequest(BaseModel):
    service_id: int = Field(ge=1)
    metric_name: str = Field(description=f"one of {METRICS}")
    ts_start: str
    ts_end: str
    include_baseline: bool = True


class SimilarIncidentsRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(3, ge=1, le=10)


class FileIssueRequest(BaseModel):
    title: str = Field(min_length=1)
    body: str = Field(min_length=1)
    labels: list[str] = Field(default_factory=list)


def _run_tool(name, session, **kwargs):
    """Shared adapter: a ToolError is the caller's fault, so it's a 400.

    Without this the agent and the HTTP surface would disagree about what
    counts as a bad request, and a tool fix would have to be made twice.
    """
    try:
        return agent_tools.dispatch(session, name, kwargs)
    except ToolError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/tools", tags=["tools"], summary="The exact tool schemas handed to the model",
         description="Response shape is the JSON-Schema tool contract itself, so it is "
                     "intentionally untyped here — `tools[].input_schema` is the "
                     "authority for what each /api/tools/* endpoint accepts.")
def list_tools():
    return {"tools": TOOL_SCHEMAS, "github_mode": agent_tools.github_mode()[0]}


@app.post("/api/tools/query_logs", tags=["tools"])
def tool_query_logs(request: QueryLogsRequest, session=Depends(db_session)):
    return _run_tool("query_logs", session, **request.model_dump())


@app.post("/api/tools/query_metrics", tags=["tools"])
def tool_query_metrics(request: QueryMetricsRequest, session=Depends(db_session)):
    return _run_tool("query_metrics", session, **request.model_dump())


@app.post("/api/tools/query_similar_incidents", tags=["tools"])
def tool_query_similar_incidents(request: SimilarIncidentsRequest, session=Depends(db_session)):
    return _run_tool("query_similar_incidents", session, **request.model_dump())


@app.post("/api/tools/file_github_issue", tags=["tools"],
          summary="Prepare (or, when explicitly enabled, file) a GitHub issue")
def tool_file_github_issue(request: FileIssueRequest, session=Depends(db_session)):
    """Defaults to a dry run — check the `filed` field in the response."""
    return _run_tool("file_github_issue", session, **request.model_dump())
