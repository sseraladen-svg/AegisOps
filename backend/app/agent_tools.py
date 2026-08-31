"""The four tools the investigation agent can call.

Each tool is a plain function returning a JSON-serialisable dict, so the same
implementation backs three callers without divergence: the Claude tool-use loop
(app/investigator.py), the HTTP endpoints under /api/tools (app/main.py), and
tests. TOOL_SCHEMAS below is the wire contract handed to the model.

Design rule for every tool here: return *evidence*, not verdicts. The agent is
asked to cite specific numbers and log lines, so a tool that returned "this
looks anomalous" would be doing the reasoning the report is supposed to show
its work for --- and would be uncheckable when it was wrong. query_metrics
therefore returns the window's statistics next to a preceding baseline
window's, and lets the model draw the conclusion.
"""
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timedelta

from sqlalchemy import func

from app.config import METRICS, SAMPLE_INTERVAL_MINUTES
from app.knowledge_base import get_index
from app.models import Log, Metric, Service

LOG_LEVELS = ['DEBUG', 'INFO', 'WARN', 'ERROR']
MAX_LOG_LIMIT = 200
MAX_METRIC_SAMPLES = 60

GITHUB_API = "https://api.github.com"


class ToolError(Exception):
    """A tool was called with input it can't honour.

    Raised rather than returned so the loop can mark the tool_result
    `is_error`, which is what tells the model to correct itself instead of
    treating a failure message as data.
    """


def _parse_ts(value, field):
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        raise ToolError(f"{field} must be an ISO-8601 timestamp string, got {type(value).__name__}")
    text = value.strip().replace('Z', '+00:00')
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ToolError(f"{field}={value!r} is not a valid ISO-8601 timestamp: {exc}") from exc
    # Telemetry is stored naive/UTC; comparing an aware value against it raises
    # deep inside SQLAlchemy with an unhelpful message.
    return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed


def _validate_range(ts_start, ts_end):
    start, end = _parse_ts(ts_start, 'ts_start'), _parse_ts(ts_end, 'ts_end')
    if end < start:
        raise ToolError(f"ts_end ({end.isoformat()}) is before ts_start ({start.isoformat()})")
    return start, end


def _require_service(session, service_id):
    try:
        service_id = int(service_id)
    except (TypeError, ValueError):
        raise ToolError(f"service_id must be an integer, got {service_id!r}") from None
    service = session.get(Service, service_id)
    if service is None:
        known = [(s.id, s.name) for s in session.query(Service).order_by(Service.id).all()]
        raise ToolError(f"no service with id {service_id}; known services: {known}")
    return service


# --------------------------------------------------------------------------
# tool 1: query_logs
# --------------------------------------------------------------------------

def query_logs(session, service_id, ts_start, ts_end, level=None, limit=50):
    """Log lines for one service in a time range, newest-relevant first."""
    service = _require_service(session, service_id)
    start, end = _validate_range(ts_start, ts_end)

    try:
        limit = max(1, min(int(limit), MAX_LOG_LIMIT))
    except (TypeError, ValueError):
        raise ToolError(f"limit must be an integer, got {limit!r}") from None

    query = session.query(Log).filter(
        Log.service_id == service.id, Log.ts >= start, Log.ts <= end)

    if level is not None:
        level = str(level).upper()
        if level not in LOG_LEVELS:
            raise ToolError(f"level must be one of {LOG_LEVELS}, got {level!r}")
        query = query.filter(Log.level == level)

    # Count by level before truncating: "3 of 40 lines were errors" and "3 of 3
    # were" support very different conclusions, and the truncated page alone
    # can't tell them apart.
    level_counts = dict(
        session.query(Log.level, func.count()).filter(
            Log.service_id == service.id, Log.ts >= start, Log.ts <= end,
        ).group_by(Log.level).all()
    )

    rows = query.order_by(Log.ts).limit(limit).all()
    return {
        "service_id": service.id,
        "service_name": service.name,
        "ts_start": start.isoformat(),
        "ts_end": end.isoformat(),
        "level_filter": level,
        "total_lines_in_window": sum(level_counts.values()),
        "level_counts": level_counts,
        "returned": len(rows),
        "truncated": len(rows) >= limit,
        "logs": [{
            "ts": row.ts.isoformat(),
            "level": row.level,
            "message": row.message,
            "request_id": row.request_id,
        } for row in rows],
    }


# --------------------------------------------------------------------------
# tool 2: query_metrics
# --------------------------------------------------------------------------

def _summarise(values):
    if not values:
        return None
    array = sorted(values)
    n = len(array)
    mean = sum(array) / n
    return {
        "count": n,
        "mean": round(mean, 3),
        "min": round(array[0], 3),
        "max": round(array[-1], 3),
        "p50": round(array[n // 2], 3),
        "p95": round(array[min(int(n * 0.95), n - 1)], 3),
    }


def query_metrics(session, service_id, metric_name, ts_start, ts_end, include_baseline=True):
    """Statistics for one metric in a window, alongside the preceding window.

    The baseline is the equally-long window immediately before the one asked
    about, which is what makes "cpu rose from 30 to 91" a statement the agent
    can make from tool output rather than from memory.
    """
    service = _require_service(session, service_id)
    start, end = _validate_range(ts_start, ts_end)

    metric_name = str(metric_name)
    if metric_name not in METRICS:
        raise ToolError(f"metric_name must be one of {METRICS}, got {metric_name!r}")

    def fetch(a, b):
        return [v for (v,) in session.query(Metric.value).filter(
            Metric.service_id == service.id, Metric.metric_name == metric_name,
            Metric.ts >= a, Metric.ts <= b,
        ).order_by(Metric.ts).all()]

    rows = session.query(Metric.ts, Metric.value).filter(
        Metric.service_id == service.id, Metric.metric_name == metric_name,
        Metric.ts >= start, Metric.ts <= end,
    ).order_by(Metric.ts).all()

    if not rows:
        raise ToolError(
            f"no {metric_name} samples for service {service.id} between "
            f"{start.isoformat()} and {end.isoformat()} — check the range")

    values = [v for _, v in rows]
    # A long window would otherwise return hundreds of points and crowd out the
    # rest of the evidence; stride keeps the shape while capping the token cost.
    stride = max(1, len(rows) // MAX_METRIC_SAMPLES)

    result = {
        "service_id": service.id,
        "service_name": service.name,
        "metric_name": metric_name,
        "ts_start": start.isoformat(),
        "ts_end": end.isoformat(),
        "sample_interval_minutes": SAMPLE_INTERVAL_MINUTES,
        "window": _summarise(values),
        "samples_downsampled_by": stride,
        "samples": [{"ts": ts.isoformat(), "value": round(v, 3)}
                    for ts, v in rows[::stride]],
    }

    if include_baseline:
        span = end - start
        baseline_end = start - timedelta(minutes=SAMPLE_INTERVAL_MINUTES)
        baseline_values = fetch(baseline_end - span, baseline_end)
        result["baseline"] = {
            "ts_start": (baseline_end - span).isoformat(),
            "ts_end": baseline_end.isoformat(),
            "stats": _summarise(baseline_values),
            "note": "equally-long window immediately before the one queried",
        }
        window_mean = result["window"]["mean"]
        base = result["baseline"]["stats"]
        if base and base["mean"]:
            result["baseline"]["pct_change_in_mean"] = round(
                100.0 * (window_mean - base["mean"]) / abs(base["mean"]), 2)

    return result


# --------------------------------------------------------------------------
# tool 3: query_similar_incidents
# --------------------------------------------------------------------------

def query_similar_incidents(session, query, k=3):
    """Nearest past write-ups from the FAISS index. `session` is unused but kept
    so every tool in this module shares one call signature."""
    try:
        k = max(1, min(int(k), 10))
    except (TypeError, ValueError):
        raise ToolError(f"k must be an integer, got {k!r}") from None

    try:
        hits = get_index().search(query, k=k)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc

    return {
        "query": query,
        "matches": [{
            "id": hit.incident["id"],
            "title": hit.incident["title"],
            "service": hit.incident.get("service"),
            "similarity": round(hit.score, 4),
            "summary": hit.incident.get("summary"),
            "root_cause": hit.incident.get("root_cause"),
            "evidence": hit.incident.get("evidence"),
            "resolution": hit.incident.get("resolution"),
            "lesson": hit.incident.get("lesson"),
        } for hit in hits],
        "note": ("similarity is cosine over TF-IDF vectors; a low score means "
                 "the corpus holds nothing comparable, not that the incident is novel"),
    }


# --------------------------------------------------------------------------
# tool 4: file_github_issue
# --------------------------------------------------------------------------

def github_mode():
    """'live' only when explicitly opted in *and* fully configured.

    Filing an issue is the one tool here with an effect outside this process,
    so it defaults to a dry run. Making it live takes a deliberate
    AGENT_GITHUB_MODE=live plus real credentials — an agent should never file
    against someone's repository because a config value happened to be present.
    """
    requested = os.environ.get('AGENT_GITHUB_MODE', 'dry_run').strip().lower()
    if requested != 'live':
        return 'dry_run', "AGENT_GITHUB_MODE is not 'live'"
    if not os.environ.get('GITHUB_TOKEN'):
        return 'dry_run', "AGENT_GITHUB_MODE=live but GITHUB_TOKEN is unset"
    if not os.environ.get('GITHUB_REPO'):
        return 'dry_run', "AGENT_GITHUB_MODE=live but GITHUB_REPO (owner/name) is unset"
    return 'live', None


def file_github_issue(session, title, body, labels=None):
    """File a GitHub issue, or describe the issue that would be filed."""
    if not title or not str(title).strip():
        raise ToolError("title must be a non-empty string")
    if not body or not str(body).strip():
        raise ToolError("body must be a non-empty string")

    if labels is None:
        labels = []
    elif isinstance(labels, str):
        labels = [labels]
    elif not isinstance(labels, list):
        raise ToolError(f"labels must be a list of strings, got {type(labels).__name__}")
    labels = [str(label) for label in labels]

    mode, reason = github_mode()
    payload = {"title": str(title).strip(), "body": str(body).strip(), "labels": labels}

    if mode == 'dry_run':
        return {
            "mode": "dry_run",
            "filed": False,
            "reason": reason,
            "issue": payload,
            "note": ("No issue was created. The report is still complete — say the "
                     "issue was prepared, not that it was filed."),
        }

    repo = os.environ['GITHUB_REPO']
    request = urllib.request.Request(
        f"{GITHUB_API}/repos/{repo}/issues",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": "agentic-copilot",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            created = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise ToolError(
            f"GitHub returned {exc.code} filing against {repo}: "
            f"{exc.read().decode(errors='replace')[:300]}") from exc
    except urllib.error.URLError as exc:
        raise ToolError(f"could not reach GitHub: {exc.reason}") from exc

    return {
        "mode": "live",
        "filed": True,
        "issue_number": created.get("number"),
        "issue_url": created.get("html_url"),
        "issue": payload,
    }


# --------------------------------------------------------------------------
# wire contract
# --------------------------------------------------------------------------

TOOL_FUNCTIONS = {
    "query_logs": query_logs,
    "query_metrics": query_metrics,
    "query_similar_incidents": query_similar_incidents,
    "file_github_issue": file_github_issue,
}

_TS = {"type": "string", "description": "ISO-8601 timestamp, e.g. 2026-01-01T08:20:00"}

TOOL_SCHEMAS = [
    {
        "name": "query_logs",
        "description": (
            "Read application log lines for one service over a time range. Returns "
            "the lines plus a count of every level in the window, so you can tell a "
            "handful of errors among hundreds of lines from a window that is all "
            "errors. Use this to find what the service was complaining about while "
            "a metric was anomalous."),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "service_id": {"type": "integer", "description": "1=auth-service, 2=payment-gateway, 3=user-profile"},
                "ts_start": _TS,
                "ts_end": _TS,
                "level": {"type": ["string", "null"], "enum": [*LOG_LEVELS, None],
                          "description": "Restrict to one level, or null for all levels."},
                "limit": {"type": "integer", "description": f"Max lines to return, 1-{MAX_LOG_LIMIT}."},
            },
            "required": ["service_id", "ts_start", "ts_end", "level", "limit"],
            "additionalProperties": False,
        },
    },
    {
        "name": "query_metrics",
        "description": (
            "Read one metric for one service over a time range. Returns summary "
            "statistics for the window, a downsampled series, and the same "
            "statistics for the equally-long window immediately before it, with the "
            "percent change between them. Use the baseline comparison to state how "
            "far a metric actually moved instead of estimating."),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "service_id": {"type": "integer", "description": "1=auth-service, 2=payment-gateway, 3=user-profile"},
                "metric_name": {"type": "string", "enum": METRICS},
                "ts_start": _TS,
                "ts_end": _TS,
                "include_baseline": {"type": "boolean",
                                     "description": "Include the preceding comparison window. Normally true."},
            },
            "required": ["service_id", "metric_name", "ts_start", "ts_end", "include_baseline"],
            "additionalProperties": False,
        },
    },
    {
        "name": "query_similar_incidents",
        "description": (
            "Search past incident write-ups for ones resembling what you are seeing. "
            "Describe the *pattern* in your own words — 'cpu rose with no change in "
            "request rate', 'latency ramping over hours' — rather than naming a "
            "service, since the useful matches are usually from other services. "
            "Each match includes its root cause and the lesson learned. Treat these "
            "as hypotheses to check against this incident's data, never as findings."),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Description of the observed pattern."},
                "k": {"type": "integer", "description": "How many matches to return, 1-10."},
            },
            "required": ["query", "k"],
            "additionalProperties": False,
        },
    },
    {
        "name": "file_github_issue",
        "description": (
            "Prepare a GitHub issue for this incident. Unless the deployment has "
            "explicitly enabled live filing, this is a dry run that returns the issue "
            "it would have created — check the `filed` field in the result and never "
            "claim an issue was filed when it is false. Call this once, last, after "
            "you have reached a conclusion. The body should be readable by an "
            "on-call engineer who has not seen any of your tool output."),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "One line, names the service and the symptom."},
                "body": {"type": "string", "description": "Markdown: what happened, evidence with concrete numbers, likely cause, recommended action."},
                "labels": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["title", "body", "labels"],
            "additionalProperties": False,
        },
    },
]


def dispatch(session, name, arguments):
    """Run one tool by name. Raises ToolError for anything the model got wrong."""
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        raise ToolError(f"unknown tool {name!r}; available: {sorted(TOOL_FUNCTIONS)}")
    if not isinstance(arguments, dict):
        raise ToolError(f"tool arguments must be an object, got {type(arguments).__name__}")
    try:
        return function(session, **arguments)
    except TypeError as exc:
        # Wrong/missing argument names arrive here; surfacing the signature error
        # verbatim lets the model fix its own call on the next turn.
        raise ToolError(f"bad arguments for {name}: {exc}") from exc
