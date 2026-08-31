"""The four agent tools.

These are the project's Day 4 deliverable and the surface the model actually
drives, but nothing exercised them: a tool that raises the wrong exception type
turns a recoverable "you passed a bad timestamp" into a crashed investigation,
and a tool that returns a verdict instead of evidence silently does the
reasoning the report is supposed to show its work for.

Every bad-input case here asserts ToolError specifically, because that is what
run_tool_loop marks `is_error` on — which is what lets the model correct itself
rather than treat a failure string as data.
"""
from datetime import datetime, timedelta

import pytest

from app import agent_tools
from app.agent_tools import ToolError, dispatch
from app.models import Log, Metric, Service

START = datetime(2026, 1, 1, 0, 0, 0)


@pytest.fixture()
def tools_session(db_session):
    """One service with a flat baseline and a clearly raised second half."""
    db_session.add(Service(id=1, name='auth-service'))
    db_session.add(Service(id=2, name='payment-gateway'))

    for i in range(48):                       # 4 hours at 5-minute sampling
        ts = START + timedelta(minutes=5 * i)
        # First half sits at 30, second half at 90 — a doubling the baseline
        # comparison has to report, not a trend the test has to eyeball.
        db_session.add(Metric(service_id=1, metric_name='cpu_usage', ts=ts,
                              value=30.0 if i < 24 else 90.0))
        db_session.add(Metric(service_id=1, metric_name='request_rate', ts=ts,
                              value=100.0))

    for i in range(12):
        ts = START + timedelta(minutes=10 * i)
        db_session.add(Log(service_id=1, ts=ts,
                           level='ERROR' if i % 3 == 0 else 'INFO',
                           message=f"upstream timeout after {400 + i}ms calling billing-api",
                           request_id=f"req-{i}"))

    db_session.commit()
    return db_session


def iso(minutes):
    return (START + timedelta(minutes=minutes)).isoformat()


# --------------------------------------------------------------------------
# query_metrics
# --------------------------------------------------------------------------

def test_query_metrics_reports_the_window_against_its_baseline(tools_session):
    result = agent_tools.query_metrics(
        tools_session, 1, 'cpu_usage', iso(120), iso(235), include_baseline=True)

    assert result["window"]["mean"] == 90.0
    assert result["baseline"]["stats"]["mean"] == 30.0
    # The whole point of the baseline: "rose 200%" is a claim the agent can
    # make from tool output rather than from memory.
    assert result["baseline"]["pct_change_in_mean"] == 200.0
    # Evidence, not verdicts: the tool must not label the window for the model.
    assert "anomaly" not in str(result).lower()


def test_query_metrics_omits_the_baseline_when_not_asked(tools_session):
    result = agent_tools.query_metrics(
        tools_session, 1, 'cpu_usage', iso(120), iso(235), include_baseline=False)
    assert "baseline" not in result


def test_query_metrics_downsamples_and_says_by_how_much(tools_session):
    result = agent_tools.query_metrics(
        tools_session, 1, 'cpu_usage', iso(0), iso(235), include_baseline=False)
    assert len(result["samples"]) <= agent_tools.MAX_METRIC_SAMPLES
    assert result["samples_downsampled_by"] >= 1


def test_query_metrics_rejects_an_unknown_metric(tools_session):
    with pytest.raises(ToolError, match="metric_name must be one of"):
        agent_tools.query_metrics(tools_session, 1, 'disk_io', iso(0), iso(60))


def test_query_metrics_rejects_an_inverted_range(tools_session):
    with pytest.raises(ToolError, match="is before"):
        agent_tools.query_metrics(tools_session, 1, 'cpu_usage', iso(120), iso(0))


def test_query_metrics_rejects_an_unparseable_timestamp(tools_session):
    with pytest.raises(ToolError, match="not a valid ISO-8601"):
        agent_tools.query_metrics(tools_session, 1, 'cpu_usage', "yesterday", iso(60))


def test_query_metrics_errors_on_an_empty_window_rather_than_returning_nothing(tools_session):
    """An empty result is the model's cue to widen the range; a silent {} is not."""
    with pytest.raises(ToolError, match="check the range"):
        agent_tools.query_metrics(
            tools_session, 1, 'cpu_usage', iso(10_000), iso(10_060))


def test_query_metrics_rejects_an_unknown_service_and_names_the_real_ones(tools_session):
    with pytest.raises(ToolError, match="known services"):
        agent_tools.query_metrics(tools_session, 99, 'cpu_usage', iso(0), iso(60))


def test_aware_timestamps_are_accepted_and_normalised(tools_session):
    """Telemetry is stored naive/UTC; a Z-suffixed input must not explode deep
    inside SQLAlchemy."""
    result = agent_tools.query_metrics(
        tools_session, 1, 'cpu_usage', "2026-01-01T02:00:00Z", "2026-01-01T03:55:00Z",
        include_baseline=False)
    assert result["window"]["count"] > 0


# --------------------------------------------------------------------------
# query_logs
# --------------------------------------------------------------------------

def test_query_logs_counts_every_level_in_the_window_not_just_the_page(tools_session):
    """"3 of 40 lines were errors" and "3 of 3 were" support very different
    conclusions, and a truncated page alone cannot tell them apart."""
    result = agent_tools.query_logs(tools_session, 1, iso(0), iso(115), limit=2)

    assert len(result["logs"]) == 2
    assert result["truncated"] is True
    assert result["total_lines_in_window"] == 12
    assert result["level_counts"] == {"ERROR": 4, "INFO": 8}


def test_query_logs_level_filter_narrows_the_page_but_not_the_counts(tools_session):
    result = agent_tools.query_logs(tools_session, 1, iso(0), iso(115), level='error')
    assert {line["level"] for line in result["logs"]} == {"ERROR"}
    assert result["level_counts"]["INFO"] == 8      # unfiltered, on purpose


def test_query_logs_rejects_an_unknown_level(tools_session):
    with pytest.raises(ToolError, match="level must be one of"):
        agent_tools.query_logs(tools_session, 1, iso(0), iso(115), level='CRITICAL')


def test_query_logs_clamps_an_oversized_limit(tools_session):
    result = agent_tools.query_logs(tools_session, 1, iso(0), iso(115), limit=10_000)
    assert len(result["logs"]) <= agent_tools.MAX_LOG_LIMIT


def test_query_logs_on_a_quiet_service_returns_an_empty_page_not_an_error(tools_session):
    result = agent_tools.query_logs(tools_session, 2, iso(0), iso(115))
    assert result["logs"] == []
    assert result["total_lines_in_window"] == 0


# --------------------------------------------------------------------------
# query_similar_incidents
# --------------------------------------------------------------------------

def test_query_similar_incidents_returns_ranked_matches_with_their_lessons(tools_session):
    result = agent_tools.query_similar_incidents(
        tools_session, "cpu rose with no change in request rate", k=3)

    assert 1 <= len(result["matches"]) <= 3
    scores = [m["similarity"] for m in result["matches"]]
    assert scores == sorted(scores, reverse=True)
    top = result["matches"][0]
    assert top["root_cause"] and top["lesson"]


def test_query_similar_incidents_rejects_an_empty_query(tools_session):
    with pytest.raises(ToolError):
        agent_tools.query_similar_incidents(tools_session, "   ", k=3)


def test_query_similar_incidents_says_similarity_is_not_a_finding(tools_session):
    """The corpus is a source of hypotheses, never of evidence about *this*
    incident, and the tool result has to say so."""
    result = agent_tools.query_similar_incidents(tools_session, "latency ramp", k=1)
    assert "not that the incident is novel" in result["note"]


# --------------------------------------------------------------------------
# file_github_issue
# --------------------------------------------------------------------------

def test_file_github_issue_defaults_to_a_dry_run(tools_session, monkeypatch):
    monkeypatch.delenv('AGENT_GITHUB_MODE', raising=False)
    result = agent_tools.file_github_issue(tools_session, "title", "body", ["incident"])

    assert result["mode"] == "dry_run"
    assert result["filed"] is False
    assert "not" in result["note"].lower()


def test_live_mode_requires_credentials_and_falls_back_to_dry_run(monkeypatch):
    """A config value happening to be present must never be enough to file
    against someone's repository."""
    monkeypatch.setenv('AGENT_GITHUB_MODE', 'live')
    monkeypatch.delenv('GITHUB_TOKEN', raising=False)
    monkeypatch.delenv('GITHUB_REPO', raising=False)

    mode, reason = agent_tools.github_mode()
    assert mode == 'dry_run'
    assert 'GITHUB_TOKEN' in reason


def test_file_github_issue_rejects_an_empty_body(tools_session):
    with pytest.raises(ToolError, match="body must be"):
        agent_tools.file_github_issue(tools_session, "title", "   ")


def test_file_github_issue_accepts_a_bare_string_label(tools_session):
    result = agent_tools.file_github_issue(tools_session, "t", "b", "incident")
    assert result["issue"]["labels"] == ["incident"]


# --------------------------------------------------------------------------
# dispatch — the layer the model's mistakes actually hit
# --------------------------------------------------------------------------

def test_dispatch_routes_to_the_named_tool(tools_session):
    result = dispatch(tools_session, "query_logs", {
        "service_id": 1, "ts_start": iso(0), "ts_end": iso(115),
        "level": None, "limit": 5})
    assert result["service_name"] == 'auth-service'


def test_dispatch_rejects_an_unknown_tool_and_lists_the_real_ones(tools_session):
    with pytest.raises(ToolError, match="available:"):
        dispatch(tools_session, "query_traces", {})


def test_dispatch_turns_a_wrong_argument_name_into_a_correctable_error(tools_session):
    """A TypeError would crash the loop; a ToolError lets the model retry."""
    with pytest.raises(ToolError, match="bad arguments for query_logs"):
        dispatch(tools_session, "query_logs", {"service": 1})


def test_every_advertised_tool_is_dispatchable():
    """TOOL_SCHEMAS is the contract handed to the model — a name in it with no
    implementation behind it is a guaranteed mid-investigation failure."""
    for schema in agent_tools.TOOL_SCHEMAS:
        assert schema["name"] in agent_tools.TOOL_FUNCTIONS


def test_every_tool_schema_is_strict_and_closed():
    """strict mode requires additionalProperties: false and a full `required`."""
    for schema in agent_tools.TOOL_SCHEMAS:
        spec = schema["input_schema"]
        assert schema.get("strict") is True, schema["name"]
        assert spec["additionalProperties"] is False, schema["name"]
        assert set(spec["required"]) == set(spec["properties"]), schema["name"]
