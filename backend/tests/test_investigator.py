"""The investigation loop and its report validator.

validate_report is the mechanical half of the project's central claim: the
prompt *asks* for cited evidence, and this is what *checks* it. An agent report
whose claims aren't backed by its own tool trace is worse than no report — it
is a confident wrong answer with a citation shape — so the check itself needs
tests more than most code here does.

The loop tests drive the scripted client from app/offline_agent.py, which
exercises everything around the model (dispatch, parallel tool_result batching,
trace capture, persistence) without an API key or a network call.
"""
from datetime import datetime, timedelta

import pytest

from app.investigator import (
    Evidence, InvestigationReport, build_incident_brief, investigate,
    run_tool_loop, validate_report,
)
from app.models import Incident, Investigation, Log, Metric, Service
from app.offline_agent import ScriptedClient

START = datetime(2026, 1, 1, 0, 0, 0)


def make_report(**overrides):
    defaults = dict(
        hypothesis="CPU rose without a matching change in request rate.",
        confidence=0.6,
        severity="medium",
        evidence=[Evidence(claim="cpu_usage rose", source_tool="query_metrics",
                           detail="mean 30.0 -> 90.0, +200%")],
        ruled_out=["Traffic: request_rate flat at 100/min."],
        recommended_action="Check the deploy that landed before the window.",
        issue_prepared=False,
        issue_filed=False,
    )
    defaults.update(overrides)
    return InvestigationReport(**defaults)


def stop_state(tools_used=('query_metrics',), github_result=None, state="completed"):
    return {"stop_state": state, "tools_used": list(tools_used),
            "github_result": github_result}


# --------------------------------------------------------------------------
# validate_report
# --------------------------------------------------------------------------

def test_a_well_supported_report_produces_no_warnings():
    assert validate_report(make_report(), stop_state()) == []


def test_evidence_citing_an_uncalled_tool_is_flagged():
    """The failure this whole check exists for: a claim attributed to a tool
    the loop never successfully ran."""
    report = make_report(evidence=[Evidence(
        claim="The logs showed timeouts", source_tool="query_logs",
        detail="4 ERROR lines of 12")])

    warnings = validate_report(report, stop_state(tools_used=['query_metrics']))
    assert len(warnings) == 1
    assert "query_logs" in warnings[0] and "never called" in warnings[0]


def test_evidence_without_a_concrete_value_is_flagged():
    """"CPU was elevated" is not a claim anyone can check."""
    report = make_report(evidence=[Evidence(
        claim="CPU was elevated", source_tool="query_metrics",
        detail="it was clearly much higher than usual")])

    warnings = validate_report(report, stop_state())
    assert any("no concrete value" in w for w in warnings)


def test_a_report_citing_nothing_at_all_is_flagged():
    warnings = validate_report(make_report(evidence=[]), stop_state())
    assert "report cites no evidence at all" in warnings


def test_claiming_an_issue_was_filed_when_it_was_a_dry_run_is_flagged():
    """The dry-run tool result says filed: false. A report saying otherwise is
    the one lie that would reach a real issue tracker."""
    report = make_report(issue_prepared=True, issue_filed=True)
    warnings = validate_report(report, stop_state(
        tools_used=['query_metrics', 'file_github_issue'],
        github_result={"filed": False, "mode": "dry_run"}))

    assert any("claims an issue was filed" in w for w in warnings)


def test_a_genuinely_filed_issue_is_not_flagged():
    report = make_report(issue_prepared=True, issue_filed=True)
    warnings = validate_report(report, stop_state(
        tools_used=['query_metrics', 'file_github_issue'],
        github_result={"filed": True, "issue_url": "https://example.invalid/1"}))
    assert warnings == []


def test_claiming_preparation_without_calling_the_tool_is_flagged():
    report = make_report(issue_prepared=True)
    warnings = validate_report(report, stop_state(tools_used=['query_metrics']))
    assert any("file_github_issue was never called" in w for w in warnings)


def test_high_confidence_on_thin_evidence_is_flagged():
    """Calibration, checked rather than requested."""
    warnings = validate_report(make_report(confidence=0.95), stop_state())
    assert any("above 0.8" in w for w in warnings)


def test_high_confidence_with_enough_evidence_is_accepted():
    report = make_report(confidence=0.95, evidence=[
        Evidence(claim="cpu rose", source_tool="query_metrics", detail="30 -> 90"),
        Evidence(claim="errors spiked", source_tool="query_logs", detail="4 of 12 lines"),
    ])
    warnings = validate_report(report, stop_state(
        tools_used=['query_metrics', 'query_logs']))
    assert warnings == []


def test_warnings_accumulate_rather_than_short_circuiting():
    report = make_report(confidence=0.99, evidence=[
        Evidence(claim="x", source_tool="query_logs", detail="no numbers here")])
    warnings = validate_report(report, stop_state(tools_used=['query_metrics']))
    assert len(warnings) >= 3


# --------------------------------------------------------------------------
# the loop, end to end, against the scripted client
# --------------------------------------------------------------------------

@pytest.fixture()
def incident_session(db_session):
    db_session.add(Service(id=1, name='auth-service'))
    for i in range(96):
        ts = START + timedelta(minutes=5 * i)
        db_session.add(Metric(service_id=1, metric_name='cpu_usage', ts=ts,
                              value=30.0 if i < 48 else 90.0))
        db_session.add(Metric(service_id=1, metric_name='request_rate', ts=ts,
                              value=100.0))
    for i in range(24):
        db_session.add(Log(service_id=1, ts=START + timedelta(minutes=20 * i),
                           level='ERROR' if i % 2 else 'INFO',
                           message=f"upstream timeout after {500 + i}ms calling billing-api",
                           request_id=f"req-{i}"))
    db_session.add(Incident(
        id=1, service_id=1, metric_name='cpu_usage',
        ts_start=START + timedelta(minutes=5 * 48),
        ts_end=START + timedelta(minutes=5 * 60),
        detector_source='lstm_autoencoder', anomaly_score=12.5, status='open'))
    db_session.commit()
    return db_session


def test_the_brief_withholds_the_data_the_agent_must_go_and_fetch(incident_session):
    incident = incident_session.get(Incident, 1)
    brief = build_incident_brief(incident_session, incident)

    assert "auth-service" in brief and "cpu_usage" in brief
    # If the brief handed over the values, a report could cite numbers without
    # ever calling a tool, and the whole evidence check would be theatre.
    assert "90.0" not in brief and "mean" not in brief


def test_the_loop_batches_parallel_tool_results_into_one_user_message(incident_session):
    """Splitting tool_results across messages teaches the model to stop making
    parallel calls, so the batching is a real invariant, not a style choice."""
    incident = incident_session.get(Incident, 1)
    brief = build_incident_brief(incident_session, incident)

    messages, trace, state = run_tool_loop(
        incident_session, ScriptedClient(), brief, verbose=False)

    tool_result_messages = [
        m for m in messages
        if m["role"] == "user" and isinstance(m["content"], list)
        and all(b.get("type") == "tool_result" for b in m["content"])
    ]
    assert tool_result_messages, "the loop never returned tool results"
    # The scripted client issues two parallel calls on its first two turns.
    assert any(len(m["content"]) == 2 for m in tool_result_messages)


def test_the_loop_records_an_ordered_trace_of_turns_and_results(incident_session):
    incident = incident_session.get(Incident, 1)
    brief = build_incident_brief(incident_session, incident)

    _, trace, state = run_tool_loop(
        incident_session, ScriptedClient(), brief, verbose=False)

    assert [step["step"] for step in trace] == list(range(1, len(trace) + 1))
    assert {step["type"] for step in trace} <= {"model_turn", "tool_result", "refusal"}
    assert state["stop_state"] == "completed"
    # All four tools are reachable from the loop, not just the two easy ones.
    assert set(state["tools_used"]) == {
        "query_metrics", "query_logs", "query_similar_incidents", "file_github_issue"}


def test_the_loop_stops_instead_of_running_forever(incident_session):
    incident = incident_session.get(Incident, 1)
    brief = build_incident_brief(incident_session, incident)

    _, _, state = run_tool_loop(
        incident_session, ScriptedClient(), brief, max_iterations=2, verbose=False)
    assert state["stop_state"] == "max_iterations"


def test_investigate_persists_a_report_and_its_trace(incident_session):
    incident = incident_session.get(Incident, 1)
    summary = investigate(incident_session, incident, ScriptedClient(), verbose=False)
    incident_session.commit()

    stored = incident_session.query(Investigation).filter(
        Investigation.incident_id == 1).one()

    assert stored.hypothesis and stored.confidence is not None
    assert stored.tool_calls_json, "the trace is a product here, not debug output"
    assert stored.evidence_json["tools_used"] == summary["tools_used"]
    # The dry-run tool result must not leave a fake issue URL behind.
    assert stored.github_issue_url is None


def test_rerunning_replaces_the_investigation_rather_than_accumulating(incident_session):
    incident = incident_session.get(Incident, 1)
    investigate(incident_session, incident, ScriptedClient(), verbose=False)
    investigate(incident_session, incident, ScriptedClient(), verbose=False)
    incident_session.commit()

    assert incident_session.query(Investigation).filter(
        Investigation.incident_id == 1).count() == 1


def test_the_scripted_report_survives_its_own_validator(incident_session):
    """The stub grounds its report in real tool output, so it should produce a
    clean bill of health — if it doesn't, the stub is fabricating."""
    incident = incident_session.get(Incident, 1)
    summary = investigate(incident_session, incident, ScriptedClient(), verbose=False)
    assert summary["validation_warnings"] == []


def test_the_scripted_client_is_stateless_across_incidents(incident_session):
    """One client is reused for every incident in a run; instance state would
    leak from one investigation into the next, and would do so silently."""
    incident = incident_session.get(Incident, 1)
    brief = build_incident_brief(incident_session, incident)
    client = ScriptedClient()

    first = run_tool_loop(incident_session, client, brief, verbose=False)[1]
    second = run_tool_loop(incident_session, client, brief, verbose=False)[1]

    assert [s["type"] for s in first] == [s["type"] for s in second]
