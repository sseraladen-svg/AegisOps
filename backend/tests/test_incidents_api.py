"""Regression tests for /api/incidents paging.

The `investigated` filter used to be applied to the rows LIMIT had already
returned, which made `limit` cap the rows *scanned* rather than the rows
returned: `investigated=true&limit=5` answered "none" while five investigated
incidents sat just past the cut. The bug is invisible whenever `limit` exceeds
the table, which is why it survived — these tests pin the behaviour at a limit
small enough to expose it.
"""
from datetime import datetime, timedelta

import pytest

from app.models import Incident, Investigation, Service

START = datetime(2026, 1, 1)


@pytest.fixture()
def paged_session(db_session):
    """Twenty incidents; only the last five have reports."""
    db_session.add(Service(id=1, name='auth-service'))
    for i in range(20):
        db_session.add(Incident(
            id=i + 1, service_id=1, metric_name='cpu_usage',
            ts_start=START + timedelta(hours=i), ts_end=START + timedelta(hours=i, minutes=5),
            detector_source='lstm_autoencoder', anomaly_score=float(i), status='open'))
    for incident_id in range(16, 21):
        db_session.add(Investigation(
            incident_id=incident_id, tool_calls_json=[], hypothesis="stub",
            confidence=0.5, evidence_json={}))
    db_session.commit()
    return db_session


def test_investigated_filter_is_not_truncated_by_limit(api_client, paged_session):
    """The five investigated incidents are the newest, so a naive
    filter-after-limit returns zero of them."""
    rows = api_client.get("/api/incidents?investigated=true&limit=5").json()
    assert [r["id"] for r in rows] == [16, 17, 18, 19, 20]
    assert all(r["has_investigation"] for r in rows)


def test_uninvestigated_filter_respects_limit_as_a_page_size(api_client, paged_session):
    rows = api_client.get("/api/incidents?investigated=false&limit=5").json()
    assert len(rows) == 5
    assert not any(r["has_investigation"] for r in rows)


def test_limit_still_caps_the_unfiltered_result(api_client, paged_session):
    rows = api_client.get("/api/incidents?limit=7").json()
    assert len(rows) == 7


def test_filter_and_ordering_compose(api_client, paged_session):
    rows = api_client.get(
        "/api/incidents?investigated=true&order_by=anomaly_score&limit=3").json()
    assert [r["id"] for r in rows] == [20, 19, 18]


def test_has_investigation_flag_matches_the_filter(api_client, paged_session):
    everything = api_client.get("/api/incidents?limit=100").json()
    flagged = {r["id"] for r in everything if r["has_investigation"]}
    filtered = {r["id"] for r in api_client.get(
        "/api/incidents?investigated=true&limit=100").json()}
    assert flagged == filtered == {16, 17, 18, 19, 20}
