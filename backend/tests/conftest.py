"""Shared fixtures.

Tests run against an in-memory SQLite database, not backend/telemetry.db —
that file is real seeded output that may or may not exist yet (a fresh
checkout hasn't run `make seed`) and changes shape as seed.py evolves. A test
suite that only passes when someone happened to run `make seed` first isn't
actually testing anything reliably.
"""
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models import Base, GroundTruthAnomaly, Incident, Service


@pytest.fixture()
def db_session():
    # StaticPool + check_same_thread=False: a plain in-memory engine hands out
    # a fresh, empty database to every new connection, and the API test client
    # runs the app in a different thread than the test itself. Without both of
    # these, the app either can't see the data the test just inserted, or
    # sqlite3 refuses to hand the connection to that other thread at all.
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def api_client(db_session):
    """A TestClient wired to db_session instead of the real telemetry.db."""
    from fastapi.testclient import TestClient

    from app.main import app, db_session as db_session_dependency

    def override():
        yield db_session

    app.dependency_overrides[db_session_dependency] = override
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def ts(s):
    return datetime.fromisoformat(s)


@pytest.fixture()
def seeded_session(db_session):
    """A minimal, hand-built fixture standing in for one detector's output.

    One ground-truth row per tier (obvious_spike, gradual_drift), a second
    subtle_correlated row on a different service (mirroring seed.py's
    one-row-per-affected-service convention), and a mix of incidents: some
    real catches, some misses, and some false positives — deliberately
    including a tier with zero incidents at all, since that's the real 0/0
    precision case eval_harness has to resolve.
    """
    db_session.add(Service(id=1, name='auth-service'))
    db_session.add(Service(id=2, name='payment-gateway'))
    db_session.add(Service(id=3, name='user-profile'))

    db_session.add(GroundTruthAnomaly(
        service_id=1, metric_name='cpu_usage',
        ts_start=ts('2026-01-01T08:20:00'), ts_end=ts('2026-01-01T09:10:00'),
        difficulty_tier='obvious_spike'))
    db_session.add(GroundTruthAnomaly(
        service_id=2, metric_name='latency_ms',
        ts_start=ts('2026-01-02T17:40:00'), ts_end=ts('2026-01-03T05:40:00'),
        difficulty_tier='gradual_drift'))
    db_session.add(GroundTruthAnomaly(
        service_id=1, metric_name='cpu_usage',
        ts_start=ts('2026-01-04T11:20:00'), ts_end=ts('2026-01-04T13:20:00'),
        difficulty_tier='subtle_correlated'))
    db_session.add(GroundTruthAnomaly(
        service_id=3, metric_name='cpu_usage',
        ts_start=ts('2026-01-04T11:20:00'), ts_end=ts('2026-01-04T13:20:00'),
        difficulty_tier='subtle_correlated'))

    # detector_a: catches obvious_spike, misses gradual_drift entirely (no
    # incident anywhere near it), never touches subtle_correlated, and has one
    # unrelated false positive on a clean series.
    db_session.add(Incident(
        service_id=1, metric_name='cpu_usage',
        ts_start=ts('2026-01-01T08:20:00'), ts_end=ts('2026-01-01T09:10:00'),
        detector_source='detector_a', anomaly_score=5.0))
    db_session.add(Incident(
        service_id=3, metric_name='latency_ms',
        ts_start=ts('2026-01-10T00:00:00'), ts_end=ts('2026-01-10T00:05:00'),
        detector_source='detector_a', anomaly_score=3.1))

    # detector_b: catches all three tiers (both subtle_correlated rows), plus
    # one false positive.
    db_session.add(Incident(
        service_id=1, metric_name='cpu_usage',
        ts_start=ts('2026-01-01T08:15:00'), ts_end=ts('2026-01-01T09:15:00'),
        detector_source='detector_b', anomaly_score=9.0))
    db_session.add(Incident(
        service_id=2, metric_name='latency_ms',
        ts_start=ts('2026-01-02T18:00:00'), ts_end=ts('2026-01-03T00:00:00'),
        detector_source='detector_b', anomaly_score=4.0))
    db_session.add(Incident(
        service_id=1, metric_name='cpu_usage',
        ts_start=ts('2026-01-04T11:20:00'), ts_end=ts('2026-01-04T11:25:00'),
        detector_source='detector_b', anomaly_score=2.0))
    db_session.add(Incident(
        service_id=3, metric_name='cpu_usage',
        ts_start=ts('2026-01-04T11:20:00'), ts_end=ts('2026-01-04T11:25:00'),
        detector_source='detector_b', anomaly_score=2.0))
    db_session.add(Incident(
        service_id=2, metric_name='cpu_usage',
        ts_start=ts('2026-01-12T00:00:00'), ts_end=ts('2026-01-12T00:05:00'),
        detector_source='detector_b', anomaly_score=1.0))

    db_session.commit()
    return db_session
