from datetime import datetime, timezone

from sqlalchemy import (
    Column, DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


def _utcnow():
    # datetime.utcnow() is deprecated in 3.12 and returns a naive value that
    # lies about its zone; this is explicit about being UTC.
    return datetime.now(timezone.utc)


class Service(Base):
    __tablename__ = 'services'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)


class Metric(Base):
    __tablename__ = 'metrics'
    id = Column(Integer, primary_key=True)
    service_id = Column(Integer, ForeignKey('services.id'), nullable=False)
    metric_name = Column(String, nullable=False)
    ts = Column(DateTime, nullable=False)
    value = Column(Float, nullable=False)

    # Every read path is "one series, ordered by ts" (detector_utils.load_series)
    # or "all series at one timestamp" (windowing.load_matrix). The composite
    # index serves the first as a covering scan and the standalone ts index the
    # second; without them each detector pass full-scans ~48k rows per series.
    __table_args__ = (
        Index('ix_metrics_series_ts', 'service_id', 'metric_name', 'ts'),
        Index('ix_metrics_ts', 'ts'),
    )


class Log(Base):
    __tablename__ = 'logs'
    id = Column(Integer, primary_key=True)
    service_id = Column(Integer, ForeignKey('services.id'), nullable=False)
    ts = Column(DateTime, nullable=False, index=True)
    level = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    request_id = Column(String, index=True)

    # query_logs always filters "one service, a time range, maybe a level".
    __table_args__ = (
        Index('ix_logs_service_ts', 'service_id', 'ts'),
    )


class GroundTruthAnomaly(Base):
    __tablename__ = 'ground_truth_anomalies'
    id = Column(Integer, primary_key=True)
    service_id = Column(Integer, ForeignKey('services.id'), nullable=False)
    metric_name = Column(String, nullable=False)
    ts_start = Column(DateTime, nullable=False)
    ts_end = Column(DateTime, nullable=False)
    difficulty_tier = Column(String, nullable=False)  # obvious gradual or correlated


class Incident(Base):
    __tablename__ = 'incidents'
    id = Column(Integer, primary_key=True)
    service_id = Column(Integer, ForeignKey('services.id'), nullable=False)
    metric_name = Column(String, nullable=False)
    ts_start = Column(DateTime, nullable=False)
    ts_end = Column(DateTime, nullable=False)
    detector_source = Column(String, nullable=False)  # which model caught it
    anomaly_score = Column(Float, nullable=False)
    status = Column(String, nullable=False, default='open')

    # replace_detector_incidents() deletes by exactly this triple on every rerun.
    __table_args__ = (
        Index('ix_incidents_detector_series', 'detector_source', 'service_id', 'metric_name'),
    )


class Investigation(Base):
    __tablename__ = 'investigations'
    id = Column(Integer, primary_key=True)
    incident_id = Column(Integer, ForeignKey('incidents.id'), nullable=False)
    tool_calls_json = Column(JSON)
    hypothesis = Column(Text)
    confidence = Column(Float)
    evidence_json = Column(JSON)
    github_issue_url = Column(String)
    created_at = Column(DateTime, default=_utcnow)
