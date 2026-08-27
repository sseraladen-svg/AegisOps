from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Service(Base):
    __tablename__ = 'services'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)

class Metric(Base):
    __tablename__ = 'metrics'
    id = Column(Integer, primary_key=True)
    service_id = Column(Integer, ForeignKey('services.id'))
    metric_name = Column(String, nullable=False)
    ts = Column(DateTime, nullable=False, index=True)
    value = Column(Float, nullable=False)

class Log(Base):
    __tablename__ = 'logs'
    id = Column(Integer, primary_key=True)
    service_id = Column(Integer, ForeignKey('services.id'))
    ts = Column(DateTime, nullable=False, index=True)
    level = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    request_id = Column(String, index=True)

class GroundTruthAnomaly(Base):
    __tablename__ = 'ground_truth_anomalies'
    id = Column(Integer, primary_key=True)
    service_id = Column(Integer, ForeignKey('services.id'))
    metric_name = Column(String)
    ts_start = Column(DateTime, nullable=False)
    ts_end = Column(DateTime, nullable=False)
    difficulty_tier = Column(String, nullable=False) # obvious gradual or correlated

class Incident(Base):
    __tablename__ = 'incidents'
    id = Column(Integer, primary_key=True)
    service_id = Column(Integer, ForeignKey('services.id'))
    metric_name = Column(String)
    ts_start = Column(DateTime, nullable=False)
    ts_end = Column(DateTime, nullable=False)
    detector_source = Column(String, nullable=False) # which model caught it
    anomaly_score = Column(Float, nullable=False)
    status = Column(String, default='open')

class Investigation(Base):
    __tablename__ = 'investigations'
    id = Column(Integer, primary_key=True)
    incident_id = Column(Integer, ForeignKey('incidents.id'))
    tool_calls_json = Column(JSON)
    hypothesis = Column(Text)
    confidence = Column(Float)
    evidence_json = Column(JSON)
    github_issue_url = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)