import pandas as pd
from models import Metric, Incident, GroundTruthAnomaly


def list_series(session):
    """Distinct (service_id, metric_name) pairs actually present in metrics."""
    rows = session.query(Metric.service_id, Metric.metric_name).distinct().all()
    return sorted(rows)


def load_series(session, service_id, metric_name):
    """Full (ts, value) history for one series, sorted by ts."""
    rows = (
        session.query(Metric.ts, Metric.value)
        .filter(Metric.service_id == service_id, Metric.metric_name == metric_name)
        .order_by(Metric.ts)
        .all()
    )
    return pd.DataFrame(rows, columns=['ts', 'value'])


def merge_flags_to_windows(df, flagged, score):
    """Merge contiguous flagged rows (evenly-spaced ts) into incident windows.

    df, flagged, and score must share df's index. Returns a list of
    {"ts_start", "ts_end", "anomaly_score"} dicts, one per contiguous run.
    """
    flagged = flagged.fillna(False)
    if not flagged.any():
        return []
    run_id = (flagged != flagged.shift(fill_value=False)).cumsum()
    windows = []
    for _, group in df[flagged].groupby(run_id[flagged]):
        windows.append({
            "ts_start": group['ts'].min(),
            "ts_end": group['ts'].max(),
            "anomaly_score": float(score.loc[group.index].max()),
        })
    return windows


def load_ground_truth(session):
    """All ground_truth_anomalies rows. Read-only — never write here (Hard Rule 1)."""
    return session.query(GroundTruthAnomaly).all()


def windows_overlap(a_start, a_end, b_start, b_end):
    return a_start <= b_end and a_end >= b_start


def replace_detector_incidents(session, detector_source, service_id, metric_name, windows):
    """Delete this detector's prior incidents for one series, then insert fresh ones.

    Scoped to (detector_source, service_id, metric_name) so a rerun never
    touches another series' or another detector's rows.
    """
    session.query(Incident).filter(
        Incident.detector_source == detector_source,
        Incident.service_id == service_id,
        Incident.metric_name == metric_name,
    ).delete()
    for w in windows:
        session.add(Incident(
            service_id=service_id,
            metric_name=metric_name,
            ts_start=w["ts_start"],
            ts_end=w["ts_end"],
            detector_source=detector_source,
            anomaly_score=w["anomaly_score"],
            status='open',
        ))
