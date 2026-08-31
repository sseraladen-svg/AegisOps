"""Shared plumbing for every detector: loading series, turning per-point flags
into incident windows, and writing those windows back scoped to one detector.
"""
from collections import defaultdict

import pandas as pd

from app.models import GroundTruthAnomaly, Incident, Metric

SERIES_COLUMNS = ['ts', 'value']


def list_series(session):
    """Distinct (service_id, metric_name) pairs actually present in metrics."""
    rows = session.query(Metric.service_id, Metric.metric_name).distinct().all()
    return sorted((int(sid), str(name)) for sid, name in rows)


def load_series(session, service_id, metric_name):
    """Full (ts, value) history for one series, sorted by ts.

    Always returns the declared dtypes: an empty result from pd.DataFrame([])
    would otherwise come back all-object and blow up the first .rolling() call
    with a confusing TypeError instead of an empty frame.
    """
    rows = (
        session.query(Metric.ts, Metric.value)
        .filter(Metric.service_id == service_id, Metric.metric_name == metric_name)
        .order_by(Metric.ts)
        .all()
    )
    df = pd.DataFrame(rows, columns=SERIES_COLUMNS)
    if df.empty:
        return pd.DataFrame({'ts': pd.Series(dtype='datetime64[ns]'),
                             'value': pd.Series(dtype='float64')})
    df['ts'] = pd.to_datetime(df['ts'])
    df['value'] = df['value'].astype('float64')
    return df.reset_index(drop=True)


def merge_flags_to_windows(df, flagged, score):
    """Merge contiguous flagged rows (evenly-spaced ts) into incident windows.

    df, flagged, and score must share df's index. Returns a list of
    {"ts_start", "ts_end", "anomaly_score"} dicts, one per contiguous run.
    """
    if df.empty:
        return []
    flagged = pd.Series(flagged, index=df.index).fillna(False).astype(bool)
    if not flagged.any():
        return []
    score = pd.Series(score, index=df.index).astype('float64')

    run_id = (flagged != flagged.shift(fill_value=False)).cumsum()
    windows = []
    for _, group in df[flagged].groupby(run_id[flagged], sort=True):
        peak = score.loc[group.index].max()
        windows.append({
            "ts_start": group['ts'].min(),
            "ts_end": group['ts'].max(),
            # An all-NaN run would serialise as a bare `NaN` token that no
            # strict JSON parser (the frontend's included) will accept.
            "anomaly_score": float(peak) if pd.notna(peak) else 0.0,
        })
    return windows


def load_ground_truth(session):
    """All ground_truth_anomalies rows. Read-only — never write here (Hard Rule 1)."""
    return session.query(GroundTruthAnomaly).all()


def group_ground_truth(ground_truth):
    """Index ground truth by (service_id, metric_name).

    Both the contamination search and the eval harness used to rescan the whole
    ground-truth list once per candidate window; this makes those lookups O(1).
    """
    by_series = defaultdict(list)
    for gt in ground_truth:
        by_series[(gt.service_id, gt.metric_name)].append(gt)
    return by_series


def windows_overlap(a_start, a_end, b_start, b_end):
    return a_start <= b_end and a_end >= b_start


def window_matches_any(window, gts):
    return any(windows_overlap(window["ts_start"], window["ts_end"],
                               gt.ts_start, gt.ts_end)
               for gt in gts)


def replace_detector_incidents(session, detector_source, service_id, metric_name, windows):
    """Delete this detector's prior incidents for one series, then insert fresh ones.

    Scoped to (detector_source, service_id, metric_name) so a rerun never
    touches another series' or another detector's rows.
    """
    session.query(Incident).filter(
        Incident.detector_source == detector_source,
        Incident.service_id == service_id,
        Incident.metric_name == metric_name,
    ).delete(synchronize_session=False)
    for w in windows:
        if w["ts_end"] < w["ts_start"]:
            raise ValueError(f"inverted incident window for {detector_source}: {w}")
        session.add(Incident(
            service_id=service_id,
            metric_name=metric_name,
            ts_start=w["ts_start"],
            ts_end=w["ts_end"],
            detector_source=detector_source,
            anomaly_score=w["anomaly_score"],
            status='open',
        ))


def clear_detector_incidents(session, detector_source):
    """Drop every incident from one detector, across all series.

    A detector whose series coverage can shrink between runs (the LSTM
    autoencoder attributes windows to whichever channels reconstruct worst, so
    a series flagged last run may produce nothing this run) can't rely on
    per-series replacement alone to clean up after itself.
    """
    session.query(Incident).filter(
        Incident.detector_source == detector_source,
    ).delete(synchronize_session=False)
