"""Naive z-score/EWMA anomaly detector.

No ML library — this is the floor the other detectors need to beat. For each
(service, metric) series, scores each point against a causal EWMA baseline
(mean/std computed only from strictly-prior points) and flags points whose
z-score exceeds a fixed threshold.
"""
from db import get_session
from detector_utils import list_series, load_series, merge_flags_to_windows, replace_detector_incidents

SPAN = 48          # 4 hours at 5-min sampling
MIN_PERIODS = 48   # no scoring during the first 4h warm-up
Z_THRESHOLD = 3.0

DETECTOR_SOURCE = 'naive'


def score_series(df):
    prior = df['value'].shift(1)
    baseline_mean = prior.ewm(span=SPAN, min_periods=MIN_PERIODS).mean()
    baseline_std = prior.ewm(span=SPAN, min_periods=MIN_PERIODS).std()
    z = (df['value'] - baseline_mean) / baseline_std
    return z


def main():
    session = get_session()
    total_windows = 0
    for service_id, metric_name in list_series(session):
        df = load_series(session, service_id, metric_name)
        z = score_series(df)
        flagged = z.abs() > Z_THRESHOLD
        windows = merge_flags_to_windows(df, flagged, z.abs())
        replace_detector_incidents(session, DETECTOR_SOURCE, service_id, metric_name, windows)
        total_windows += len(windows)
        print(f"  naive: service={service_id} metric={metric_name} -> {len(windows)} incident window(s)")
    session.commit()
    print(f"naive detector: {total_windows} incident window(s) written")


if __name__ == "__main__":
    main()
