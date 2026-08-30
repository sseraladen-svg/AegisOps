"""Isolation Forest anomaly detector, run per (service, metric) over sliding
windows of the metrics table.

One model is fit per (service, metric) series since cpu_usage (diurnal) and
latency_ms (no seasonality) have incompatible feature distributions that a
single shared model would blur together.
"""
import pandas as pd
from sklearn.ensemble import IsolationForest

from db import get_session
from detector_utils import (
    list_series, load_series, merge_flags_to_windows,
    replace_detector_incidents, load_ground_truth, windows_overlap,
)

WINDOW = 12          # 1 hour of trailing context per scored point
FEATURE_COLS = ['mean', 'std', 'min', 'max', 'last']
RANDOM_STATE = 42
CONTAMINATION_GRID = [0.01, 0.02, 0.05, 0.1]

DETECTOR_SOURCE = 'isolation_forest'


def rolling_features(df):
    roll = df['value'].rolling(window=WINDOW, min_periods=WINDOW)
    feat = pd.DataFrame({
        'ts': df['ts'],
        'mean': roll.mean(),
        'std': roll.std(),
        'min': roll.min(),
        'max': roll.max(),
        'last': df['value'],
    })
    return feat.dropna().reset_index(drop=True)


def build_windows(features, contamination):
    if len(features) < 2:
        return []
    model = IsolationForest(contamination=contamination, random_state=RANDOM_STATE)
    X = features[FEATURE_COLS].values
    pred = model.fit_predict(X)
    scores = -model.score_samples(X)  # higher = more anomalous
    flagged = pd.Series(pred == -1, index=features.index)
    score = pd.Series(scores, index=features.index)
    return merge_flags_to_windows(features, flagged, score)


def pooled_score(windows_by_series, ground_truth):
    """Simple, un-tiered precision/recall/F1 used only to pick contamination."""
    tp_gt = 0
    for gt in ground_truth:
        key = (gt.service_id, gt.metric_name)
        if any(windows_overlap(w["ts_start"], w["ts_end"], gt.ts_start, gt.ts_end)
               for w in windows_by_series.get(key, [])):
            tp_gt += 1

    total_incidents = 0
    matching_incidents = 0
    for key, windows in windows_by_series.items():
        gts = [g for g in ground_truth if (g.service_id, g.metric_name) == key]
        for w in windows:
            total_incidents += 1
            if any(windows_overlap(w["ts_start"], w["ts_end"], g.ts_start, g.ts_end) for g in gts):
                matching_incidents += 1

    recall = tp_gt / len(ground_truth) if ground_truth else 0.0
    precision = matching_incidents / total_incidents if total_incidents else 0.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def main():
    session = get_session()
    ground_truth = load_ground_truth(session)
    series_list = list_series(session)
    features_by_series = {key: rolling_features(load_series(session, *key)) for key in series_list}

    print("Tuning contamination (pooled, un-tiered score across all series):")
    best = None
    for c in CONTAMINATION_GRID:
        windows_by_series = {key: build_windows(feat, c) for key, feat in features_by_series.items()}
        precision, recall, f1 = pooled_score(windows_by_series, ground_truth)
        print(f"  contamination={c:<5} -> precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")
        if best is None or f1 > best[1] or (f1 == best[1] and c < best[0]):
            best = (c, f1, windows_by_series)

    chosen_c, chosen_f1, chosen_windows = best
    print(f"selected contamination={chosen_c} (pooled f1={chosen_f1:.3f}, ties broken toward smaller c)")

    total_windows = 0
    for (service_id, metric_name), windows in chosen_windows.items():
        replace_detector_incidents(session, DETECTOR_SOURCE, service_id, metric_name, windows)
        total_windows += len(windows)
        print(f"  isolation_forest: service={service_id} metric={metric_name} -> {len(windows)} incident window(s)")
    session.commit()
    print(f"isolation_forest detector: {total_windows} incident window(s) written (contamination={chosen_c})")


if __name__ == "__main__":
    main()
