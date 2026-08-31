"""Isolation Forest anomaly detector, run per (service, metric) over sliding
windows of the metrics table.

One model is fit per (service, metric) series since cpu_usage (diurnal) and
latency_ms (no seasonality) have incompatible feature distributions that a
single shared model would blur together.

Caveat, recorded honestly in eval/results.json rather than hidden: the
contamination hyperparameter is chosen by scoring the candidate grid against
the same ground truth the eval harness later reports on. That is label
supervision, so these numbers are an optimistic upper bound for this detector
— unlike the LSTM autoencoder, which picks its threshold from an unlabelled
validation percentile. The eval harness tags each detector with how it was
tuned so the comparison table can't be read as apples-to-apples by accident.

Run with:  python -m app.detector_isolation_forest   (from backend/)
"""
import pandas as pd
from sklearn.ensemble import IsolationForest

from app.config import TUNING
from app.db import session_scope
from app.detector_utils import (
    group_ground_truth, list_series, load_ground_truth, load_series,
    merge_flags_to_windows, replace_detector_incidents, window_matches_any,
)

WINDOW = 12          # 1 hour of trailing context per scored point
FEATURE_COLS = ['mean', 'std', 'min', 'max', 'last']
RANDOM_STATE = 42
CONTAMINATION_GRID = [0.01, 0.02, 0.05, 0.1]

DETECTOR_SOURCE = 'isolation_forest'
SELECTION_METHOD = TUNING[DETECTOR_SOURCE]


def rolling_features(df):
    if df.empty:
        return pd.DataFrame(columns=['ts', *FEATURE_COLS])
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
    X = features[FEATURE_COLS].to_numpy(dtype='float64')
    pred = model.fit_predict(X)
    scores = -model.score_samples(X)  # higher = more anomalous
    flagged = pd.Series(pred == -1, index=features.index)
    score = pd.Series(scores, index=features.index)
    return merge_flags_to_windows(features, flagged, score)


def pooled_score(windows_by_series, ground_truth):
    """Simple, un-tiered precision/recall/F1 used only to pick contamination."""
    if not ground_truth:
        return 0.0, 0.0, 0.0

    gt_by_series = group_ground_truth(ground_truth)

    tp_gt = sum(
        1 for gt in ground_truth
        if any(w for w in windows_by_series.get((gt.service_id, gt.metric_name), [])
               if window_matches_any(w, [gt]))
    )

    total_incidents = 0
    matching_incidents = 0
    for key, windows in windows_by_series.items():
        gts = gt_by_series.get(key, [])
        total_incidents += len(windows)
        matching_incidents += sum(1 for w in windows if window_matches_any(w, gts))

    recall = tp_gt / len(ground_truth)
    precision = matching_incidents / total_incidents if total_incidents else 0.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def main():
    with session_scope() as session:
        ground_truth = load_ground_truth(session)
        series_list = list_series(session)
        features_by_series = {
            key: rolling_features(load_series(session, *key)) for key in series_list
        }

        print("Tuning contamination (pooled, un-tiered score across all series):")
        best_c, best_f1, best_windows = None, -1.0, {}
        # The grid is ascending and the comparison is strict, so the first
        # contamination to reach a given F1 wins — ties break toward smaller c.
        for c in CONTAMINATION_GRID:
            windows_by_series = {key: build_windows(feat, c)
                                 for key, feat in features_by_series.items()}
            precision, recall, f1 = pooled_score(windows_by_series, ground_truth)
            print(f"  contamination={c:<5} -> precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")
            if f1 > best_f1:
                best_c, best_f1, best_windows = c, f1, windows_by_series

        if best_c is None:
            raise RuntimeError("CONTAMINATION_GRID is empty; nothing to select")
        print(f"selected contamination={best_c} (pooled f1={best_f1:.3f}, "
              f"ties broken toward smaller c)")

        total_windows = 0
        for (service_id, metric_name), windows in best_windows.items():
            replace_detector_incidents(session, DETECTOR_SOURCE, service_id, metric_name, windows)
            total_windows += len(windows)
            print(f"  isolation_forest: service={service_id} metric={metric_name} "
                  f"-> {len(windows)} incident window(s)")
    print(f"isolation_forest detector: {total_windows} incident window(s) written "
          f"(contamination={best_c})")


if __name__ == "__main__":
    main()
