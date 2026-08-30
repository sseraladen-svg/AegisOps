"""Scores each detector's incidents against ground_truth_anomalies, broken
out by difficulty_tier, and writes eval/results.json.

Matching rule: an incident matches a ground-truth row iff they share the
same (service_id, metric_name) and their time ranges overlap at all (no
minimum-overlap/IoU threshold — a short precise ground-truth window
shouldn't be penalized for a detection that's slightly early or late).

Precision convention: a false positive isn't "of" any tier (it doesn't
correspond to a real ground-truth row), so every tier's precision for a
given detector is charged that detector's *entire* false-positive count as a
shared denominator term, while the numerator is however many of that
detector's true positives landed in that specific tier. The alternative of
scoping FPs to "the tier's own series" would let a detector spraying false
alarms on an unrelated clean series look artificially precise in every tier.
"""
import json
from pathlib import Path

from db import get_session
from detector_utils import load_ground_truth, windows_overlap
from models import Incident

TIERS = ["obvious_spike", "gradual_drift", "subtle_correlated"]
RESULTS_PATH = Path(__file__).resolve().parents[2] / "eval" / "results.json"


def load_incidents(session, detector_source):
    return session.query(Incident).filter(Incident.detector_source == detector_source).all()


def evaluate_detector(session, detector_source):
    ground_truth = load_ground_truth(session)
    incidents = load_incidents(session, detector_source)

    def matches(incident, gt):
        return (incident.service_id, incident.metric_name) == (gt.service_id, gt.metric_name) \
            and windows_overlap(incident.ts_start, incident.ts_end, gt.ts_start, gt.ts_end)

    # False positives are detector-wide (they don't belong to a tier).
    fp = sum(1 for inc in incidents if not any(matches(inc, gt) for gt in ground_truth))

    precision, recall, f1 = [], [], []
    debug_rows = []
    for tier in TIERS:
        tier_gt = [gt for gt in ground_truth if gt.difficulty_tier == tier]

        tp_gt = sum(1 for gt in tier_gt if any(matches(inc, gt) for inc in incidents))
        fn = len(tier_gt) - tp_gt
        recall_t = tp_gt / len(tier_gt) if tier_gt else 0.0

        tp_incidents = sum(
            1 for inc in incidents
            if any(matches(inc, gt) for gt in tier_gt)
        )
        prec_denom = tp_incidents + fp
        precision_t = tp_incidents / prec_denom if prec_denom else 0.0
        if prec_denom == 0:
            print(f"  WARNING: {detector_source}/{tier} precision resolved via 0/0 convention "
                  f"(no incidents fired in this tier, and no unrelated false positives either)")

        f1_t = 0.0 if (precision_t + recall_t) == 0 else \
            2 * precision_t * recall_t / (precision_t + recall_t)

        precision.append(round(precision_t, 4))
        recall.append(round(recall_t, 4))
        f1.append(round(f1_t, 4))
        debug_rows.append((tier, tp_incidents, fp, len(tier_gt), tp_gt, fn))

    print(f"  {detector_source} debug (tier, TP_incidents, FP_detector, GT_total, TP_gt, FN):")
    for row in debug_rows:
        print(f"    {row}")

    return {"precision": precision, "recall": recall, "f1": f1}


def write_results(naive_stats, isolation_forest_stats):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "tiers": TIERS,
        "detectors": {
            "naive": naive_stats,
            "isolation_forest": isolation_forest_stats,
            "lstm_autoencoder": {"precision": [0, 0, 0], "recall": [0, 0, 0], "f1": [0, 0, 0]},
        },
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(data, f, indent=2)
    return data


def main():
    session = get_session()
    print("Evaluating naive:")
    naive_stats = evaluate_detector(session, 'naive')
    print("Evaluating isolation_forest:")
    isolation_forest_stats = evaluate_detector(session, 'isolation_forest')
    data = write_results(naive_stats, isolation_forest_stats)
    print(f"wrote {RESULTS_PATH}")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
