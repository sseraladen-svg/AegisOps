"""Scores every detector's incidents against ground_truth_anomalies, broken
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

Comparability: detectors are not tuned the same way, and pretending otherwise
would make the frozen table misleading. Each detector's `tuning` field records
how its operating point was chosen — the Isolation Forest's contamination is
selected against the very labels reported here, while the naive detector uses
a fixed threshold and the LSTM autoencoder an unlabelled validation
percentile.

Run with:  python -m app.eval_harness   (from backend/)
"""
import json
from collections import Counter
from datetime import datetime, timezone

from app.config import DETECTORS, RESULTS_PATH, TIERS, TUNING
from app.db import get_session
from app.detector_utils import group_ground_truth, load_ground_truth, windows_overlap
from app.models import Incident

def load_incidents(session, detector_source):
    return session.query(Incident).filter(
        Incident.detector_source == detector_source).all()


def _f1(precision, recall):
    return 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)


def evaluate_detector(detector_source, incidents, ground_truth):
    gt_by_series = group_ground_truth(ground_truth)

    def matching_gts(incident):
        return [gt for gt in gt_by_series.get((incident.service_id, incident.metric_name), [])
                if windows_overlap(incident.ts_start, incident.ts_end, gt.ts_start, gt.ts_end)]

    # Resolve each incident's matches once — this used to be an O(incidents x
    # ground_truth) rescan repeated for every tier.
    matches_by_incident = {inc.id: matching_gts(inc) for inc in incidents}
    matched_gt_ids = {m.id for matches in matches_by_incident.values() for m in matches}

    # False positives are detector-wide (they don't belong to a tier).
    fp = sum(1 for inc in incidents if not matches_by_incident[inc.id])

    precision, recall, f1 = [], [], []
    debug_rows = []
    total_tp_gt = 0
    total_tp_incidents = 0

    for tier in TIERS:
        tier_gt = [gt for gt in ground_truth if gt.difficulty_tier == tier]
        tier_gt_ids = {gt.id for gt in tier_gt}

        tp_gt = sum(1 for gt in tier_gt if gt.id in matched_gt_ids)
        fn = len(tier_gt) - tp_gt
        recall_t = tp_gt / len(tier_gt) if tier_gt else 0.0

        tp_incidents = sum(
            1 for inc in incidents
            if any(m.id in tier_gt_ids for m in matches_by_incident[inc.id])
        )
        prec_denom = tp_incidents + fp
        precision_t = tp_incidents / prec_denom if prec_denom else 0.0
        if prec_denom == 0:
            print(f"  note: {detector_source}/{tier} precision resolved via the 0/0 "
                  f"convention (nothing fired in this tier and no unrelated false "
                  f"positives either)")

        precision.append(round(precision_t, 4))
        recall.append(round(recall_t, 4))
        f1.append(round(_f1(precision_t, recall_t), 4))
        debug_rows.append((tier, tp_incidents, fp, len(tier_gt), tp_gt, fn))
        total_tp_gt += tp_gt
        total_tp_incidents += tp_incidents

    print(f"  {detector_source} debug (tier, TP_incidents, FP_detector, GT_total, TP_gt, FN):")
    for row in debug_rows:
        print(f"    {row}")

    overall_recall = total_tp_gt / len(ground_truth) if ground_truth else 0.0
    overall_precision = (len(incidents) - fp) / len(incidents) if incidents else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "overall": {
            "precision": round(overall_precision, 4),
            "recall": round(overall_recall, 4),
            "f1": round(_f1(overall_precision, overall_recall), 4),
        },
        "incidents": len(incidents),
        "false_positives": fp,
        "tuning": TUNING.get(detector_source, "unknown"),
    }


def build_results(session, detectors=DETECTORS):
    ground_truth = load_ground_truth(session)
    if not ground_truth:
        raise RuntimeError("ground_truth_anomalies is empty — run `make seed` first")

    results = {}
    for detector_source in detectors:
        print(f"Evaluating {detector_source}:")
        incidents = load_incidents(session, detector_source)
        if not incidents:
            print(f"  WARNING: no incidents found for '{detector_source}'. Its row will "
                  f"be all zeros — run the detector before the harness.")
        results[detector_source] = evaluate_detector(detector_source, incidents, ground_truth)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec='seconds'),
        "tiers": TIERS,
        "ground_truth_counts": {
            tier: Counter(gt.difficulty_tier for gt in ground_truth).get(tier, 0)
            for tier in TIERS
        },
        "detectors": results,
    }


def write_results(data, path=RESULTS_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename so the frontend can never read a half-written file.
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)
    return path


def main():
    session = get_session()
    try:
        data = build_results(session)
    finally:
        session.close()
    write_results(data)
    print(f"wrote {RESULTS_PATH}")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
