"""Tests for the matching/scoring convention CLAUDE.md calls the most
important property of the whole project: ground truth is injected before any
detector runs, and precision/recall/F1 have to be real, not asserted. These
tests exist to catch a regression in that logic specifically.
"""
from datetime import datetime

import pytest

from app.detector_utils import load_ground_truth
from app.eval_harness import evaluate_detector, load_incidents
from app.models import GroundTruthAnomaly, Incident


def _evaluate(session, detector_source):
    """evaluate_detector takes pre-loaded lists, not a session — matches
    build_results()'s call convention in app/eval_harness.py."""
    return evaluate_detector(
        detector_source, load_incidents(session, detector_source), load_ground_truth(session))


def test_evaluate_detector_a_catches_only_obvious_spike(seeded_session):
    result = _evaluate(seeded_session, 'detector_a')

    assert result['recall'] == [1.0, 0.0, 0.0]
    assert result['precision'] == [0.5, 0.0, 0.0]
    assert result['f1'] == [0.6667, 0.0, 0.0]
    assert result['incidents'] == 2
    assert result['false_positives'] == 1


def test_evaluate_detector_b_catches_every_tier(seeded_session):
    result = _evaluate(seeded_session, 'detector_b')

    assert result['recall'] == [1.0, 1.0, 1.0]
    assert result['precision'] == [0.5, 0.5, 0.6667]
    assert result['f1'] == [0.6667, 0.6667, 0.8]
    assert result['incidents'] == 5
    assert result['false_positives'] == 1


def test_false_positives_are_shared_across_every_tiers_precision(seeded_session):
    """The key convention: one detector-wide FP count, not per-tier scoping.

    A detector that only ever fires on service 3 (unrelated to any ground
    truth) should show the same false-positive count charged against every
    tier's precision denominator, not just the tier "closest" to that series.
    """
    seeded_session.add(Incident(
        service_id=3, metric_name='error_rate',
        ts_start=datetime(2026, 1, 6, 0, 0), ts_end=datetime(2026, 1, 6, 0, 5),
        detector_source='noisy_only', anomaly_score=1.0))
    seeded_session.commit()

    result = _evaluate(seeded_session, 'noisy_only')

    assert result['false_positives'] == 1
    assert result['recall'] == [0.0, 0.0, 0.0]
    # Zero true positives in every tier, but the same one false positive
    # shows up as the denominator for all three — not just one.
    assert result['precision'] == [0.0, 0.0, 0.0]


def test_zero_incidents_resolves_precision_to_zero_not_error(db_session, capsys):
    """A detector that never fired anything hits 0/0 (0 TP + 0 FP) in every
    tier. This must resolve to 0.0, not raise ZeroDivisionError, and must
    print the explicit warning so a silent 0.0 is never mistaken for "the
    detector had false positives here"."""
    db_session.add(GroundTruthAnomaly(
        service_id=1, metric_name='cpu_usage',
        ts_start=datetime(2026, 1, 1, 8, 20), ts_end=datetime(2026, 1, 1, 9, 10),
        difficulty_tier='obvious_spike'))
    db_session.commit()

    result = _evaluate(db_session, 'never_ran')

    assert result['precision'] == [0.0, 0.0, 0.0]
    assert result['recall'] == [0.0, 0.0, 0.0]
    assert result['f1'] == [0.0, 0.0, 0.0]
    assert result['incidents'] == 0
    assert result['false_positives'] == 0
    assert "0/0 convention" in capsys.readouterr().out


def test_unknown_detector_source_has_no_incidents(seeded_session):
    result = _evaluate(seeded_session, 'detector_that_never_ran')
    assert result['incidents'] == 0
    assert result['recall'] == [0.0, 0.0, 0.0]


@pytest.mark.parametrize('detector', ['detector_a', 'detector_b'])
def test_ground_truth_is_never_written_by_evaluate_detector(seeded_session, detector):
    """Hard Rule 1: ground_truth_anomalies is injector-only. A regression that
    made the eval harness write to it would compromise the whole evaluation."""
    before = seeded_session.query(GroundTruthAnomaly).count()
    _evaluate(seeded_session, detector)
    after = seeded_session.query(GroundTruthAnomaly).count()
    assert before == after
