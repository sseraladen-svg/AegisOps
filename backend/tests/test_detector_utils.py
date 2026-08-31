from datetime import datetime, timedelta

import pandas as pd

from app.detector_utils import merge_flags_to_windows, windows_overlap


def _series(n, start='2026-01-01T00:00:00', step_minutes=5):
    start = datetime.fromisoformat(start)
    return pd.DataFrame({'ts': [start + timedelta(minutes=step_minutes * i) for i in range(n)]})


def test_merge_flags_to_windows_groups_contiguous_runs():
    df = _series(10)
    flagged = pd.Series([False, False, True, True, True, False, False, True, False, False])
    score = pd.Series([0, 0, 1, 2, 3, 0, 0, 5, 0, 0], dtype='float64')

    windows = merge_flags_to_windows(df, flagged, score)

    assert len(windows) == 2
    assert windows[0]['ts_start'] == df['ts'][2]
    assert windows[0]['ts_end'] == df['ts'][4]
    assert windows[0]['anomaly_score'] == 3.0
    assert windows[1]['ts_start'] == windows[1]['ts_end'] == df['ts'][7]
    assert windows[1]['anomaly_score'] == 5.0


def test_merge_flags_to_windows_no_flags_returns_empty():
    df = _series(5)
    flagged = pd.Series([False] * 5)
    score = pd.Series([0.0] * 5)
    assert merge_flags_to_windows(df, flagged, score) == []


def test_merge_flags_to_windows_empty_df_returns_empty():
    df = pd.DataFrame({'ts': pd.Series(dtype='datetime64[ns]')})
    assert merge_flags_to_windows(df, pd.Series(dtype=bool), pd.Series(dtype='float64')) == []


def test_merge_flags_to_windows_handles_nan_scores_without_raising():
    # A run whose score is entirely NaN would otherwise serialise as a bare
    # `NaN` token that no strict JSON parser accepts (see the code comment).
    df = _series(3)
    flagged = pd.Series([True, True, False])
    score = pd.Series([float('nan'), float('nan'), 0.0])
    windows = merge_flags_to_windows(df, flagged, score)
    assert len(windows) == 1
    assert windows[0]['anomaly_score'] == 0.0


def test_windows_overlap_identical_ranges():
    a = datetime(2026, 1, 1, 8, 0)
    b = datetime(2026, 1, 1, 9, 0)
    assert windows_overlap(a, b, a, b) is True


def test_windows_overlap_touching_edges_counts_as_overlap():
    # ts_end == other's ts_start is an inclusive boundary, not a gap.
    t1, t2, t3 = datetime(2026, 1, 1, 8), datetime(2026, 1, 1, 9), datetime(2026, 1, 1, 10)
    assert windows_overlap(t1, t2, t2, t3) is True


def test_windows_overlap_disjoint_ranges():
    t1, t2 = datetime(2026, 1, 1, 8), datetime(2026, 1, 1, 9)
    t3, t4 = datetime(2026, 1, 1, 10), datetime(2026, 1, 1, 11)
    assert windows_overlap(t1, t2, t3, t4) is False


def test_windows_overlap_one_contains_the_other():
    outer = (datetime(2026, 1, 1, 0), datetime(2026, 1, 2, 0))
    inner = (datetime(2026, 1, 1, 8), datetime(2026, 1, 1, 9))
    assert windows_overlap(*outer, *inner) is True
