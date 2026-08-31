import numpy as np
import pytest

from app.models import GroundTruthAnomaly
from app.windowing import MatrixData, fit_region_overlaps_truth, make_windows, split_indices


def _matrix(n_rows=20, n_channels=2):
    timestamps = np.array(
        ['2026-01-01T00:00'], dtype='datetime64[m]') + np.arange(n_rows)
    values = np.arange(n_rows * n_channels, dtype='float64').reshape(n_rows, n_channels)
    return MatrixData(timestamps=timestamps, values=values,
                      channels=[(1, 'cpu_usage'), (1, 'latency_ms')][:n_channels])


def test_make_windows_shape_and_alignment():
    matrix = _matrix(n_rows=10, n_channels=2)
    window_set = make_windows(matrix, window_len=4, stride=1)

    assert len(window_set) == 10 - 4 + 1
    assert window_set.tensor.shape == (7, 4, 2)
    # Window i covers rows [i, i+window_len), so it starts/ends there.
    assert window_set.starts[0] == matrix.timestamps[0]
    assert window_set.ends[0] == matrix.timestamps[3]
    assert window_set.starts[-1] == matrix.timestamps[6]
    assert window_set.ends[-1] == matrix.timestamps[9]


def test_make_windows_raises_if_too_short():
    matrix = _matrix(n_rows=3, n_channels=1)
    with pytest.raises(ValueError):
        make_windows(matrix, window_len=10)


def test_split_indices_chronological_and_non_overlapping():
    train, val = split_indices(100, fit_region_start_frac=0.4, validation_frac=0.2)
    assert train[-1] < val[0]                    # chronological, no overlap
    assert train[0] == 40                         # region starts at 40% in
    assert val[-1] == 99                          # region runs to the end
    assert len(val) == max(1, int((100 - 40) * 0.2))


def test_split_indices_raises_when_region_too_small():
    with pytest.raises(ValueError):
        split_indices(2, fit_region_start_frac=0.99)


def _gt(service_id, metric_name, ts_start, ts_end, tier):
    return GroundTruthAnomaly(
        service_id=service_id, metric_name=metric_name,
        ts_start=ts_start, ts_end=ts_end, difficulty_tier=tier)


def test_fit_region_overlaps_truth_detects_real_overlap():
    matrix = _matrix(n_rows=10, n_channels=1)
    window_set = make_windows(matrix, window_len=4, stride=1)
    train_idx = np.array([0, 1])
    val_idx = np.array([2, 3])  # covers window ends up to timestamps[5]

    inside = _gt(1, 'cpu_usage', matrix.timestamps[0], matrix.timestamps[2], 'obvious_spike')
    outside = _gt(1, 'cpu_usage', matrix.timestamps[8], matrix.timestamps[9], 'gradual_drift')

    contaminated = fit_region_overlaps_truth(window_set, train_idx, val_idx, [inside, outside])
    assert contaminated == [inside]


def test_fit_region_overlaps_truth_empty_when_no_overlap():
    matrix = _matrix(n_rows=10, n_channels=1)
    window_set = make_windows(matrix, window_len=4, stride=1)
    train_idx, val_idx = np.array([0]), np.array([1])

    far_away = _gt(1, 'cpu_usage', matrix.timestamps[8], matrix.timestamps[9], 'gradual_drift')
    assert fit_region_overlaps_truth(window_set, train_idx, val_idx, [far_away]) == []
