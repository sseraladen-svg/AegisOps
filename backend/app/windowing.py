"""Multivariate windowing: turn the long-format metrics table into the
(n_windows, window_len, n_channels) tensor the LSTM autoencoder trains on.

A "channel" is one (service_id, metric_name) pair, so cpu/latency/error/req for
all three services are stacked side by side in a single window. That stacking
is the whole point: it's the only representation in this project where a model
can see that two services' CPU rose together, which is exactly what the
`subtle_correlated` tier is built to test and what the per-series naive and
Isolation Forest detectors are structurally unable to notice.

Channel order is taken from config.SERVICES x config.METRICS and is stable, so
a saved checkpoint stays meaningful across runs.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.config import METRICS, SERVICES
from app.models import Metric

WINDOW_LEN = 24     # 2 hours of context at 5-minute sampling
STRIDE = 1          # score every timestamp; cheap at this data size

# Fraction of the timeline reserved as the "healthy reference period" the
# autoencoder is fitted on. Real deployments pick a period an operator believes
# was clean rather than the whole history; see fit_region_overlaps_truth() for
# the honesty check that reports when that assumption doesn't hold here.
FIT_REGION_START_FRAC = 0.40
VALIDATION_FRAC = 0.20   # trailing share of the fit region, held out for the threshold


@dataclass(frozen=True)
class MatrixData:
    """Wide, ts-aligned view of every channel."""
    timestamps: np.ndarray          # (T,) datetime64[ns]
    values: np.ndarray              # (T, C) float64
    channels: list                  # [(service_id, metric_name), ...] length C

    @property
    def n_channels(self):
        return self.values.shape[1]


@dataclass(frozen=True)
class WindowSet:
    """Sliding windows over a MatrixData."""
    tensor: np.ndarray              # (N, WINDOW_LEN, C) float32
    starts: np.ndarray              # (N,) datetime64[ns], first ts in each window
    ends: np.ndarray                # (N,) datetime64[ns], last ts in each window
    channels: list

    def __len__(self):
        return self.tensor.shape[0]


def expected_channels():
    """Canonical channel order: service-major, metric-minor."""
    return [(svc_id, metric)
            for svc_id in range(1, len(SERVICES) + 1)
            for metric in METRICS]


def load_matrix(session):
    """Pivot the metrics table into a dense (T, C) matrix aligned on timestamp.

    Channels absent from the database are dropped (with a warning) rather than
    silently filled with zeros, which would teach the autoencoder that a
    flatlined service is normal.
    """
    rows = session.query(Metric.ts, Metric.service_id, Metric.metric_name, Metric.value).all()
    if not rows:
        raise RuntimeError("metrics table is empty — run `make seed` first")

    df = pd.DataFrame(rows, columns=['ts', 'service_id', 'metric_name', 'value'])
    df['ts'] = pd.to_datetime(df['ts'])
    wide = df.pivot_table(
        index='ts', columns=['service_id', 'metric_name'], values='value', aggfunc='mean',
    ).sort_index()

    channels = [c for c in expected_channels() if c in wide.columns]
    missing = [c for c in expected_channels() if c not in wide.columns]
    if missing:
        print(f"  WARNING: {len(missing)} configured channel(s) absent from the "
              f"database and excluded: {missing}")
    if not channels:
        raise RuntimeError(
            "none of the configured (service, metric) channels exist in the "
            "database; config.METRICS and seed.py have diverged")

    wide = wide[channels]

    # Gaps mean one service missed a scrape while others reported. Carrying the
    # last observation forward keeps the window grid regular; any leading NaN
    # (a channel with no history yet) has nothing to carry, so those rows go.
    wide = wide.ffill()
    before = len(wide)
    wide = wide.dropna()
    if len(wide) < before:
        print(f"  dropped {before - len(wide)} leading timestamp(s) with incomplete coverage")

    return MatrixData(
        timestamps=wide.index.to_numpy(),
        values=wide.to_numpy(dtype='float64'),
        channels=channels,
    )


def make_windows(matrix, window_len=WINDOW_LEN, stride=STRIDE):
    """Sliding windows over the full timeline, oldest first."""
    total, n_channels = matrix.values.shape
    if total < window_len:
        raise ValueError(
            f"need at least {window_len} samples to build a window, got {total}")

    # sliding_window_view gives a zero-copy (T-W+1, C, W) view; move the window
    # axis into the middle so each row reads (timesteps, channels).
    view = np.lib.stride_tricks.sliding_window_view(
        matrix.values, window_shape=window_len, axis=0)
    tensor = np.ascontiguousarray(view.transpose(0, 2, 1)[::stride], dtype='float32')

    start_idx = np.arange(0, total - window_len + 1, stride)
    return WindowSet(
        tensor=tensor,
        starts=matrix.timestamps[start_idx],
        ends=matrix.timestamps[start_idx + window_len - 1],
        channels=matrix.channels,
    )


def split_indices(n_windows,
                  fit_region_start_frac=FIT_REGION_START_FRAC,
                  validation_frac=VALIDATION_FRAC):
    """Chronological (train, validation) index ranges inside the fit region.

    Returns two ranges of window indices. Everything outside them is still
    scored — the split only governs what the model is fitted on and what the
    reconstruction-error threshold is calibrated against.
    """
    if n_windows <= 0:
        raise ValueError("no windows to split")
    region_start = int(n_windows * fit_region_start_frac)
    region = n_windows - region_start
    if region < 2:
        raise ValueError(
            f"fit region holds {region} window(s); lower FIT_REGION_START_FRAC "
            f"or seed a longer history")

    n_val = max(1, int(region * validation_frac))
    n_train = region - n_val
    if n_train < 1:
        n_train, n_val = region - 1, 1

    train = np.arange(region_start, region_start + n_train)
    val = np.arange(region_start + n_train, n_windows)
    return train, val


def fit_region_overlaps_truth(window_set, train_idx, val_idx, ground_truth):
    """Ground-truth windows that fall inside the fit region.

    The fit region is chosen positionally, not from labels, so it can overlap a
    real anomaly. When it does the autoencoder partly learns to reconstruct
    that anomaly and the validation percentile threshold is calibrated on dirty
    data — both of which flatter the reported numbers. This surfaces it instead
    of leaving it to be discovered later.
    """
    idx = np.concatenate([train_idx, val_idx])
    region_start = window_set.starts[idx].min()
    region_end = window_set.ends[idx].max()
    return [gt for gt in ground_truth
            if np.datetime64(gt.ts_start) <= region_end
            and np.datetime64(gt.ts_end) >= region_start]


class ChannelScaler:
    """Per-channel z-score, fitted on training windows only.

    Fitting on the full timeline would leak the anomalies' inflated variance
    into the normalisation and shrink exactly the deviations the model is meant
    to find.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, tensor):
        flat = tensor.reshape(-1, tensor.shape[-1])
        self.mean_ = flat.mean(axis=0)
        std = flat.std(axis=0)
        # A constant channel has zero spread; dividing by it yields inf/NaN, so
        # scale it by 1.0 and let it contribute a flat zero instead.
        self.std_ = np.where(std > 1e-8, std, 1.0)
        return self

    def transform(self, tensor):
        if self.mean_ is None:
            raise RuntimeError("ChannelScaler.transform called before fit")
        return ((tensor - self.mean_) / self.std_).astype('float32')

    def fit_transform(self, tensor):
        return self.fit(tensor).transform(tensor)

    def state_dict(self):
        return {'mean': self.mean_.tolist(), 'std': self.std_.tolist()}

    @classmethod
    def from_state_dict(cls, state):
        scaler = cls()
        scaler.mean_ = np.asarray(state['mean'], dtype='float64')
        scaler.std_ = np.asarray(state['std'], dtype='float64')
        return scaler
