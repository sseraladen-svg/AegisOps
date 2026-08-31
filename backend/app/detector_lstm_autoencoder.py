"""LSTM autoencoder over multivariate windows.

The model sees every (service, metric) channel at once — see windowing.py for
why that matters — compresses a 2-hour window to a single hidden vector, and
reconstructs it. Windows that reconstruct badly are the ones that don't look
like the healthy reference period.

Threshold selection is unsupervised: the cutoff is a high percentile of the
reconstruction error on a held-out *validation* slice of the reference period,
never on the labels. That's a deliberate contrast with the Isolation Forest,
whose contamination is tuned against ground truth; eval/results.json records
which detector used which so the comparison stays legible.

Attribution: a flagged window is charged to the channels actually responsible
for it, so incidents land on concrete (service_id, metric_name) rows the shared
eval harness can match. Channels are compared on *normalised excess* --- how far
a channel's error sits above its own validation median, in units of its own
validation spread --- because raw MSE isn't comparable across channels (a mostly
noise channel like error_rate reconstructs worse than request_rate even when
nothing is wrong). A channel is charged when it clears its own validation
percentile *and* carries at least ATTRIBUTION_SHARE of the window's peak excess;
without that second condition one real anomaly corrupts the shared hidden state,
every channel reconstructs badly at once, and a single incident sprays a false
positive across all twelve series. A window with no standout channel falls back
to its single worst, so an alarm is never silently dropped.

Run with:  python -m app.detector_lstm_autoencoder   (from backend/)
"""
import argparse
import json

import numpy as np
import torch
from torch import nn

from app.config import LSTM_CHECKPOINT, SEED, TUNING
from app.db import session_scope
from app.detector_utils import (
    clear_detector_incidents, load_ground_truth, replace_detector_incidents,
)
from app.windowing import (
    ChannelScaler, WINDOW_LEN, fit_region_overlaps_truth, load_matrix,
    make_windows, split_indices,
)

DETECTOR_SOURCE = 'lstm_autoencoder'
SELECTION_METHOD = TUNING[DETECTOR_SOURCE]

HIDDEN_SIZE = 32
NUM_LAYERS = 1
EPOCHS = 60
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
PATIENCE = 8               # epochs without val improvement before stopping

THRESHOLD_PERCENTILE = 99.5    # window-level cutoff, from validation errors
CHANNEL_PERCENTILE = 99.0      # per-channel cutoff used for attribution
ATTRIBUTION_SHARE = 0.5        # min share of a window's peak excess to be charged


class LSTMAutoencoder(nn.Module):
    """Encode a window to the encoder's final hidden state, then decode it back.

    The bottleneck is the hidden state alone — the decoder is handed that one
    vector repeated across the window, never the input timesteps — so the model
    can't cheat by copying its input forward.
    """

    def __init__(self, n_channels, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.LSTM(n_channels, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, n_channels)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        latent = hidden[-1]                                  # (B, H)
        repeated = latent.unsqueeze(1).repeat(1, x.size(1), 1)  # (B, W, H)
        decoded, _ = self.decoder(repeated)
        return self.output(decoded)


def set_determinism(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Single-threaded CPU matmul keeps epoch-to-epoch results bit-reproducible,
    # which is what lets the frozen comparison numbers be re-derived.
    torch.set_num_threads(1)


def train(model, train_tensor, val_tensor, epochs=EPOCHS, verbose=True):
    """Fit on train_tensor, early-stopping on val_tensor. Returns the best state."""
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    generator = torch.Generator().manual_seed(SEED)

    best_val = float('inf')
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_tensor), generator=generator)
        epoch_loss, n_batches = 0.0, 0
        for start in range(0, len(order), BATCH_SIZE):
            batch = train_tensor[order[start:start + BATCH_SIZE]]
            optimiser.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            # Long sequences through an LSTM can produce an exploding gradient
            # that turns the weights to NaN and silently ruins every later run.
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_tensor), val_tensor).item()

        if not np.isfinite(val_loss):
            raise RuntimeError(f"validation loss diverged to {val_loss} at epoch {epoch}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"  epoch {epoch:>3}/{epochs}  train={epoch_loss / max(n_batches, 1):.5f}  "
                  f"val={val_loss:.5f}")

        if epochs_without_improvement >= PATIENCE:
            if verbose:
                print(f"  early stop at epoch {epoch} (no val improvement for {PATIENCE})")
            break

    model.load_state_dict(best_state)
    return best_val


def reconstruction_errors(model, tensor, batch_size=256):
    """Per-window, per-channel mean squared error. Returns (N, C)."""
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(tensor), batch_size):
            batch = tensor[start:start + batch_size]
            squared = (model(batch) - batch) ** 2
            chunks.append(squared.mean(dim=1))   # average over timesteps
    return torch.cat(chunks).numpy() if chunks else np.empty((0, tensor.shape[-1]))


def channel_excess(errors, val_idx, percentile=CHANNEL_PERCENTILE):
    """Per-channel error rescaled so 0 == validation median and 1 == validation
    percentile. Makes noisy and quiet channels directly comparable."""
    reference = errors[val_idx]
    median = np.median(reference, axis=0)
    high = np.percentile(reference, percentile, axis=0)
    spread = np.maximum(high - median, 1e-9)
    return (errors - median) / spread


def attribute_window(excess_row, share=ATTRIBUTION_SHARE):
    """Channel indices to charge for one flagged window."""
    peak = excess_row.max()
    responsible = [c for c in np.flatnonzero(excess_row >= 1.0)
                   if excess_row[c] >= share * peak]
    return responsible or [int(np.argmax(excess_row))]


def merge_flagged_windows(indices, ends, scores):
    """Merge contiguous flagged window indices into incident spans.

    A window's timestamp is its *last* sample — the same convention the
    Isolation Forest uses for its trailing rolling features — so the two
    detectors' incident extents are directly comparable.
    """
    if len(indices) == 0:
        return []
    indices = np.sort(np.asarray(indices))
    breaks = np.where(np.diff(indices) > 1)[0] + 1
    windows = []
    for run in np.split(indices, breaks):
        peak = float(np.max(scores[run]))
        windows.append({
            "ts_start": _to_datetime(ends[run[0]]),
            "ts_end": _to_datetime(ends[run[-1]]),
            "anomaly_score": peak if np.isfinite(peak) else 0.0,
        })
    return windows


def _to_datetime(value):
    """numpy datetime64 -> python datetime, which is what SQLAlchemy binds."""
    return np.datetime64(value, 'us').astype('datetime64[us]').item()


def run(save_checkpoint=True, verbose=True):
    set_determinism()

    with session_scope() as session:
        matrix = load_matrix(session)
        ground_truth = load_ground_truth(session)

        window_set = make_windows(matrix)
        train_idx, val_idx = split_indices(len(window_set))
        print(f"windows: {len(window_set)} total "
              f"({len(window_set)} x {WINDOW_LEN} x {matrix.n_channels}), "
              f"train={len(train_idx)} val={len(val_idx)}")

        contaminated = fit_region_overlaps_truth(window_set, train_idx, val_idx, ground_truth)
        if contaminated:
            tiers = sorted({gt.difficulty_tier for gt in contaminated})
            print(f"  WARNING: the fit region overlaps {len(contaminated)} ground-truth "
                  f"anomal(ies) ({', '.join(tiers)}); reported numbers are optimistic")

        scaler = ChannelScaler().fit(window_set.tensor[train_idx])
        scaled = torch.from_numpy(scaler.transform(window_set.tensor))

        model = LSTMAutoencoder(matrix.n_channels)
        print("training LSTM autoencoder...")
        best_val = train(model, scaled[train_idx], scaled[val_idx], verbose=verbose)
        print(f"  best validation MSE: {best_val:.5f}")

        errors = reconstruction_errors(model, scaled)            # (N, C)
        window_errors = errors.mean(axis=1)                      # (N,)

        # Thresholds come from the validation slice only — no labels involved.
        val_errors = window_errors[val_idx]
        threshold = float(np.percentile(val_errors, THRESHOLD_PERCENTILE))
        excess = channel_excess(errors, val_idx)
        print(f"  threshold = p{THRESHOLD_PERCENTILE} of validation error = {threshold:.5f} "
              f"(validation median {np.median(val_errors):.5f})")

        flagged = np.flatnonzero(window_errors > threshold)
        print(f"  {len(flagged)} / {len(window_set)} window(s) over threshold")

        # Attribute each flagged window to the channels responsible for it.
        per_channel_indices = {channel: [] for channel in window_set.channels}
        for w in flagged:
            for c in attribute_window(excess[w]):
                per_channel_indices[window_set.channels[c]].append(w)

        # A channel that fired last run but not this one must have its stale
        # incidents removed too, so clear the detector wholesale first.
        clear_detector_incidents(session, DETECTOR_SOURCE)

        total_windows = 0
        for c, channel in enumerate(window_set.channels):
            service_id, metric_name = channel
            # Score the incident by normalised excess, not raw MSE, so a score
            # means the same thing on every series.
            windows = merge_flagged_windows(
                per_channel_indices[channel], window_set.ends, excess[:, c])
            replace_detector_incidents(
                session, DETECTOR_SOURCE, service_id, metric_name, windows)
            total_windows += len(windows)
            if windows:
                print(f"  lstm_autoencoder: service={service_id} metric={metric_name} "
                      f"-> {len(windows)} incident window(s)")

        summary = {
            "detector": DETECTOR_SOURCE,
            "window_len": WINDOW_LEN,
            "n_channels": matrix.n_channels,
            "channels": [list(c) for c in window_set.channels],
            "n_windows": int(len(window_set)),
            "n_train_windows": int(len(train_idx)),
            "n_val_windows": int(len(val_idx)),
            "threshold_percentile": THRESHOLD_PERCENTILE,
            "channel_percentile": CHANNEL_PERCENTILE,
            "attribution_share": ATTRIBUTION_SHARE,
            "threshold": threshold,
            "best_val_mse": best_val,
            "flagged_windows": int(len(flagged)),
            "incident_windows": int(total_windows),
            "selection_method": SELECTION_METHOD,
            "fit_region_contaminated": bool(contaminated),
        }

    if save_checkpoint:
        LSTM_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "scaler": scaler.state_dict(),
            "summary": summary,
        }, LSTM_CHECKPOINT)
        print(f"  checkpoint -> {LSTM_CHECKPOINT}")

    print(f"lstm_autoencoder detector: {total_windows} incident window(s) written")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-checkpoint', action='store_true',
                        help="train and write incidents but don't save the model")
    parser.add_argument('--quiet', action='store_true', help="suppress per-epoch logging")
    args = parser.parse_args()

    summary = run(save_checkpoint=not args.no_checkpoint, verbose=not args.quiet)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
