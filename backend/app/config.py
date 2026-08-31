"""Single source of truth for paths and constants shared across modules.

Everything here used to be duplicated (or, worse, silently disagreed) between
seed.py, db.py and eval_harness.py. Paths are anchored to this file's location
rather than the process CWD so `python -m app.seed` and `uvicorn app.main:app`
resolve the same database no matter where they're launched from.
"""
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent

DB_PATH = BACKEND_DIR / "telemetry.db"
DB_URL = f"sqlite:///{DB_PATH}"

RESULTS_PATH = REPO_ROOT / "eval" / "results.json"
DATA_DIR = BACKEND_DIR / "data"
PAST_INCIDENTS_PATH = DATA_DIR / "past_incidents.json"
ARTIFACT_DIR = BACKEND_DIR / "artifacts"
LSTM_CHECKPOINT = ARTIFACT_DIR / "lstm_autoencoder.pt"
FAISS_INDEX_PATH = ARTIFACT_DIR / "past_incidents.faiss"
FAISS_VECTORIZER_PATH = ARTIFACT_DIR / "past_incidents_vectorizer.joblib"

SERVICES = ['auth-service', 'payment-gateway', 'user-profile']

# The multivariate feature set. Order is load-bearing: it fixes the channel
# order of the (window, service, metric) tensors the LSTM autoencoder trains
# on, and a checkpoint is only valid for the order it was trained with.
METRICS = ['cpu_usage', 'latency_ms', 'error_rate', 'request_rate']

TIERS = ["obvious_spike", "gradual_drift", "subtle_correlated"]

DETECTORS = ["naive", "isolation_forest", "lstm_autoencoder"]

# How each detector's operating point is chosen. Recorded verbatim in
# eval/results.json because the three are *not* tuned the same way, and a
# comparison table that hides that is misleading: the Isolation Forest's
# contamination is picked against the same labels the harness reports on,
# so its row is an optimistic upper bound.
TUNING = {
    'naive': 'fixed z-score threshold, no tuning against labels',
    'isolation_forest': 'contamination grid tuned against ground-truth labels',
    'lstm_autoencoder': 'reconstruction-error percentile on an unlabelled validation split',
}

SAMPLE_INTERVAL_MINUTES = 5
DAYS = 14
SEED = 42

# --- agent / investigation loop -------------------------------------------

AGENT_MODEL = "claude-opus-5"
AGENT_MAX_TOKENS = 16000
AGENT_MAX_ITERATIONS = 12   # hard stop on the tool-use loop
