"""Service simulator + anomaly injector.

Generates DAYS of multivariate telemetry for every (service, metric) pair in
config.METRICS, plus a correlated application log stream, then injects three
labelled anomalies of increasing subtlety.
Deterministic: fixed RNG seed and a fixed START_TIME anchor (not
datetime.now()) so `make seed` reproduces byte-identical values, which is what
lets the frozen eval numbers mean anything.

Run with:  python -m app.seed   (from backend/)
"""
import uuid
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy import insert

from app.config import DAYS, METRICS, SAMPLE_INTERVAL_MINUTES, SEED, SERVICES
from app.db import engine, session_scope
from app.models import Base, GroundTruthAnomaly, Log, Metric, Service

START_TIME = datetime(2026, 1, 1)  # fixed anchor so `make seed` is reproducible

# Per-service multipliers so the three services aren't three copies of the same
# series — the whole point of a cross-service model is that they differ.
SERVICE_SCALE = {1: 1.0, 2: 1.35, 3: 0.8}

# gradual_drift: linear ramp applied to latency_ms over the labeled window
DRIFT_MAGNITUDE = 40.0

# subtle_correlated: small correlated bump applied to the same metric on two
# services at once, since a single-series detector has no way to use the
# cross-service correlation that's supposed to make this tier distinct from
# a small single-series spike
SUBTLE_CORRELATED_SERVICES = [1, 3]  # auth-service, user-profile
SUBTLE_CORRELATED_METRIC = 'cpu_usage'
SUBTLE_CORRELATED_MAGNITUDE = 8.0

INSERT_CHUNK = 5000

# Log volume. Emitting a line per service per sample would be 12k rows of noise;
# one line per LOG_EVERY_N samples keeps the table small enough to page through
# in an agent tool call while still covering every anomaly window.
LOG_EVERY_N = 2                 # one baseline log line per 10 minutes per service
BASELINE_ERROR_CHANCE = 0.04    # share of baseline lines that are ERROR
ANOMALY_ERROR_CHANCE = 0.65     # ...and inside an injected anomaly window

# Message templates. The agent is asked to cite specific log evidence, so the
# text has to carry something citable — a subsystem and a failure mode — rather
# than a generic "something went wrong".
INFO_MESSAGES = [
    "handled request batch, {n} requests ok",
    "connection pool healthy ({n}/64 in use)",
    "cache hit ratio {n}%",
    "background reconciliation completed in {n}ms",
]
WARN_MESSAGES = [
    "connection pool at {n}/64 in use, above soft limit",
    "slow query detected ({n}ms) on table sessions",
    "retrying downstream call to billing-api, attempt {n}",
    "GC pause {n}ms exceeded target",
]
ERROR_MESSAGES = [
    "upstream timeout after {n}ms calling billing-api",
    "connection pool exhausted, {n} requests queued",
    "unhandled exception in request handler: TimeoutError after {n}ms",
    "failed to acquire database connection ({n}ms wait), aborting request",
]


def _timestamps():
    total_minutes = DAYS * 24 * 60
    return [START_TIME + timedelta(minutes=i)
            for i in range(0, total_minutes, SAMPLE_INTERVAL_MINUTES)]


def _simulate_service(rng, timestamps, scale):
    """One service's four channels, as a dict of metric_name -> np.ndarray.

    The channels are deliberately coupled (traffic drives cpu, cpu drives
    latency, latency drives errors) so there is real cross-channel structure
    for a multivariate model to learn and a univariate one to miss.
    """
    n = len(timestamps)
    # Fractional hour-of-day, so the diurnal term is smooth instead of stepping
    # once an hour (a step every 12 samples shows up as fake variance in the
    # rolling-window features the Isolation Forest consumes).
    hod = np.array([ts.hour + ts.minute / 60.0 for ts in timestamps])
    diurnal = np.sin(np.pi * hod / 12.0)          # -1 .. 1, 24h period
    traffic = 0.5 * (1.0 + diurnal)               # 0 .. 1

    request_rate = np.maximum(
        1.0, scale * (80.0 + 90.0 * traffic) + rng.normal(0, 5, n))
    load = (request_rate / (scale * 125.0)) - 1.0  # ~0-centred normalised load

    cpu = np.maximum(5.0, 30.0 + 10.0 * diurnal + 6.0 * load + rng.normal(0, 2, n))
    latency = np.maximum(10.0, 50.0 + 0.35 * (cpu - 30.0) + rng.normal(0, 5, n))
    error_rate = np.clip(
        0.4 + 0.02 * (latency - 50.0) + rng.normal(0, 0.15, n), 0.0, 100.0)

    return {
        'cpu_usage': cpu,
        'latency_ms': latency,
        'error_rate': error_rate,
        'request_rate': request_rate,
    }


def generate_baseline_data(session, timestamps):
    """Insert services and DAYS of clean telemetry. Returns the row count."""
    rng = np.random.default_rng(SEED)
    rows = []

    for svc_id, svc_name in enumerate(SERVICES, 1):
        session.add(Service(id=svc_id, name=svc_name))
        channels = _simulate_service(rng, timestamps, SERVICE_SCALE.get(svc_id, 1.0))

        for metric_name in METRICS:
            if metric_name not in channels:
                raise KeyError(
                    f"config.METRICS lists '{metric_name}' but the simulator "
                    f"emits {sorted(channels)} — they must agree.")
            values = channels[metric_name]
            rows.extend(
                {'service_id': svc_id, 'metric_name': metric_name,
                 'ts': ts, 'value': float(v)}
                for ts, v in zip(timestamps, values)
            )

    session.flush()  # services must exist before the metric FKs land
    for start in range(0, len(rows), INSERT_CHUNK):
        session.execute(insert(Metric), rows[start:start + INSERT_CHUNK])
    return len(rows)


def generate_logs(session, timestamps, anomaly_spans):
    """Emit a correlated log stream, with ERROR volume elevated inside anomalies.

    The investigation agent's whole job is to explain an incident from evidence,
    so the logs have to actually *contain* the explanation: inside an injected
    window a service's lines skew to timeouts and pool exhaustion, and outside
    it they're mostly routine. Without this the metrics tell a story the logs
    can't corroborate and every hypothesis the agent produces is unfalsifiable.
    """
    rng = np.random.default_rng(SEED + 1)
    rows = []

    for svc_id in range(1, len(SERVICES) + 1):
        spans = [(s, e) for sid, s, e in anomaly_spans if sid == svc_id]
        for ts in timestamps[::LOG_EVERY_N]:
            in_anomaly = any(start <= ts <= end for start, end in spans)
            error_chance = ANOMALY_ERROR_CHANCE if in_anomaly else BASELINE_ERROR_CHANCE
            draw = rng.random()

            if draw < error_chance:
                level, template = 'ERROR', ERROR_MESSAGES[rng.integers(len(ERROR_MESSAGES))]
            elif draw < error_chance + 0.15:
                level, template = 'WARN', WARN_MESSAGES[rng.integers(len(WARN_MESSAGES))]
            else:
                level, template = 'INFO', INFO_MESSAGES[rng.integers(len(INFO_MESSAGES))]

            magnitude = int(rng.integers(400, 3000) if level == 'ERROR'
                            else rng.integers(20, 400))
            rows.append({
                'service_id': svc_id,
                'ts': ts,
                'level': level,
                'message': template.format(n=magnitude),
                # Deterministic per-row id: seeded uuid4 would still vary, and a
                # reproducible seed that changes request ids on every run makes
                # a cited log line impossible to look up twice.
                'request_id': str(uuid.uuid5(uuid.NAMESPACE_OID, f"{svc_id}-{ts.isoformat()}")),
            })

    for start in range(0, len(rows), INSERT_CHUNK):
        session.execute(insert(Log), rows[start:start + INSERT_CHUNK])
    return len(rows)


def _shift_window(session, service_id, metric_name, ts_start, ts_end, delta):
    """Add a constant to every value of one series inside [ts_start, ts_end]."""
    session.query(Metric).filter(
        Metric.service_id == service_id,
        Metric.metric_name == metric_name,
        Metric.ts >= ts_start,
        Metric.ts <= ts_end,
    ).update({"value": Metric.value + delta}, synchronize_session=False)


def inject_anomalies(session, timestamps):
    """Inject the three difficulty tiers and label each one in ground truth.

    Returns the (service_id, ts_start, ts_end) spans so the log generator can
    make the log stream agree with the metrics.
    """
    if len(timestamps) <= 1024:
        raise ValueError(
            f"anomaly windows are indexed up to 1024 but only {len(timestamps)} "
            f"samples were generated; lower DAYS/SAMPLE_INTERVAL_MINUTES broke them.")

    spans = []

    # --- tier 1: obvious spike -------------------------------------------
    spike_start, spike_end = timestamps[100], timestamps[110]
    session.add(GroundTruthAnomaly(
        service_id=1, metric_name='cpu_usage',
        ts_start=spike_start, ts_end=spike_end, difficulty_tier='obvious_spike',
    ))
    _shift_window(session, 1, 'cpu_usage', spike_start, spike_end, 60.0)
    spans.append((1, spike_start, spike_end))

    # --- tier 2: gradual drift -------------------------------------------
    # Ramp linearly from 0 to DRIFT_MAGNITUDE across the window so the ground
    # truth label matches real injected signal at every point inside it.
    drift_start, drift_end = timestamps[500], timestamps[644]
    session.add(GroundTruthAnomaly(
        service_id=2, metric_name='latency_ms',
        ts_start=drift_start, ts_end=drift_end, difficulty_tier='gradual_drift',
    ))
    drift_rows = session.query(Metric).filter(
        Metric.service_id == 2, Metric.metric_name == 'latency_ms',
        Metric.ts >= drift_start, Metric.ts <= drift_end,
    ).order_by(Metric.ts).all()
    denom = max(len(drift_rows) - 1, 1)
    for i, row in enumerate(drift_rows):
        row.value += DRIFT_MAGNITUDE * (i / denom)
    spans.append((2, drift_start, drift_end))

    # --- tier 3: subtle correlated ---------------------------------------
    # A small bump on the same metric across two services in the same window,
    # so it's only obvious once you look across services.
    corr_start, corr_end = timestamps[1000], timestamps[1024]
    for svc_id in SUBTLE_CORRELATED_SERVICES:
        session.add(GroundTruthAnomaly(
            service_id=svc_id, metric_name=SUBTLE_CORRELATED_METRIC,
            ts_start=corr_start, ts_end=corr_end, difficulty_tier='subtle_correlated',
        ))
        _shift_window(session, svc_id, SUBTLE_CORRELATED_METRIC,
                      corr_start, corr_end, SUBTLE_CORRELATED_MAGNITUDE)
        spans.append((svc_id, corr_start, corr_end))

    return spans


def main():
    # Drop and recreate so a reseed never leaves stale rows or a stale schema.
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    timestamps = _timestamps()
    with session_scope() as session:
        print(f"generating baseline data ({len(SERVICES)} services x "
              f"{len(METRICS)} metrics x {len(timestamps)} samples)...")
        n_rows = generate_baseline_data(session, timestamps)
        print("injecting anomalies...")
        spans = inject_anomalies(session, timestamps)
        print("generating correlated logs...")
        n_logs = generate_logs(session, timestamps, spans)
    print(f"done. {n_rows} metric rows and {n_logs} log rows inserted")


if __name__ == "__main__":
    main()
