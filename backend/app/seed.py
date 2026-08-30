import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, Service, Metric, Log, GroundTruthAnomaly

# setup sqlite and drop everything so we start fresh
engine = create_engine('sqlite:///telemetry.db')
Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

SERVICES = ['auth-service', 'payment-gateway', 'user-profile']
METRICS = ['cpu_usage', 'memory_usage', 'latency_ms', 'error_rate']
DAYS = 14
MINUTES_INTERVAL = 5  # emit every 5 mins

SEED = 42
START_TIME = datetime(2026, 1, 1)  # fixed anchor (not datetime.now()) so make seed is reproducible

# gradual_drift: linear ramp applied to latency_ms over the labeled window
DRIFT_MAGNITUDE = 40.0

# subtle_correlated: small correlated bump applied to the same metric on two
# services at once, since a single-series detector has no way to use the
# cross-service correlation that's supposed to make this tier distinct from
# a small single-series spike
SUBTLE_CORRELATED_SERVICES = [1, 3]  # auth-service, user-profile
SUBTLE_CORRELATED_METRIC = 'cpu_usage'
SUBTLE_CORRELATED_MAGNITUDE = 8.0


def generate_baseline_data():
    # make 14 days of normal data
    np.random.seed(SEED)
    start_time = START_TIME
    timestamps = [start_time + timedelta(minutes=i) for i in range(0, DAYS * 24 * 60, MINUTES_INTERVAL)]

    metrics_data = []

    for svc_id, svc_name in enumerate(SERVICES, 1):
        session.add(Service(id=svc_id, name=svc_name))

        for ts in timestamps:
            # fake cpu data with some noise
            hour = ts.hour
            diurnal = np.sin(np.pi * (hour / 12.0)) * 10
            cpu = max(5, 30 + diurnal + np.random.normal(0, 2))

            # fake latency
            latency = max(10, 50 + np.random.normal(0, 5))

            metrics_data.append(Metric(service_id=svc_id, metric_name='cpu_usage', ts=ts, value=cpu))
            metrics_data.append(Metric(service_id=svc_id, metric_name='latency_ms', ts=ts, value=latency))
            # add memory and error rate here later to hit 20k rows

    session.bulk_save_objects(metrics_data)
    session.commit()
    return timestamps


def inject_anomalies(timestamps):
    # obvious cpu spike
    spike_start = timestamps[100]
    spike_end = timestamps[110]

    session.add(GroundTruthAnomaly(
        service_id=1, metric_name='cpu_usage',
        ts_start=spike_start, ts_end=spike_end, difficulty_tier='obvious_spike'
    ))

    # update db to show the spike
    session.query(Metric).filter(
        Metric.service_id == 1, Metric.metric_name == 'cpu_usage',
        Metric.ts >= spike_start, Metric.ts <= spike_end
    ).update({"value": Metric.value + 60.0})

    # gradual drift for latency: ramp linearly from 0 to DRIFT_MAGNITUDE across
    # the window so the ground truth label matches real injected signal
    drift_start = timestamps[500]
    drift_end = timestamps[644]

    session.add(GroundTruthAnomaly(
        service_id=2, metric_name='latency_ms',
        ts_start=drift_start, ts_end=drift_end, difficulty_tier='gradual_drift'
    ))

    drift_rows = session.query(Metric).filter(
        Metric.service_id == 2, Metric.metric_name == 'latency_ms',
        Metric.ts >= drift_start, Metric.ts <= drift_end
    ).order_by(Metric.ts).all()
    n_drift = len(drift_rows)
    for i, row in enumerate(drift_rows):
        row.value += DRIFT_MAGNITUDE * (i / max(n_drift - 1, 1))

    # subtle correlated: small bump on the same metric across two services in
    # the same window, so it's only obvious once you look across services
    corr_start = timestamps[1000]
    corr_end = timestamps[1024]

    for svc_id in SUBTLE_CORRELATED_SERVICES:
        session.add(GroundTruthAnomaly(
            service_id=svc_id, metric_name=SUBTLE_CORRELATED_METRIC,
            ts_start=corr_start, ts_end=corr_end, difficulty_tier='subtle_correlated'
        ))
        session.query(Metric).filter(
            Metric.service_id == svc_id, Metric.metric_name == SUBTLE_CORRELATED_METRIC,
            Metric.ts >= corr_start, Metric.ts <= corr_end
        ).update({"value": Metric.value + SUBTLE_CORRELATED_MAGNITUDE})

    session.commit()


if __name__ == "__main__":
    print("generating baseline data...")
    ts_array = generate_baseline_data()
    print("injecting anomalies...")
    inject_anomalies(ts_array)
    print("done. 20k rows inserted")