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


def generate_baseline_data():
    # make 14 days of normal data
    start_time = datetime.now() - timedelta(days=DAYS)
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

    # gradual drift for latency
    drift_start = timestamps[500]
    drift_end = timestamps[644]

    session.add(GroundTruthAnomaly(
        service_id=2, metric_name='latency_ms',
        ts_start=drift_start, ts_end=drift_end, difficulty_tier='gradual_drift'
    ))

    # do gradual drift logic here

    # subtle correlated stuff goes here

    session.commit()


if __name__ == "__main__":
    print("generating baseline data...")
    ts_array = generate_baseline_data()
    print("injecting anomalies...")
    inject_anomalies(ts_array)
    print("done. 20k rows inserted")