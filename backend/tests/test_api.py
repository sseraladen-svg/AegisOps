import json
from datetime import datetime

from app.models import GroundTruthAnomaly, Incident, Metric, Service


def test_health_reports_incident_and_investigation_counts(api_client, db_session):
    db_session.add(Service(id=1, name='auth-service'))
    db_session.add(Incident(
        service_id=1, metric_name='cpu_usage',
        ts_start=datetime(2026, 1, 1), ts_end=datetime(2026, 1, 1, 0, 5),
        detector_source='naive', anomaly_score=1.0))
    db_session.commit()

    response = api_client.get('/health')
    assert response.status_code == 200
    body = response.json()
    assert body['status'] == 'ok'
    assert body['incident_count'] == 1
    assert body['investigation_count'] == 0


def test_list_services_returns_ordered_rows(api_client, db_session):
    db_session.add(Service(id=2, name='payment-gateway'))
    db_session.add(Service(id=1, name='auth-service'))
    db_session.commit()

    response = api_client.get('/api/services')
    assert response.status_code == 200
    assert [s['id'] for s in response.json()] == [1, 2]


def test_get_metrics_downsamples_and_reports_stride(api_client, db_session):
    db_session.add(Service(id=1, name='auth-service'))
    for i in range(100):
        db_session.add(Metric(
            service_id=1, metric_name='cpu_usage',
            ts=datetime(2026, 1, 1, 0, i % 60), value=float(i)))
    db_session.commit()

    response = api_client.get(
        '/api/metrics', params={'service_id': 1, 'metric_name': 'cpu_usage', 'max_points': 10})
    assert response.status_code == 200
    body = response.json()
    assert body['total_available'] == 100
    assert body['downsampled_by'] == 10          # 100 // 10
    assert body['count'] == 10
    assert body['unit'] == 'percent'


def test_get_metrics_404_when_service_has_no_samples(api_client, db_session):
    db_session.add(Service(id=1, name='auth-service'))
    db_session.commit()

    response = api_client.get(
        '/api/metrics', params={'service_id': 1, 'metric_name': 'cpu_usage'})
    assert response.status_code == 404


def test_get_metrics_422_for_unknown_metric_name(api_client, db_session):
    db_session.add(Service(id=1, name='auth-service'))
    db_session.commit()

    response = api_client.get(
        '/api/metrics', params={'service_id': 1, 'metric_name': 'not_a_real_metric'})
    assert response.status_code == 422


def test_get_metrics_404_for_unknown_service(api_client, db_session):
    response = api_client.get(
        '/api/metrics', params={'service_id': 999, 'metric_name': 'cpu_usage'})
    assert response.status_code == 404


def test_list_incidents_filters_by_detector(api_client, db_session):
    db_session.add(Service(id=1, name='auth-service'))
    db_session.add(Incident(
        service_id=1, metric_name='cpu_usage',
        ts_start=datetime(2026, 1, 1), ts_end=datetime(2026, 1, 1, 0, 5),
        detector_source='naive', anomaly_score=1.0))
    db_session.add(Incident(
        service_id=1, metric_name='cpu_usage',
        ts_start=datetime(2026, 1, 2), ts_end=datetime(2026, 1, 2, 0, 5),
        detector_source='isolation_forest', anomaly_score=2.0))
    db_session.commit()

    response = api_client.get('/api/incidents', params={'detector': 'naive'})
    assert response.status_code == 200
    rows = response.json()
    assert len(rows) == 1
    assert rows[0]['detector_source'] == 'naive'


def test_list_incidents_422_for_unknown_detector(api_client, db_session):
    response = api_client.get('/api/incidents', params={'detector': 'not_a_real_detector'})
    assert response.status_code == 422


def test_get_incident_404_for_unknown_id(api_client, db_session):
    assert api_client.get('/api/incidents/999').status_code == 404


def test_get_investigation_404_before_investigated(api_client, db_session):
    db_session.add(Service(id=1, name='auth-service'))
    db_session.add(Incident(
        id=42, service_id=1, metric_name='cpu_usage',
        ts_start=datetime(2026, 1, 1), ts_end=datetime(2026, 1, 1, 0, 5),
        detector_source='naive', anomaly_score=1.0))
    db_session.commit()

    response = api_client.get('/api/incidents/42/investigation')
    assert response.status_code == 404
    assert 'app.investigator' in response.json()['detail']


def test_list_ground_truth(api_client, db_session):
    db_session.add(GroundTruthAnomaly(
        service_id=1, metric_name='cpu_usage',
        ts_start=datetime(2026, 1, 1, 8, 20), ts_end=datetime(2026, 1, 1, 9, 10),
        difficulty_tier='obvious_spike'))
    db_session.commit()

    response = api_client.get('/api/ground-truth')
    assert response.status_code == 200
    rows = response.json()
    assert len(rows) == 1
    assert rows[0]['difficulty_tier'] == 'obvious_spike'


def test_eval_results_404_when_file_missing(api_client, monkeypatch, tmp_path):
    monkeypatch.setattr('app.main.RESULTS_PATH', tmp_path / 'nonexistent.json')
    response = api_client.get('/api/eval/results')
    assert response.status_code == 404


def test_eval_results_returns_file_contents(api_client, monkeypatch, tmp_path):
    payload = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "tiers": ["obvious_spike", "gradual_drift", "subtle_correlated"],
        "ground_truth_counts": {"obvious_spike": 1, "gradual_drift": 1, "subtle_correlated": 2},
        "detectors": {
            "naive": {"precision": [1, 0, 0], "recall": [1, 0, 0], "f1": [1, 0, 0],
                      "overall": {"precision": 1, "recall": 0.25, "f1": 0.4},
                      "incidents": 1, "false_positives": 0, "tuning": "fixed threshold"},
            "isolation_forest": {"precision": [1, 1, 1], "recall": [1, 1, 1], "f1": [1, 1, 1],
                                 "overall": {"precision": 1, "recall": 1, "f1": 1},
                                 "incidents": 4, "false_positives": 0, "tuning": "tuned"},
            "lstm_autoencoder": {"precision": [0, 0, 0], "recall": [0, 0, 0], "f1": [0, 0, 0],
                                 "overall": {"precision": 0, "recall": 0, "f1": 0},
                                 "incidents": 0, "false_positives": 0, "tuning": "percentile"},
        },
    }
    results_path = tmp_path / 'results.json'
    results_path.write_text(json.dumps(payload))
    monkeypatch.setattr('app.main.RESULTS_PATH', results_path)

    response = api_client.get('/api/eval/results')
    assert response.status_code == 200
    assert response.json()['tiers'] == payload['tiers']


def test_demo_tour_503_without_ground_truth(api_client, db_session):
    response = api_client.get('/api/eval/demo-tour')
    assert response.status_code == 503
