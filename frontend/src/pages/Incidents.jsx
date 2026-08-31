import { useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import { DETECTORS, DETECTOR_SHORT, METRICS, METRIC_LABELS, api } from '../api/client';
import { useApi } from '../api/useApi';
import { AsyncBoundary, Card, DetectorSwatch, IncidentLink } from '../components/ui';
import { formatDuration, formatNumber, formatTs } from '../components/format';

export default function Incidents() {
  const [params, setParams] = useSearchParams();

  const detector = DETECTORS.includes(params.get('detector')) ? params.get('detector') : '';
  const serviceId = params.get('service') ?? '';
  const metric = METRICS.includes(params.get('metric')) ? params.get('metric') : '';
  const investigated = params.get('investigated') ?? '';
  const orderBy = params.get('order') === 'anomaly_score' ? 'anomaly_score' : 'ts_start';

  const setParam = (key, value) => {
    const next = new URLSearchParams(params);
    if (!value) next.delete(key);
    else next.set(key, value);
    setParams(next, { replace: true });
  };

  const services = useApi(useCallback((opts) => api.services(opts), []), []);
  const incidents = useApi(
    useCallback((opts) => api.incidents({
      detector: detector || undefined,
      service_id: serviceId || undefined,
      metric_name: metric || undefined,
      investigated: investigated === '' ? undefined : investigated === 'yes',
      order_by: orderBy,
      limit: 1000,
    }, opts), [detector, serviceId, metric, investigated, orderBy]),
    [detector, serviceId, metric, investigated, orderBy],
  );

  const names = Object.fromEntries((services.data ?? []).map((s) => [s.id, s.name]));

  return (
    <>
      <div className="page-head">
        <h1>Incidents</h1>
        <p className="lede">
          Every window a detector flagged. Most are false positives — there are four real
          anomalies in fourteen days — which is the point the evaluation page quantifies.
        </p>
      </div>

      <div className="filters">
        <div className="filter-group">
          <label className="filter-label" htmlFor="f-detector">Detector</label>
          <select id="f-detector" value={detector} onChange={(e) => setParam('detector', e.target.value)}>
            <option value="">All</option>
            {DETECTORS.map((d) => <option key={d} value={d}>{DETECTOR_SHORT[d]}</option>)}
          </select>
        </div>
        <div className="filter-group">
          <label className="filter-label" htmlFor="f-service">Service</label>
          <select id="f-service" value={serviceId} onChange={(e) => setParam('service', e.target.value)}>
            <option value="">All</option>
            {(services.data ?? []).map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}
          </select>
        </div>
        <div className="filter-group">
          <label className="filter-label" htmlFor="f-metric">Metric</label>
          <select id="f-metric" value={metric} onChange={(e) => setParam('metric', e.target.value)}>
            <option value="">All</option>
            {METRICS.map((m) => <option key={m} value={m}>{METRIC_LABELS[m]}</option>)}
          </select>
        </div>
        <div className="filter-group">
          <label className="filter-label" htmlFor="f-inv">Report</label>
          <select id="f-inv" value={investigated} onChange={(e) => setParam('investigated', e.target.value)}>
            <option value="">Any</option>
            <option value="yes">Investigated</option>
            <option value="no">Not investigated</option>
          </select>
        </div>
        <div className="filter-group">
          <label className="filter-label" htmlFor="f-order">Sort</label>
          <select id="f-order" value={orderBy} onChange={(e) => setParam('order', e.target.value)}>
            <option value="ts_start">Time</option>
            <option value="anomaly_score">Score (high first)</option>
          </select>
        </div>
      </div>

      <Card>
        <AsyncBoundary
          {...incidents}
          onRetry={incidents.refresh}
          loadingRows={6}
          empty={Object.assign((rows) => !rows.length, {
            title: 'No incidents match these filters',
            hint: 'Clear a filter, or run `make pipeline` if the database is empty.',
          })}
        >
          {(rows) => (
            <>
              <p className="muted" style={{ fontSize: '0.75rem', marginBottom: '0.75rem' }}>
                {rows.length.toLocaleString()} incident{rows.length === 1 ? '' : 's'}
                {rows.length === 1000 ? ' (capped at 1000)' : ''}
              </p>
              <div className="table-wrap">
                <table className="data">
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>Detector</th>
                      <th>Service</th>
                      <th>Metric</th>
                      <th>Start</th>
                      <th className="num">Duration</th>
                      <th className="num">Score</th>
                      <th>Report</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((row) => (
                      <tr key={row.id}>
                        <td><IncidentLink id={row.id} /></td>
                        <td>
                          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4375rem' }}>
                            <DetectorSwatch detector={row.detector_source} />
                            {DETECTOR_SHORT[row.detector_source]}
                          </span>
                        </td>
                        <td>{row.service_name ?? names[row.service_id] ?? row.service_id}</td>
                        <td>{METRIC_LABELS[row.metric_name] ?? row.metric_name}</td>
                        <td className="nowrap">{formatTs(row.ts_start)}</td>
                        <td className="num">{formatDuration(row.duration_minutes)}</td>
                        <td className="num">{formatNumber(row.anomaly_score, 3)}</td>
                        <td>
                          {row.has_investigation
                            ? <IncidentLink id={row.id}>View</IncidentLink>
                            : <span className="muted">—</span>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </AsyncBoundary>
      </Card>
    </>
  );
}
