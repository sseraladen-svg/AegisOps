import { useCallback, useMemo } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import {
  DETECTORS, DETECTOR_LABELS, DETECTOR_SHORT, METRICS, METRIC_LABELS, TIER_LABELS, api,
} from '../api/client';
import { useApi } from '../api/useApi';
import { TimelineChart } from '../components/TimelineChart';
import { AsyncBoundary, Card, DetectorSwatch, IncidentLink } from '../components/ui';
import { formatDuration, formatNumber, formatTs } from '../components/format';

export default function Timeline() {
  const [params, setParams] = useSearchParams();
  const navigate = useNavigate();

  const serviceId = Number(params.get('service') ?? 1);
  const metric = METRICS.includes(params.get('metric')) ? params.get('metric') : 'cpu_usage';
  const activeDetectors = useMemo(() => {
    const raw = params.get('detectors');
    if (raw === null) return DETECTORS;
    // An explicit empty value means "none selected", which is different from
    // the parameter being absent.
    return raw === '' ? [] : raw.split(',').filter((d) => DETECTORS.includes(d));
  }, [params]);

  const setParam = (key, value) => {
    const next = new URLSearchParams(params);
    if (value === null || value === undefined) next.delete(key);
    else next.set(key, value);
    setParams(next, { replace: true });
  };

  const toggleDetector = (id) => {
    const next = activeDetectors.includes(id)
      ? activeDetectors.filter((d) => d !== id)
      : DETECTORS.filter((d) => activeDetectors.includes(d) || d === id);
    setParam('detectors', next.join(','));
  };

  const services = useApi(useCallback((opts) => api.services(opts), []), []);
  const series = useApi(
    useCallback(
      (opts) => api.metrics({ service_id: serviceId, metric_name: metric, max_points: 1500 }, opts),
      [serviceId, metric],
    ),
    [serviceId, metric],
  );
  const incidents = useApi(
    useCallback(
      (opts) => api.incidents({ service_id: serviceId, metric_name: metric, limit: 2000 }, opts),
      [serviceId, metric],
    ),
    [serviceId, metric],
  );
  const groundTruth = useApi(useCallback((opts) => api.groundTruth(opts), []), []);

  const incidentsByDetector = useMemo(() => {
    const grouped = Object.fromEntries(DETECTORS.map((d) => [d, []]));
    (incidents.data ?? []).forEach((inc) => grouped[inc.detector_source]?.push(inc));
    return grouped;
  }, [incidents.data]);

  const relevantGroundTruth = useMemo(
    () => (groundTruth.data ?? []).filter(
      (gt) => gt.service_id === serviceId && gt.metric_name === metric,
    ),
    [groundTruth.data, serviceId, metric],
  );

  const serviceName = services.data?.find((s) => s.id === serviceId)?.name ?? `service ${serviceId}`;

  return (
    <>
      <div className="page-head">
        <h1>Timeline</h1>
        <p className="lede">
          One metric over the full fourteen days, with each detector&apos;s flagged windows
          in its own lane below the plot. A gap in a lane where the shaded ground-truth
          band sits is a miss.
        </p>
      </div>

      <div className="filters">
        <div className="filter-group">
          <label className="filter-label" htmlFor="svc">Service</label>
          <select
            id="svc"
            value={serviceId}
            onChange={(e) => setParam('service', e.target.value)}
          >
            {(services.data ?? [{ id: serviceId, name: serviceName }]).map((s) => (
              <option key={s.id} value={s.id}>{s.name}</option>
            ))}
          </select>
        </div>

        <div className="filter-group">
          <span className="filter-label">Metric</span>
          <div className="segmented" role="group" aria-label="Metric">
            {METRICS.map((m) => (
              <button
                key={m}
                type="button"
                aria-pressed={metric === m}
                onClick={() => setParam('metric', m)}
              >
                {METRIC_LABELS[m]}
              </button>
            ))}
          </div>
        </div>

        <div className="filter-group">
          <span className="filter-label">Detectors</span>
          {DETECTORS.map((id) => (
            <button
              key={id}
              type="button"
              className="toggle-chip"
              aria-pressed={activeDetectors.includes(id)}
              onClick={() => toggleDetector(id)}
            >
              <DetectorSwatch detector={id} />
              {DETECTOR_SHORT[id]}
            </button>
          ))}
        </div>
      </div>

      <div className="stack">
        <Card
          title={`${serviceName} · ${METRIC_LABELS[metric]}`}
          subtitle={
            series.data
              ? `${series.data.total_available.toLocaleString()} samples${series.data.downsampled_by > 1 ? `, plotted every ${series.data.downsampled_by}${ordinal(series.data.downsampled_by)}` : ''}`
              : undefined
          }
        >
          <AsyncBoundary
            loading={series.loading || incidents.loading}
            refetching={series.refetching}
            error={series.error ?? incidents.error}
            data={series.data && incidents.data ? series.data : null}
            onRetry={() => { series.refresh(); incidents.refresh(); }}
            loadingRows={5}
          >
            {(data) => (
              <>
                <TimelineChart
                  series={data}
                  incidentsByDetector={incidentsByDetector}
                  detectors={activeDetectors}
                  groundTruth={relevantGroundTruth}
                  height={300}
                  onSelectIncident={(inc) => navigate(`/incidents/${inc.id}`)}
                />
                <div className="legend">
                  <span className="item">
                    <span className="swatch line" style={{ background: 'var(--text-primary)' }} aria-hidden="true" />
                    {METRIC_LABELS[metric]}
                  </span>
                  <span className="item">
                    <span
                      className="swatch"
                      style={{ background: 'var(--text-primary)', opacity: 0.18 }}
                      aria-hidden="true"
                    />
                    Injected anomaly (ground truth)
                  </span>
                  {activeDetectors.map((id) => (
                    <span className="item" key={id}>
                      <DetectorSwatch detector={id} />
                      {DETECTOR_LABELS[id]}
                    </span>
                  ))}
                  {activeDetectors.length === 0 && (
                    <span className="muted">No detector lanes shown — pick one above.</span>
                  )}
                </div>
                {relevantGroundTruth.length > 0 && (
                  <p className="muted" style={{ fontSize: '0.75rem', marginTop: '0.625rem' }}>
                    Ground truth here:{' '}
                    {relevantGroundTruth
                      .map((gt) => `${TIER_LABELS[gt.difficulty_tier] ?? gt.difficulty_tier} (${formatTs(gt.ts_start)} – ${formatTs(gt.ts_end)})`)
                      .join(', ')}
                  </p>
                )}
              </>
            )}
          </AsyncBoundary>
        </Card>

        <Card
          title="Incidents on this series"
          subtitle="The table view of the lanes above — every window, with its score."
        >
          <AsyncBoundary
            {...incidents}
            onRetry={incidents.refresh}
            empty={Object.assign(
              (rows) => !rows.filter((r) => activeDetectors.includes(r.detector_source)).length,
              {
                title: 'No incidents on this series',
                hint: 'Try another metric or service, or enable more detectors above.',
              },
            )}
          >
            {(rows) => (
              <div className="table-wrap">
                <table className="data">
                  <thead>
                    <tr>
                      <th>Detector</th>
                      <th>Start</th>
                      <th>End</th>
                      <th className="num">Duration</th>
                      <th className="num">Score</th>
                      <th>Report</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows
                      .filter((row) => activeDetectors.includes(row.detector_source))
                      .sort((a, b) => a.ts_start.localeCompare(b.ts_start))
                      .map((row) => (
                        <tr key={row.id}>
                          <td>
                            <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4375rem' }}>
                              <DetectorSwatch detector={row.detector_source} />
                              <IncidentLink id={row.id}>{DETECTOR_SHORT[row.detector_source]}</IncidentLink>
                            </span>
                          </td>
                          <td className="nowrap">{formatTs(row.ts_start)}</td>
                          <td className="nowrap">{formatTs(row.ts_end)}</td>
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
            )}
          </AsyncBoundary>
        </Card>
      </div>
    </>
  );
}

function ordinal(n) {
  if (n % 100 >= 11 && n % 100 <= 13) return 'th';
  return { 1: 'st', 2: 'nd', 3: 'rd' }[n % 10] ?? 'th';
}
