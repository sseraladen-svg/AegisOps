import { useCallback, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  DETECTORS, DETECTOR_LABELS, DETECTOR_SHORT, METRIC_LABELS, METRIC_UNITS, TIER_LABELS, api,
} from '../api/client';
import { useApi } from '../api/useApi';
import { Sparkline } from '../components/Sparkline';
import { AsyncBoundary, Card, DetectorSwatch, EmptyState, Stat, StatusPill } from '../components/ui';
import { formatNumber, formatTs } from '../components/format';

const LOOKBACKS = [
  { hours: 24, label: '24 hours' },
  { hours: 168, label: '7 days' },
  { hours: 336, label: 'All 14 days' },
];

export default function Overview() {
  const [detector, setDetector] = useState('lstm_autoencoder');
  const [lookback, setLookback] = useState(24);

  const health = useApi(
    useCallback(
      (opts) => api.servicesHealth({ detector, lookback_hours: lookback }, opts),
      [detector, lookback],
    ),
    [detector, lookback],
  );
  const tour = useApi(useCallback((opts) => api.demoTour(opts), []), []);

  return (
    <>
      <div className="page-head">
        <h1>Overview</h1>
        <p className="lede">
          Three services, four metrics each, sampled every five minutes for fourteen days.
          Status counts the incidents the selected detector raised inside the chosen window,
          which ends at the dataset&apos;s last sample — this is a fixed historical replay, so
          &ldquo;now&rdquo; is Jan 14, not today. Widen the window to fourteen days to reach
          the injected anomalies; they all sit in the first few days.
        </p>
      </div>

      {/* One filter row, above everything it scopes. */}
      <div className="filters">
        <div className="filter-group">
          <span className="filter-label">Detector</span>
          <div className="segmented" role="group" aria-label="Detector source">
            {DETECTORS.map((id) => (
              <button
                key={id}
                type="button"
                aria-pressed={detector === id}
                onClick={() => setDetector(id)}
              >
                {DETECTOR_SHORT[id]}
              </button>
            ))}
          </div>
        </div>
        <div className="filter-group">
          <label className="filter-label" htmlFor="lookback">Window</label>
          <select
            id="lookback"
            value={lookback}
            onChange={(e) => setLookback(Number(e.target.value))}
          >
            {LOOKBACKS.map((option) => (
              <option key={option.hours} value={option.hours}>{option.label}</option>
            ))}
          </select>
        </div>
        <span className="muted" style={{ fontSize: '0.75rem' }}>
          Counts and status pills are {DETECTOR_LABELS[detector]}&apos;s.
        </span>
      </div>

      <AsyncBoundary
        {...health}
        onRetry={health.refresh}
        loadingRows={4}
        empty={Object.assign((d) => !d.services?.length, {
          title: 'No services yet',
          hint: 'Run `make pipeline` to generate telemetry and detect anomalies.',
        })}
      >
        {(data) => (
          <div className="stack">
            <div className="grid tiles">
              {data.services.map((service) => (
                <ServiceTile key={service.service_id} service={service} detector={detector} />
              ))}
            </div>

            <p className="muted" style={{ fontSize: '0.75rem' }}>
              Window ends {formatTs(data.as_of)} · degraded at {data.thresholds.degraded_at}{' '}
              incident, critical at {data.thresholds.critical_at} · sparklines average each
              bucket, so they show the daily cycle rather than aliasing it.
            </p>

            <AsyncBoundary {...tour} onRetry={tour.refresh} loadingRows={3}>
              {(tourData) => <DemoTour stops={tourData.incidents} />}
            </AsyncBoundary>
          </div>
        )}
      </AsyncBoundary>
    </>
  );
}

function ServiceTile({ service, detector }) {
  const [metric, setMetric] = useState('cpu_usage');
  const summary = service.metrics.find((m) => m.metric_name === metric) ?? service.metrics[0];

  return (
    <Card
      title={service.name}
      actions={<StatusPill status={service.status} />}
    >
      <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', gap: '1rem' }}>
        <Stat
          label={`${METRIC_LABELS[summary?.metric_name] ?? '—'} (latest)`}
          value={
            summary?.current === null || summary?.current === undefined
              ? '—'
              : `${formatNumber(summary.current, 1)}${METRIC_UNITS[summary.metric_name] ?? ''}`
          }
          note={
            summary?.pct_change_vs_earlier === null || summary?.pct_change_vs_earlier === undefined
              ? 'no trend available'
              : `${summary.pct_change_vs_earlier >= 0 ? '+' : ''}${formatNumber(summary.pct_change_vs_earlier, 1)}% in the last quarter of the window`
          }
        />
        <Sparkline
          values={summary?.sparkline ?? []}
          label={`${service.name} ${summary?.metric_name ?? ''}`}
        />
      </div>

      <div className="segmented" style={{ marginTop: '0.875rem' }} role="group" aria-label={`Metric for ${service.name}`}>
        {service.metrics.map((m) => (
          <button
            key={m.metric_name}
            type="button"
            aria-pressed={metric === m.metric_name}
            onClick={() => setMetric(m.metric_name)}
          >
            {METRIC_LABELS[m.metric_name]}
          </button>
        ))}
      </div>

      <div
        style={{
          marginTop: '0.875rem', paddingTop: '0.75rem', borderTop: '1px solid var(--gridline)',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '0.75rem',
          flexWrap: 'wrap',
        }}
      >
        <span className="secondary" style={{ fontSize: '0.8125rem', display: 'inline-flex', alignItems: 'center', gap: '0.4375rem' }}>
          <DetectorSwatch detector={detector} />
          {service.incident_count} incident{service.incident_count === 1 ? '' : 's'} in window
        </span>
        <span style={{ display: 'inline-flex', gap: '0.75rem', fontSize: '0.8125rem' }}>
          <Link to={`/timeline?service=${service.service_id}`}>Timeline</Link>
          <Link to={`/incidents?service=${service.service_id}&detector=${detector}`}>Incidents</Link>
        </span>
      </div>
    </Card>
  );
}

function DemoTour({ stops }) {
  if (!stops?.length) {
    return (
      <Card title="Guided tour">
        <EmptyState
          title="No ground truth loaded"
          hint="Run `make seed` to inject the three labelled anomalies."
        />
      </Card>
    );
  }

  return (
    <Card
      title="The three anomalies, by difficulty"
      subtitle="Which detectors caught each one is computed from the incidents table, not asserted."
    >
      <div className="grid tiles">
        {stops.map((stop) => (
          <div key={stop.tier} style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            <div>
              <span className="tag">{TIER_LABELS[stop.tier] ?? stop.tier}</span>
              <h3 style={{ marginTop: '0.375rem' }}>{stop.label}</h3>
            </div>
            <p className="secondary" style={{ fontSize: '0.8125rem' }}>{stop.why_it_matters}</p>
            <p className="muted" style={{ fontSize: '0.75rem' }}>
              {stop.service_name} · {stop.metric_name} · {formatTs(stop.ts_start)}
            </p>

            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', alignItems: 'center' }}>
              {stop.caught_by.map((d) => (
                <span key={d} className="pill" title={`${DETECTOR_LABELS[d]} caught this`}>
                  <DetectorSwatch detector={d} />
                  <span aria-hidden="true">✓</span>
                  {DETECTOR_SHORT[d]}
                </span>
              ))}
              {stop.missed_by.map((d) => (
                <span key={d} className="pill" style={{ color: 'var(--text-muted)' }} title={`${DETECTOR_LABELS[d]} missed this`}>
                  <span className="swatch" style={{ background: 'var(--axis)' }} aria-hidden="true" />
                  <span aria-hidden="true">✗</span>
                  {DETECTOR_SHORT[d]}
                </span>
              ))}
            </div>

            {stop.incident_id && (
              <Link to={`/incidents/${stop.incident_id}`} style={{ fontSize: '0.8125rem' }}>
                Open incident #{stop.incident_id} →
              </Link>
            )}
          </div>
        ))}
      </div>
    </Card>
  );
}
