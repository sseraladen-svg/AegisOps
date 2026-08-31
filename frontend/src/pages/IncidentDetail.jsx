import { useCallback } from 'react';
import { Link, useParams } from 'react-router-dom';
import {
  DETECTOR_LABELS, METRIC_LABELS, METRIC_UNITS, api,
} from '../api/client';
import { useApi } from '../api/useApi';
import { TimelineChart } from '../components/TimelineChart';
import { AsyncBoundary, Card, DetectorSwatch, EmptyState, ErrorState, Loading, StatusPill } from '../components/ui';
import { formatDuration, formatNumber, formatTs, toApiTs } from '../components/format';

const TOOL_LABELS = {
  query_metrics: 'query_metrics',
  query_logs: 'query_logs',
  query_similar_incidents: 'query_similar_incidents',
  file_github_issue: 'file_github_issue',
};

const SEVERITY_STATUS = { low: 'healthy', medium: 'degraded', high: 'critical' };

/** Pad an incident window out to something a chart can show context in. */
function contextWindow(incident, minutes = 240) {
  const start = new Date(new Date(incident.ts_start).getTime() - minutes * 60_000);
  const end = new Date(new Date(incident.ts_end).getTime() + minutes * 60_000);
  // toApiTs, not toISOString: see its comment — the API speaks naive timestamps.
  return { ts_start: toApiTs(start), ts_end: toApiTs(end) };
}

export default function IncidentDetail() {
  const { incidentId } = useParams();

  const incident = useApi(
    useCallback((opts) => api.incident(incidentId, opts), [incidentId]),
    [incidentId],
  );
  const investigation = useApi(
    useCallback((opts) => api.investigation(incidentId, opts), [incidentId]),
    [incidentId],
  );

  const window = incident.data ? contextWindow(incident.data) : null;

  const series = useApi(
    useCallback((opts) => api.metrics({
      service_id: incident.data.service_id,
      metric_name: incident.data.metric_name,
      ts_start: window.ts_start,
      ts_end: window.ts_end,
      max_points: 800,
    }, opts), [incident.data, window]),
    [incident.data?.id],
    { skip: !incident.data },
  );

  const nearbyIncidents = useApi(
    useCallback((opts) => api.incidents({
      service_id: incident.data.service_id,
      metric_name: incident.data.metric_name,
      ts_start: window.ts_start,
      ts_end: window.ts_end,
      limit: 500,
    }, opts), [incident.data, window]),
    [incident.data?.id],
    { skip: !incident.data },
  );

  const logs = useApi(
    useCallback((opts) => api.logs({
      service_id: incident.data.service_id,
      ts_start: incident.data.ts_start,
      ts_end: incident.data.ts_end,
      limit: 50,
    }, opts), [incident.data]),
    [incident.data?.id],
    { skip: !incident.data },
  );

  if (incident.loading) return <Loading rows={5} />;
  if (incident.error) return <ErrorState error={incident.error} onRetry={incident.refresh} />;
  if (!incident.data) return null;

  const data = incident.data;
  const byDetector = {};
  (nearbyIncidents.data ?? []).forEach((inc) => {
    (byDetector[inc.detector_source] ??= []).push(inc);
  });
  const detectorsPresent = Object.keys(byDetector);

  return (
    <>
      <div className="page-head">
        <p className="muted" style={{ fontSize: '0.8125rem' }}>
          <Link to="/incidents">Incidents</Link> / #{data.id}
        </p>
        <h1 style={{ marginTop: '0.25rem' }}>
          {data.service_name} · {METRIC_LABELS[data.metric_name] ?? data.metric_name}
        </h1>
        <p className="lede">
          Flagged by {DETECTOR_LABELS[data.detector_source]} from {formatTs(data.ts_start)} to{' '}
          {formatTs(data.ts_end)} ({formatDuration(data.duration_minutes)}).
        </p>
      </div>

      <div className="stack">
        <Card
          title="What the detector saw"
          subtitle="The flagged window with four hours of context either side."
        >
          <AsyncBoundary
            {...series}
            onRetry={series.refresh}
            loadingRows={4}
          >
            {(seriesData) => (
              <>
                <TimelineChart
                  series={seriesData}
                  incidentsByDetector={byDetector}
                  detectors={detectorsPresent}
                  groundTruth={[]}
                  height={250}
                />
                <div className="legend">
                  <span className="item">
                    <span className="swatch line" style={{ background: 'var(--text-primary)' }} aria-hidden="true" />
                    {METRIC_LABELS[data.metric_name]} ({METRIC_UNITS[data.metric_name]})
                  </span>
                  {detectorsPresent.map((d) => (
                    <span className="item" key={d}>
                      <DetectorSwatch detector={d} />
                      {DETECTOR_LABELS[d]}
                    </span>
                  ))}
                </div>
              </>
            )}
          </AsyncBoundary>

          <dl className="meta-list" style={{ marginTop: '1rem' }}>
            <dt>Anomaly score</dt><dd>{formatNumber(data.anomaly_score, 4)}</dd>
            <dt>Status</dt><dd>{data.status}</dd>
            <dt>Window</dt>
            <dd>{formatTs(data.ts_start, { withSeconds: true })} → {formatTs(data.ts_end, { withSeconds: true })}</dd>
          </dl>
        </Card>

        <InvestigationPanel investigation={investigation} incidentId={data.id} />

        <Card
          title="Logs during the window"
          subtitle="The same rows the agent's query_logs tool reads."
        >
          <AsyncBoundary
            {...logs}
            onRetry={logs.refresh}
            empty={Object.assign((d) => !d.logs?.length, {
              title: 'No log lines in this window',
              hint: 'Log lines are emitted every ten minutes per service, so a short window may contain none.',
            })}
          >
            {(logData) => (
              <>
                <p className="muted" style={{ fontSize: '0.75rem', marginBottom: '0.75rem' }}>
                  {Object.entries(logData.level_counts)
                    .map(([level, count]) => `${count} ${level}`)
                    .join(' · ')} in this window
                </p>
                <div className="table-wrap">
                  <table className="data">
                    <thead>
                      <tr><th>Time</th><th>Level</th><th>Message</th></tr>
                    </thead>
                    <tbody>
                      {logData.logs.map((line) => (
                        <tr key={line.id}>
                          <td className="nowrap">{formatTs(line.ts, { withSeconds: true })}</td>
                          <td>
                            <span
                              className="tag"
                              style={line.level === 'ERROR'
                                ? { color: 'var(--critical)' }
                                : line.level === 'WARN' ? { color: 'var(--text-primary)' } : undefined}
                            >
                              {line.level}
                            </span>
                          </td>
                          <td style={{ whiteSpace: 'normal' }}>{line.message}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </AsyncBoundary>
        </Card>
      </div>
    </>
  );
}

function InvestigationPanel({ investigation, incidentId }) {
  if (investigation.loading) {
    return <Card title="Agent investigation"><Loading rows={4} /></Card>;
  }

  // A 404 here is the expected state for an un-investigated incident, not a
  // failure — the backend's own message says exactly how to produce one.
  if (investigation.error?.status === 404) {
    return (
      <Card title="Agent investigation">
        <EmptyState
          title="Not investigated yet"
          hint={
            <>
              Run <code>python -m app.investigator --incident-id {incidentId}</code> in{' '}
              <code>backend/</code>, or <code>make pipeline</code> to investigate a sample.
            </>
          }
        />
      </Card>
    );
  }

  if (investigation.error) {
    return (
      <Card title="Agent investigation">
        <ErrorState error={investigation.error} onRetry={investigation.refresh} />
      </Card>
    );
  }

  const report = investigation.data;
  if (!report) return null;

  return (
    <div className="grid halves">
      <Card
        title="Agent report"
        actions={
          report.severity
            ? <StatusPill status={SEVERITY_STATUS[report.severity] ?? 'unknown'}>
                {report.severity} severity
              </StatusPill>
            : null
        }
      >
        {report.validation_warnings?.length > 0 && (
          <div className="callout danger" style={{ marginBottom: '1rem' }}>
            <div className="callout-title">
              {report.validation_warnings.length} claim{report.validation_warnings.length === 1 ? '' : 's'} this
              report&apos;s own trace doesn&apos;t support
            </div>
            <ul style={{ margin: '0.375rem 0 0', paddingLeft: '1.1rem' }}>
              {report.validation_warnings.map((warning, i) => <li key={i}>{warning}</li>)}
            </ul>
          </div>
        )}

        <h3>Hypothesis</h3>
        <p className="secondary" style={{ marginTop: '0.3125rem' }}>{report.hypothesis ?? '—'}</p>

        <div style={{ display: 'flex', gap: '2rem', marginTop: '1rem', flexWrap: 'wrap' }}>
          <div className="stat">
            <span className="label">Confidence</span>
            <span className="value">
              {report.confidence === null || report.confidence === undefined
                ? '—' : report.confidence.toFixed(2)}
            </span>
            <span className="note">self-reported, then checked against the trace</span>
          </div>
          <div className="stat">
            <span className="label">Evidence cited</span>
            <span className="value">{report.evidence?.length ?? 0}</span>
            <span className="note">{(report.tools_used ?? []).length} tool{(report.tools_used ?? []).length === 1 ? '' : 's'} used</span>
          </div>
        </div>

        {report.recommended_action && (
          <>
            <h3 style={{ marginTop: '1.25rem' }}>Recommended action</h3>
            <p className="secondary" style={{ marginTop: '0.3125rem' }}>{report.recommended_action}</p>
          </>
        )}

        {report.ruled_out?.length > 0 && (
          <>
            <h3 style={{ marginTop: '1.25rem' }}>Ruled out</h3>
            <ul className="secondary" style={{ margin: '0.3125rem 0 0', paddingLeft: '1.1rem', fontSize: '0.8125rem' }}>
              {report.ruled_out.map((item, i) => <li key={i}>{item}</li>)}
            </ul>
          </>
        )}

        <div style={{ marginTop: '1.25rem', paddingTop: '0.875rem', borderTop: '1px solid var(--gridline)' }}>
          {report.github_issue_url ? (
            <a href={report.github_issue_url} target="_blank" rel="noreferrer">
              View the filed GitHub issue →
            </a>
          ) : (
            <p className="muted" style={{ fontSize: '0.8125rem' }}>
              No issue was filed. <code>file_github_issue</code> is a dry run unless{' '}
              <code>AGENT_GITHUB_MODE=live</code> and GitHub credentials are both set.
            </p>
          )}
        </div>
      </Card>

      <div className="stack">
        <Card title="Evidence" subtitle="Each claim names the tool result that establishes it.">
          {report.evidence?.length ? (
            <div className="evidence">
              {report.evidence.map((item, i) => (
                <div className="evidence-item" key={i}>
                  <p className="evidence-claim">{item.claim}</p>
                  <p className="evidence-detail">{item.detail}</p>
                  <span className="tag evidence-source">
                    {TOOL_LABELS[item.source_tool] ?? item.source_tool}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <EmptyState title="No evidence cited" hint="The report made no traceable claims." />
          )}
        </Card>

        <Card
          title="Agent trace"
          subtitle={`${report.trace?.length ?? 0} steps · stopped: ${report.stop_state ?? 'unknown'}`}
        >
          <AgentTrace trace={report.trace ?? []} />
        </Card>
      </div>
    </div>
  );
}

function AgentTrace({ trace }) {
  if (!trace.length) {
    return <EmptyState title="No trace recorded" hint="This investigation stored no steps." />;
  }

  return (
    <div className="trace">
      {trace.map((step, index) => {
        const isModel = step.type === 'model_turn';
        const isError = step.is_error || step.type === 'refusal';
        const last = index === trace.length - 1;

        return (
          <div
            key={step.step ?? index}
            className={`trace-step${isModel ? ' is-model' : ''}${isError ? ' is-error' : ''}`}
          >
            <div className="trace-rail">
              <span className="trace-dot" />
              {!last && <span className="trace-line" />}
            </div>
            <div className="trace-body">
              <div className="trace-title">
                {isModel && <>Model turn {step.iteration}</>}
                {step.type === 'tool_result' && (
                  <>
                    <code>{step.name}</code>
                    {isError && <span className="tag" style={{ color: 'var(--critical)' }}>error</span>}
                  </>
                )}
                {step.type === 'refusal' && <>Refused{step.category ? ` (${step.category})` : ''}</>}
                <span className="trace-meta">
                  {step.latency_ms !== undefined && `${step.latency_ms} ms`}
                  {step.usage?.output_tokens ? ` · ${step.usage.output_tokens} out` : ''}
                </span>
              </div>

              {isModel && step.text && <p className="trace-text">{step.text}</p>}
              {isModel && step.tool_calls?.length > 0 && (
                <p className="trace-text">
                  Called {step.tool_calls.map((call) => call.name).join(', ')}
                </p>
              )}

              {step.type === 'tool_result' && (
                <details className="raw">
                  <summary>
                    {step.input ? summariseArgs(step.input) : 'result'}
                  </summary>
                  <pre>{JSON.stringify(step.output, null, 2)}</pre>
                </details>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function summariseArgs(input) {
  return Object.entries(input)
    .slice(0, 3)
    .map(([key, value]) => {
      const text = String(value);
      return `${key}=${text.length > 26 ? `${text.slice(0, 26)}…` : text}`;
    })
    .join(', ');
}
