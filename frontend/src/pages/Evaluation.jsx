import { useCallback, useState } from 'react';
import { DETECTORS, DETECTOR_LABELS, DETECTOR_SHORT, TIER_LABELS, api } from '../api/client';
import { useApi } from '../api/useApi';
import { GroupedBars } from '../components/GroupedBars';
import { AsyncBoundary, Card, DetectorSwatch, Stat } from '../components/ui';
import { formatNumber, formatTs } from '../components/format';

const METRIC_PANELS = [
  {
    key: 'precision',
    title: 'Precision',
    subtitle: 'Of the incidents this detector raised in this tier, how many were real.',
  },
  {
    key: 'recall',
    title: 'Recall',
    subtitle: 'Of the real anomalies in this tier, how many it caught.',
  },
  {
    key: 'f1',
    title: 'F1',
    subtitle: 'Harmonic mean of the two above.',
  },
];

export default function Evaluation() {
  const results = useApi(useCallback((opts) => api.evalResults(opts), []), []);
  const [showTable, setShowTable] = useState(false);

  return (
    <>
      <div className="page-head">
        <h1>Evaluation</h1>
        <p className="lede">
          The frozen comparison from <code>eval/results.json</code>. An incident counts as a
          catch when it overlaps a labelled anomaly on the same service and metric at all —
          no minimum-overlap threshold, so a detection that is slightly early or late still
          counts.
        </p>
      </div>

      <AsyncBoundary
        {...results}
        onRetry={results.refresh}
        loadingRows={6}
        empty={Object.assign((d) => !d?.detectors, {
          title: 'No results yet',
          hint: 'Run `make eval` to score the detectors and write eval/results.json.',
        })}
      >
        {(data) => {
          const detectors = DETECTORS.filter((d) => data.detectors[d]);
          const tierLabels = data.tiers.map((t) => TIER_LABELS[t] ?? t);
          const totalAnomalies = Object.values(data.ground_truth_counts ?? {})
            .reduce((sum, n) => sum + n, 0);
          const best = detectors.reduce(
            (acc, d) => (data.detectors[d].overall.f1 > (data.detectors[acc]?.overall.f1 ?? -1) ? d : acc),
            detectors[0],
          );

          return (
            <div className="stack">
              <div className="grid tiles">
                {detectors.map((detector) => {
                  const stats = data.detectors[detector];
                  return (
                    <Card key={detector}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
                        <DetectorSwatch detector={detector} />
                        <h2 style={{ fontSize: '0.9375rem' }}>{DETECTOR_LABELS[detector]}</h2>
                        {detector === best && <span className="tag">best overall F1</span>}
                      </div>
                      <div style={{ display: 'flex', gap: '1.75rem', flexWrap: 'wrap' }}>
                        <Stat
                          label="Anomalies caught"
                          value={`${Math.round(stats.overall.recall * totalAnomalies)}/${totalAnomalies}`}
                          note={`overall recall ${formatNumber(stats.overall.recall, 2)}`}
                        />
                        <Stat
                          label="False positives"
                          value={stats.false_positives.toLocaleString()}
                          note={`of ${stats.incidents.toLocaleString()} incidents raised`}
                        />
                      </div>
                      <p className="muted" style={{ fontSize: '0.75rem', marginTop: '0.875rem' }}>
                        Threshold: {stats.tuning}
                      </p>
                    </Card>
                  );
                })}
              </div>

              <div className="callout">
                <div className="callout-title">Read the two rightmost columns together</div>
                The Isolation Forest and the LSTM autoencoder both reach perfect recall, but the
                Isolation Forest&apos;s contamination is tuned against the very labels this page
                reports on, while the autoencoder&apos;s threshold comes from an unlabelled
                validation split. The autoencoder also raises roughly a fifth as many alarms
                for the same catches. Absolute precision is low for everyone because four real
                anomalies over fourteen days leaves the false-positive count dominating every
                denominator — compare the detectors to each other, not to 1.0.
              </div>

              <Card
                title="Precision, recall and F1 by difficulty tier"
                subtitle="All three panels share a 0–1 axis so the tiny precision values aren't inflated by per-panel scaling."
                actions={
                  <button
                    type="button"
                    className="theme-toggle"
                    onClick={() => setShowTable((v) => !v)}
                    aria-expanded={showTable}
                  >
                    {showTable ? 'Show charts' : 'Show table'}
                  </button>
                }
              >
                {showTable ? (
                  <ResultsTable data={data} detectors={detectors} tierLabels={tierLabels} />
                ) : (
                  <>
                    <div className="grid halves">
                      {METRIC_PANELS.map((panel) => (
                        <GroupedBars
                          key={panel.key}
                          title={panel.title}
                          subtitle={panel.subtitle}
                          groups={data.tiers}
                          groupLabels={tierLabels}
                          series={detectors}
                          values={Object.fromEntries(
                            detectors.map((d) => [d, data.detectors[d][panel.key]]),
                          )}
                          height={200}
                        />
                      ))}
                    </div>
                    <div className="legend">
                      {detectors.map((d) => (
                        <span className="item" key={d}>
                          <DetectorSwatch detector={d} />
                          {DETECTOR_LABELS[d]}
                        </span>
                      ))}
                    </div>
                  </>
                )}
              </Card>

              <Card title="Ground truth" subtitle="What the detectors were scored against.">
                <div className="table-wrap">
                  <table className="data">
                    <thead>
                      <tr><th>Tier</th><th className="num">Labelled anomalies</th></tr>
                    </thead>
                    <tbody>
                      {data.tiers.map((tier) => (
                        <tr key={tier}>
                          <td>{TIER_LABELS[tier] ?? tier}</td>
                          <td className="num">{data.ground_truth_counts?.[tier] ?? 0}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="muted" style={{ fontSize: '0.75rem', marginTop: '0.875rem' }}>
                  Generated {formatTs(data.generated_at, { withSeconds: true })}.
                </p>
              </Card>
            </div>
          );
        }}
      </AsyncBoundary>
    </>
  );
}

/** The table view every chart on this page has a twin in. */
function ResultsTable({ data, detectors, tierLabels }) {
  return (
    <div className="table-wrap">
      <table className="data">
        <thead>
          <tr>
            <th>Detector</th>
            <th>Tier</th>
            <th className="num">Precision</th>
            <th className="num">Recall</th>
            <th className="num">F1</th>
          </tr>
        </thead>
        <tbody>
          {detectors.flatMap((detector) => [
            ...data.tiers.map((tier, i) => (
              <tr key={`${detector}-${tier}`}>
                <td>
                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4375rem' }}>
                    <DetectorSwatch detector={detector} />
                    {DETECTOR_SHORT[detector]}
                  </span>
                </td>
                <td>{tierLabels[i]}</td>
                <td className="num">{formatNumber(data.detectors[detector].precision[i], 4)}</td>
                <td className="num">{formatNumber(data.detectors[detector].recall[i], 4)}</td>
                <td className="num">{formatNumber(data.detectors[detector].f1[i], 4)}</td>
              </tr>
            )),
            <tr key={`${detector}-overall`} style={{ fontWeight: 600 }}>
              <td>
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4375rem' }}>
                  <DetectorSwatch detector={detector} />
                  {DETECTOR_SHORT[detector]}
                </span>
              </td>
              <td>Overall</td>
              <td className="num">{formatNumber(data.detectors[detector].overall.precision, 4)}</td>
              <td className="num">{formatNumber(data.detectors[detector].overall.recall, 4)}</td>
              <td className="num">{formatNumber(data.detectors[detector].overall.f1, 4)}</td>
            </tr>,
          ])}
        </tbody>
      </table>
    </div>
  );
}
