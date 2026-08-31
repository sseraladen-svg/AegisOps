import { useMemo, useRef, useState } from 'react';
import { DETECTOR_SHORT, METRIC_UNITS } from '../api/client';
import { ChartTooltip, TipRow } from './Tooltip';
import { formatTs } from './format';
import { useMeasure } from './useMeasure';

const MARGIN = { top: 12, right: 16, bottom: 26, left: 52 };

// Lane labels sit in the left margin, so they need to fit in it. "Isolation
// Forest" does not, and overflowed off the left edge of the card.
const LANE_LABELS = {
  naive: 'Naive',
  isolation_forest: 'iForest',
  lstm_autoencoder: 'LSTM',
};
const LANE_HEIGHT = 11;
const LANE_GAP = 5;
const LANE_LABEL_GAP = 8;

/**
 * Metric over time, with each detector's flagged windows as its own lane
 * beneath the plot.
 *
 * The lanes are the point of this chart. Drawing every detector's windows as
 * translucent full-height bands over the same plot — the obvious first
 * implementation — makes three overlapping washes that composite into a fourth
 * color nobody can decode, and the one thing a reader needs here is *which*
 * detector fired where. One opaque lane per detector answers that directly, and
 * a gap in a lane reads as "this detector missed it", which is exactly the
 * finding the whole project is built around.
 *
 * Ground truth stays a full-height neutral band: it is not a detector, so it
 * does not get a categorical hue, and putting it behind the line lets every
 * lane be compared against it at a glance.
 */
export function TimelineChart({
  series,            // { points: [{ts, value}], metric_name, unit }
  incidentsByDetector, // { [detector]: [{id, ts_start, ts_end, anomaly_score}] }
  detectors,         // ordered detector ids to show as lanes
  groundTruth = [],  // [{ts_start, ts_end, difficulty_tier}]
  height = 260,
  onSelectIncident,
}) {
  const [hostRef, width] = useMeasure();
  const svgRef = useRef(null);
  const [hover, setHover] = useState(null);

  const laneCount = detectors.length;
  const laneBlock = laneCount ? laneCount * (LANE_HEIGHT + LANE_GAP) + LANE_LABEL_GAP : 0;
  const plotHeight = Math.max(80, height - MARGIN.top - MARGIN.bottom - laneBlock);
  const totalHeight = MARGIN.top + plotHeight + laneBlock + MARGIN.bottom;
  const plotWidth = Math.max(0, width - MARGIN.left - MARGIN.right);

  const model = useMemo(() => {
    const points = (series?.points ?? [])
      .map((p) => ({ t: new Date(p.ts).getTime(), v: p.value, ts: p.ts }))
      .filter((p) => Number.isFinite(p.t) && Number.isFinite(p.v));
    if (points.length < 2) return null;

    const t0 = points[0].t;
    const t1 = points[points.length - 1].t;
    const values = points.map((p) => p.v);
    const rawMin = Math.min(...values);
    const rawMax = Math.max(...values);

    // Pad the value axis and pull the floor to zero when the data is already
    // near it — a y-axis that starts at 31.2 exaggerates a 2% wobble into a
    // cliff, which is the classic way a monitoring chart lies.
    const pad = (rawMax - rawMin || 1) * 0.08;
    const min = rawMin - pad < 0 || rawMin < (rawMax - rawMin) * 0.35 ? 0 : rawMin - pad;
    const max = rawMax + pad;

    return { points, t0, t1, min, max, span: max - min || 1 };
  }, [series]);

  if (!width) return <div ref={hostRef} style={{ height: totalHeight }} />;

  if (!model) {
    return (
      <div ref={hostRef}>
        <p className="muted" style={{ padding: '2rem 0', textAlign: 'center' }}>
          Not enough data points to draw a timeline.
        </p>
      </div>
    );
  }

  const { points, t0, t1, min, max, span } = model;
  const timeSpan = t1 - t0 || 1;

  const toX = (t) => MARGIN.left + ((t - t0) / timeSpan) * plotWidth;
  const toY = (v) => MARGIN.top + (1 - (v - min) / span) * plotHeight;

  const linePath = points
    .map((p, i) => `${i === 0 ? 'M' : 'L'}${toX(p.t).toFixed(2)},${toY(p.v).toFixed(2)}`)
    .join(' ');

  const yTicks = niceTicks(min, max, 4);
  const xTicks = timeTicks(t0, t1, Math.max(2, Math.floor(plotWidth / 110)));
  // Eleven ticks all reading "Jan 1" is noise. Below ~3 days the useful
  // distinction is the hour; above it, the day.
  const spanDays = timeSpan / 86_400_000;
  const formatXTick = spanDays < 3
    ? (t) => new Date(t).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : (t) => new Date(t).toLocaleDateString([], { month: 'short', day: 'numeric' });
  const unit = series?.unit ?? METRIC_UNITS[series?.metric_name] ?? '';

  const laneTop = MARGIN.top + plotHeight + LANE_LABEL_GAP;

  function handlePointer(event) {
    const rect = svgRef.current.getBoundingClientRect();
    const x = event.clientX - rect.left;
    if (x < MARGIN.left - 8 || x > MARGIN.left + plotWidth + 8) {
      setHover(null);
      return;
    }
    const ratio = Math.min(1, Math.max(0, (x - MARGIN.left) / (plotWidth || 1)));
    const target = t0 + ratio * timeSpan;
    // The crosshair snaps to the nearest sample, so the reader aims at a time
    // rather than at a 2px line.
    let best = 0;
    let bestDelta = Infinity;
    points.forEach((p, i) => {
      const delta = Math.abs(p.t - target);
      if (delta < bestDelta) { bestDelta = delta; best = i; }
    });
    setHover(best);
  }

  function handleKey(event) {
    if (hover === null) {
      if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
        setHover(Math.floor(points.length / 2));
        event.preventDefault();
      }
      return;
    }
    const step = event.shiftKey ? 10 : 1;
    if (event.key === 'ArrowRight') {
      setHover((i) => Math.min(points.length - 1, i + step));
      event.preventDefault();
    } else if (event.key === 'ArrowLeft') {
      setHover((i) => Math.max(0, i - step));
      event.preventDefault();
    } else if (event.key === 'Escape') {
      setHover(null);
    }
  }

  const hovered = hover === null ? null : points[hover];
  const hoveredFlags = hovered
    ? detectors
      .map((detector) => ({
        detector,
        hit: (incidentsByDetector[detector] ?? []).find(
          (inc) => new Date(inc.ts_start).getTime() <= hovered.t
            && new Date(inc.ts_end).getTime() >= hovered.t,
        ),
      }))
      .filter((row) => row.hit)
    : [];

  return (
    <div className="chart-host" ref={hostRef}>
      <figure className="chart-figure">
        <svg
          ref={svgRef}
          className="chart-svg"
          width={width}
          height={totalHeight}
          onPointerMove={handlePointer}
          onPointerLeave={() => setHover(null)}
          onKeyDown={handleKey}
          tabIndex={0}
          role="img"
          aria-label={`${series.metric_name} for ${series.service_name}, ${points.length} samples from ${formatTs(points[0].ts)} to ${formatTs(points[points.length - 1].ts)}. Use arrow keys to inspect values.`}
          style={{ outlineOffset: 2 }}
        >
          {/* ground truth: neutral, behind everything, never a series hue */}
          {groundTruth.map((gt, i) => {
            const x1 = toX(new Date(gt.ts_start).getTime());
            const x2 = toX(new Date(gt.ts_end).getTime());
            return (
              <rect
                key={`gt-${i}`}
                x={x1}
                y={MARGIN.top}
                width={Math.max(2, x2 - x1)}
                height={plotHeight}
                fill="var(--text-primary)"
                opacity={0.11}
              />
            );
          })}

          {/* gridlines: solid hairlines, one step off the surface */}
          {yTicks.map((tick) => (
            <g key={`y-${tick}`}>
              <line
                x1={MARGIN.left} x2={MARGIN.left + plotWidth}
                y1={toY(tick)} y2={toY(tick)}
                stroke="var(--gridline)" strokeWidth={1}
              />
              <text
                x={MARGIN.left - 8} y={toY(tick)}
                textAnchor="end" dominantBaseline="middle"
                fill="var(--text-muted)" fontSize={10.5}
                style={{ fontVariantNumeric: 'tabular-nums' }}
              >
                {formatTick(tick)}
              </text>
            </g>
          ))}

          {xTicks.map((tick) => (
            <text
              key={`x-${tick}`}
              x={toX(tick)} y={MARGIN.top + plotHeight + laneBlock + 16}
              textAnchor="middle" fill="var(--text-muted)" fontSize={10.5}
            >
              {formatXTick(tick)}
            </text>
          ))}

          <line
            x1={MARGIN.left} x2={MARGIN.left + plotWidth}
            y1={MARGIN.top + plotHeight} y2={MARGIN.top + plotHeight}
            stroke="var(--axis)" strokeWidth={1}
          />

          {/* the metric itself wears ink, so the categorical hues stay free
              to mean "detector" and nothing else */}
          <path
            d={linePath}
            fill="none"
            stroke="var(--text-primary)"
            strokeWidth={2}
            strokeLinejoin="round"
            strokeLinecap="round"
            opacity={0.82}
          />

          {/* one lane per detector */}
          {detectors.map((detector, laneIndex) => {
            const y = laneTop + laneIndex * (LANE_HEIGHT + LANE_GAP);
            const windows = incidentsByDetector[detector] ?? [];
            return (
              <g key={detector}>
                <rect
                  x={MARGIN.left} y={y} width={plotWidth} height={LANE_HEIGHT}
                  fill="var(--wash)" rx={3}
                />
                {windows.map((inc) => {
                  const x1 = toX(new Date(inc.ts_start).getTime());
                  const x2 = toX(new Date(inc.ts_end).getTime());
                  return (
                    <rect
                      key={inc.id}
                      x={x1}
                      y={y}
                      // A 5-minute window is sub-pixel at 14 days; floor the
                      // width so a real detection is never invisible.
                      width={Math.max(2.5, x2 - x1)}
                      height={LANE_HEIGHT}
                      rx={2.5}
                      fill={`var(--${detector})`}
                      style={{ cursor: onSelectIncident ? 'pointer' : 'default' }}
                      onClick={() => onSelectIncident?.(inc)}
                    >
                      <title>
                        {`${DETECTOR_SHORT[detector]} incident #${inc.id}: ${formatTs(inc.ts_start)} – ${formatTs(inc.ts_end)}`}
                      </title>
                    </rect>
                  );
                })}
                <text
                  x={MARGIN.left - 8} y={y + LANE_HEIGHT / 2}
                  textAnchor="end" dominantBaseline="middle"
                  fill="var(--text-muted)" fontSize={9.5}
                >
                  {LANE_LABELS[detector] ?? DETECTOR_SHORT[detector]}
                </text>
              </g>
            );
          })}

          {hovered && (
            <g pointerEvents="none">
              <line
                x1={toX(hovered.t)} x2={toX(hovered.t)}
                y1={MARGIN.top} y2={MARGIN.top + plotHeight + laneBlock}
                stroke="var(--text-muted)" strokeWidth={1}
              />
              <circle cx={toX(hovered.t)} cy={toY(hovered.v)} r={5} fill="var(--surface-1)" />
              <circle cx={toX(hovered.t)} cy={toY(hovered.v)} r={3.5} fill="var(--text-primary)" />
            </g>
          )}
        </svg>
      </figure>

      {hovered && (
        <ChartTooltip hostRef={hostRef} x={toX(hovered.t)} y={toY(hovered.v)}>
          <div className="tip-head">{formatTs(hovered.ts, { withSeconds: false })}</div>
          <TipRow name={`${series.metric_name}${unit ? ` (${unit})` : ''}`} value={hovered.v.toFixed(2)} />
          {hoveredFlags.map(({ detector, hit }) => (
            <TipRow
              key={detector}
              color={`var(--${detector})`}
              name={DETECTOR_SHORT[detector]}
              value={`#${hit.id}`}
            />
          ))}
          {hoveredFlags.length === 0 && (
            <div className="tip-row"><span className="tip-name">no detector flagged this</span></div>
          )}
        </ChartTooltip>
      )}
    </div>
  );
}

function formatTick(value) {
  const abs = Math.abs(value);
  if (abs < 1e-9) return '0';
  if (abs >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (abs >= 10) return value.toFixed(0);
  if (abs >= 1) return value.toFixed(1);
  return value.toFixed(2);
}

/** Round tick values to 1/2/5 × 10^n, so the axis reads in clean numbers. */
function niceTicks(min, max, count) {
  const raw = (max - min) / Math.max(1, count);
  const magnitude = 10 ** Math.floor(Math.log10(raw || 1));
  const normalized = raw / magnitude;
  const step = (normalized >= 5 ? 10 : normalized >= 2 ? 5 : normalized >= 1 ? 2 : 1) * magnitude;
  const ticks = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) {
    ticks.push(Number(v.toFixed(6)));
  }
  return ticks.length >= 2 ? ticks : [min, max];
}

function timeTicks(t0, t1, count) {
  const step = (t1 - t0) / Math.max(1, count);
  return Array.from({ length: count + 1 }, (_, i) => t0 + i * step);
}
