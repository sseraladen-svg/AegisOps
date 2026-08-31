import { useId } from 'react';

/**
 * A sparkline is a stat tile's trend cue, not a chart: no axes, no gridlines,
 * no legend, and a single series so no color identity is needed. It wears ink
 * rather than a categorical hue — the categorical slots belong to the three
 * detectors, and spending one on "the metric" would make a tile's line look
 * like a detector.
 */
export function Sparkline({
  values, width = 132, height = 30, strokeWidth = 1.5, label,
}) {
  const clipId = useId();
  const clean = (values ?? []).filter((v) => Number.isFinite(v));

  if (clean.length < 2) {
    return (
      <svg width={width} height={height} role="img" aria-label={`${label ?? 'Trend'}: not enough data`}>
        <line
          x1={0} y1={height / 2} x2={width} y2={height / 2}
          stroke="var(--gridline)" strokeWidth={1}
        />
      </svg>
    );
  }

  const min = Math.min(...clean);
  const max = Math.max(...clean);
  // A flat series would divide by zero; draw it mid-height instead.
  const span = max - min || 1;
  const pad = strokeWidth + 1;

  const toX = (i) => (i / (clean.length - 1)) * width;
  const toY = (v) => pad + (1 - (v - min) / span) * (height - pad * 2);

  const path = clean.map((v, i) => `${i === 0 ? 'M' : 'L'}${toX(i).toFixed(2)},${toY(v).toFixed(2)}`).join(' ');
  const areaPath = `${path} L${width},${height} L0,${height} Z`;
  const lastX = toX(clean.length - 1);
  const lastY = toY(clean[clean.length - 1]);

  return (
    <svg
      width={width}
      height={height}
      className="chart-svg"
      role="img"
      aria-label={`${label ?? 'Trend'}: ${clean.length} points, ranging ${min.toFixed(1)} to ${max.toFixed(1)}, ending ${clean[clean.length - 1].toFixed(1)}`}
      style={{ width, height, flex: 'none' }}
    >
      <clipPath id={clipId}>
        <rect x={0} y={0} width={width} height={height} />
      </clipPath>
      <g clipPath={`url(#${clipId})`}>
        <path d={areaPath} fill="var(--text-primary)" opacity={0.06} />
        <path
          d={path}
          fill="none"
          stroke="var(--text-primary)"
          strokeWidth={strokeWidth}
          strokeLinejoin="round"
          strokeLinecap="round"
          opacity={0.72}
        />
      </g>
      {/* End marker with a surface ring, so it stays legible over the line. */}
      <circle cx={lastX} cy={lastY} r={3.5} fill="var(--surface-1)" />
      <circle cx={lastX} cy={lastY} r={2.5} fill="var(--text-primary)" />
    </svg>
  );
}
