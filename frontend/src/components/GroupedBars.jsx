import { useRef, useState } from 'react';
import { DETECTOR_SHORT } from '../api/client';
import { ChartTooltip, TipRow } from './Tooltip';
import { useMeasure } from './useMeasure';

const MARGIN = { top: 20, right: 12, bottom: 40, left: 34 };
const MAX_BAR = 24;      // never fill the slot; the leftover is air
const BAR_GAP = 2;       // surface gap between adjacent bars
const HIT_MIN = 24;      // hover targets are bigger than the marks
const LABEL_CHAR_PX = 5.6;  // approx advance of a digit at 9.5px in the UI sans
const LABEL_PAD = 5;

/**
 * Grouped columns: one group per difficulty tier, one column per detector.
 *
 * The y-axis is fixed to 0–1 for every panel rather than scaled to each
 * panel's own maximum. Precision here is in the low hundredths, so a
 * per-panel scale would blow those bars up to look comparable to recall's
 * full-height ones — the reader would come away thinking precision was fine.
 * Every bar is direct-labeled instead, which keeps the small values readable
 * without distorting them, and doubles as the relief the light-mode aqua slot
 * needs to clear its contrast warning.
 */
export function GroupedBars({
  title, subtitle, groups, groupLabels, series, values, height = 190, formatValue,
}) {
  const [hostRef, width] = useMeasure();
  const svgRef = useRef(null);
  const [hover, setHover] = useState(null);

  const plotWidth = Math.max(0, width - MARGIN.left - MARGIN.right);
  const plotHeight = height - MARGIN.top - MARGIN.bottom;
  const format = formatValue ?? ((v) => v.toFixed(3).replace(/^0\./, '.'));

  if (!width) return <div ref={hostRef} style={{ height }} />;


  const groupWidth = plotWidth / groups.length;
  const bandWidth = groupWidth * 0.78;
  const bandStart = (groupWidth - bandWidth) / 2;
  const slot = bandWidth / series.length;
  const barWidth = Math.min(MAX_BAR, slot - BAR_GAP);

  // A label that won't fit doesn't get clipped or overlapped into mush — it is
  // dropped, and the table view (which every panel here has) carries the value.
  // Measured against the widest label actually rendered, not a guess.
  const widestLabel = Math.max(
    ...series.flatMap((d) => (values[d] ?? []).map((v) => format(v).length)),
    1,
  );
  const showLabels = slot >= widestLabel * LABEL_CHAR_PX + LABEL_PAD;

  const toY = (v) => MARGIN.top + (1 - Math.max(0, Math.min(1, v))) * plotHeight;
  const yTicks = [0, 0.25, 0.5, 0.75, 1];

  const hovered = hover
    ? { ...hover, value: values[hover.detector]?.[hover.groupIndex] ?? 0 }
    : null;

  return (
    <div className="chart-host" ref={hostRef}>
      <figure className="chart-figure">
        <figcaption style={{ marginBottom: '0.5rem' }}>
          <h3>{title}</h3>
          {subtitle && <p className="sub muted" style={{ fontSize: '0.75rem' }}>{subtitle}</p>}
        </figcaption>
        <svg
          ref={svgRef}
          className="chart-svg"
          width={width}
          height={height}
          role="img"
          aria-label={`${title} by difficulty tier for ${series.length} detectors. Values are also in the table below.`}
        >
          {yTicks.map((tick) => (
            <g key={tick}>
              <line
                x1={MARGIN.left} x2={MARGIN.left + plotWidth}
                y1={toY(tick)} y2={toY(tick)}
                stroke={tick === 0 ? 'var(--axis)' : 'var(--gridline)'} strokeWidth={1}
              />
              <text
                x={MARGIN.left - 7} y={toY(tick)}
                textAnchor="end" dominantBaseline="middle"
                fill="var(--text-muted)" fontSize={10}
                style={{ fontVariantNumeric: 'tabular-nums' }}
              >
                {tick}
              </text>
            </g>
          ))}

          {groups.map((group, groupIndex) => {
            const gx = MARGIN.left + groupIndex * groupWidth;
            return (
              <g key={group}>
                {series.map((detector, seriesIndex) => {
                  const value = values[detector]?.[groupIndex] ?? 0;
                  const x = gx + bandStart + seriesIndex * slot + (slot - barWidth) / 2;
                  const y = toY(value);
                  const barHeight = Math.max(0, toY(0) - y);
                  const isHovered = hover?.detector === detector && hover?.groupIndex === groupIndex;

                  return (
                    <g key={detector}>
                      {/* Hit area spans the full slot height and at least 24px
                          wide, so a near-zero bar is still hoverable. */}
                      <rect
                        x={x - Math.max(0, (HIT_MIN - barWidth) / 2)}
                        y={MARGIN.top}
                        width={Math.max(barWidth, HIT_MIN)}
                        height={plotHeight}
                        fill="transparent"
                        onPointerEnter={() => setHover({ detector, groupIndex, x: x + barWidth / 2, y })}
                        onPointerLeave={() => setHover(null)}
                        onFocus={() => setHover({ detector, groupIndex, x: x + barWidth / 2, y })}
                        onBlur={() => setHover(null)}
                        tabIndex={0}
                        role="button"
                        aria-label={`${DETECTOR_SHORT[detector]}, ${groupLabels[groupIndex]}: ${value.toFixed(4)}`}
                      />
                      {barHeight > 0.5 ? (
                        <rect
                          x={x} y={y} width={barWidth} height={barHeight}
                          fill={`var(--${detector})`}
                          opacity={hover && !isHovered ? 0.55 : 1}
                          rx={3}
                          pointerEvents="none"
                        />
                      ) : (
                        // A true zero still needs to be visible as "measured
                        // zero" rather than "no data" — a 2px stub on the
                        // baseline says that without implying a value.
                        <rect
                          x={x} y={toY(0) - 2} width={barWidth} height={2}
                          fill="var(--axis)" pointerEvents="none"
                        />
                      )}
                      {showLabels && (
                        <text
                          x={x + barWidth / 2}
                          y={Math.min(y - 5, toY(0) - 5)}
                          textAnchor="middle"
                          fill="var(--text-secondary)"
                          fontSize={9.5}
                          pointerEvents="none"
                          style={{ fontVariantNumeric: 'tabular-nums' }}
                        >
                          {format(value)}
                        </text>
                      )}
                    </g>
                  );
                })}
                {/* Anchor the outermost labels to the plot edges: centred, the
                    last group's label runs past the SVG and pushes the whole
                    page into a horizontal scroll on narrow screens. */}
                <text
                  x={
                    groupIndex === 0 ? MARGIN.left
                      : groupIndex === groups.length - 1 ? MARGIN.left + plotWidth
                        : gx + groupWidth / 2
                  }
                  y={MARGIN.top + plotHeight + 16}
                  textAnchor={
                    groupIndex === 0 ? 'start'
                      : groupIndex === groups.length - 1 ? 'end' : 'middle'
                  }
                  fill="var(--text-secondary)" fontSize={11}
                >
                  {groupLabels[groupIndex]}
                </text>
              </g>
            );
          })}
        </svg>
      </figure>

      {!showLabels && (
        <p className="muted" style={{ fontSize: '0.6875rem', marginTop: '0.25rem' }}>
          Values omitted — too narrow to label without overlap. Hover a bar, or use
          &ldquo;Show table&rdquo;.
        </p>
      )}

      {hovered && (
        <ChartTooltip hostRef={hostRef} x={hovered.x} y={hovered.y}>
          <div className="tip-head">{groupLabels[hovered.groupIndex]}</div>
          <TipRow
            color={`var(--${hovered.detector})`}
            name={DETECTOR_SHORT[hovered.detector]}
            value={hovered.value.toFixed(4)}
          />
        </ChartTooltip>
      )}
    </div>
  );
}
