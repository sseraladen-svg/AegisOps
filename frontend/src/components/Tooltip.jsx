import { useLayoutEffect, useRef, useState } from 'react';

/**
 * Chart tooltip.
 *
 * Positioned after layout against the host's box so it flips instead of
 * spilling off the right or bottom edge — a tooltip clipped by the viewport is
 * the same as no tooltip. Content is rendered as React children (text nodes),
 * never innerHTML: series names here come from API responses.
 */
export function ChartTooltip({ hostRef, x, y, children }) {
  const ref = useRef(null);
  const [offset, setOffset] = useState({ left: 0, top: 0 });

  useLayoutEffect(() => {
    const tip = ref.current;
    const host = hostRef.current;
    if (!tip || !host) return;

    const pad = 12;
    const { width, height } = tip.getBoundingClientRect();
    const bounds = host.getBoundingClientRect();

    let left = x + pad;
    if (left + width > bounds.width) left = Math.max(0, x - width - pad);

    let top = y - height / 2;
    top = Math.max(0, Math.min(top, bounds.height - height));

    setOffset({ left, top });
  }, [hostRef, x, y, children]);

  return (
    <div ref={ref} className="tooltip" style={{ left: offset.left, top: offset.top }} role="tooltip">
      {children}
    </div>
  );
}

export function TipRow({ color, name, value, dash = false }) {
  return (
    <div className="tip-row">
      {color && (
        <span
          className={dash ? 'swatch line' : 'swatch'}
          style={{ background: color }}
          aria-hidden="true"
        />
      )}
      <span className="tip-value">{value}</span>
      <span className="tip-name">{name}</span>
    </div>
  );
}
