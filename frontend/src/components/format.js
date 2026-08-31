/** Formatting helpers. Kept out of ui.jsx so that file exports only components
 *  (which is what lets Fast Refresh preserve state during development). */

/**
 * Format a Date back into the naive ISO-8601 the API speaks.
 *
 * The backend returns and accepts naive timestamps ("2026-01-01T08:20:00", no
 * zone). `new Date(...)` parses those as *local* time, so `toISOString()` sends
 * them back shifted by the viewer's UTC offset — silently querying the wrong
 * window, and reliably wrong by a different amount for every viewer. This is
 * the exact inverse of that parse, so a timestamp survives the round trip.
 */
export function toApiTs(date) {
  const pad = (n) => String(n).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`
    + `T${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

export function formatTs(value, { withDate = true, withSeconds = false } = {}) {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  const time = date.toLocaleTimeString([], {
    hour: '2-digit', minute: '2-digit',
    ...(withSeconds ? { second: '2-digit' } : {}),
  });
  if (!withDate) return time;
  return `${date.toLocaleDateString([], { month: 'short', day: 'numeric' })} ${time}`;
}

export function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return Number(value).toFixed(digits);
}

export function formatDuration(minutes) {
  if (minutes === null || minutes === undefined) return '—';
  // A window covering a single 5-minute sample has start == end. Rendering
  // that as "0m" reads as a bug rather than as "one sample wide".
  if (minutes === 0) return '1 sample';
  if (minutes < 60) return `${Math.round(minutes)}m`;
  const hours = minutes / 60;
  if (hours < 24) return `${hours.toFixed(1)}h`;
  return `${(hours / 24).toFixed(1)}d`;
}
