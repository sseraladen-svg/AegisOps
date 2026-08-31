import { Link } from 'react-router-dom';

/** Status pill. Color never carries the meaning alone — icon + label do. */
const STATUS = {
  healthy: { icon: '●', label: 'Healthy', color: 'var(--good)' },
  degraded: { icon: '▲', label: 'Degraded', color: 'var(--warning)' },
  critical: { icon: '■', label: 'Critical', color: 'var(--critical)' },
  unknown: { icon: '—', label: 'Unknown', color: 'var(--text-muted)' },
};

export function StatusPill({ status, children }) {
  const spec = STATUS[status] ?? STATUS.unknown;
  // One glyph, in the status color: the *shape* differs per status, so the
  // meaning survives without color, and a separate neutral dot would just be
  // ink repeating what the glyph already says.
  return (
    <span className="pill">
      <span className="icon" style={{ color: spec.color }} aria-hidden="true">{spec.icon}</span>
      <span>{children ?? spec.label}</span>
    </span>
  );
}

export function Card({ title, subtitle, actions, children, className = '' }) {
  return (
    <section className={`card ${className}`}>
      {(title || actions) && (
        <header className="card-head">
          <div>
            {title && <h2>{title}</h2>}
            {subtitle && <p className="sub">{subtitle}</p>}
          </div>
          {actions}
        </header>
      )}
      {children}
    </section>
  );
}

export function Stat({ label, value, note }) {
  return (
    <div className="stat">
      <span className="label">{label}</span>
      {/* Proportional figures on purpose — tabular-nums makes display-size
          numbers look loose. Tabular is reserved for table columns. */}
      <span className="value">{value}</span>
      {note && <span className="note">{note}</span>}
    </div>
  );
}

export function Loading({ label = 'Loading…', rows = 3 }) {
  return (
    <div className="state" role="status" aria-live="polite">
      <div style={{ width: '100%', display: 'grid', gap: '0.5rem' }}>
        {Array.from({ length: rows }, (_, i) => (
          <div key={i} className="skeleton" style={{ height: 14, width: `${92 - i * 16}%` }} />
        ))}
      </div>
      <span className="sr-only">{label}</span>
    </div>
  );
}

export function ErrorState({ error, onRetry }) {
  return (
    <div className="state error" role="alert">
      <span className="title">Couldn&apos;t load this</span>
      <span className="hint">{error?.message ?? 'Unknown error'}</span>
      {onRetry && (
        <button type="button" className="theme-toggle" onClick={onRetry}>Try again</button>
      )}
    </div>
  );
}

export function EmptyState({ title, hint, children }) {
  return (
    <div className="state">
      <span className="title">{title}</span>
      {hint && <span className="hint">{hint}</span>}
      {children}
    </div>
  );
}

/**
 * Loading / error / empty in one place.
 *
 * `refetching` is handled by holding the previous children at reduced opacity
 * rather than swapping in a skeleton, so a filter change never causes a
 * layout jump.
 */
export function AsyncBoundary({
  loading, refetching, error, data, onRetry, empty, children, loadingRows = 3,
}) {
  if (error) return <ErrorState error={error} onRetry={onRetry} />;
  if (loading && !refetching) return <Loading rows={loadingRows} />;
  if (!data) return null;
  if (empty?.(data)) {
    return <EmptyState title={empty.title ?? 'Nothing here yet'} hint={empty.hint} />;
  }
  return <div className={refetching ? 'refetching' : undefined}>{children(data)}</div>;
}

export function DetectorSwatch({ detector, as = 'rect' }) {
  return (
    <span
      className={as === 'line' ? 'swatch line' : 'swatch'}
      style={{ background: `var(--${detector})` }}
      aria-hidden="true"
    />
  );
}

export function IncidentLink({ id, children }) {
  return <Link to={`/incidents/${id}`}>{children ?? `#${id}`}</Link>;
}
