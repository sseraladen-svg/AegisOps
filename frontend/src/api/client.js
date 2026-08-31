/**
 * Typed-ish client for the FastAPI backend.
 *
 * Every call goes through `request`, which turns a non-2xx into an Error
 * carrying the server's own `detail` string. That matters: the backend's 404s
 * say things like "run `make eval` to generate it", and surfacing that verbatim
 * is the difference between a dashboard that tells you what to do and one that
 * says "Failed to fetch".
 */

export const API_BASE =
  import.meta.env.VITE_API_BASE?.replace(/\/$/, '') ?? 'http://localhost:8000';

export const DETECTORS = ['naive', 'isolation_forest', 'lstm_autoencoder'];

export const DETECTOR_LABELS = {
  naive: 'Naive (z-score)',
  isolation_forest: 'Isolation Forest',
  lstm_autoencoder: 'LSTM autoencoder',
};

export const DETECTOR_SHORT = {
  naive: 'Naive',
  isolation_forest: 'Isolation Forest',
  lstm_autoencoder: 'LSTM AE',
};

export const METRICS = ['cpu_usage', 'latency_ms', 'error_rate', 'request_rate'];

export const METRIC_LABELS = {
  cpu_usage: 'CPU usage',
  latency_ms: 'Latency',
  error_rate: 'Error rate',
  request_rate: 'Request rate',
};

export const METRIC_UNITS = {
  cpu_usage: '%',
  latency_ms: 'ms',
  error_rate: '%',
  request_rate: '/min',
};

export const TIER_LABELS = {
  obvious_spike: 'Obvious spike',
  gradual_drift: 'Gradual drift',
  subtle_correlated: 'Subtle correlated',
};

export class ApiError extends Error {
  constructor(message, { status, url } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.url = url;
  }
}

function buildUrl(path, params) {
  const url = new URL(`${API_BASE}${path}`);
  Object.entries(params ?? {}).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') {
      url.searchParams.set(key, String(value));
    }
  });
  return url.toString();
}

async function request(path, { params, signal, ...init } = {}) {
  const url = buildUrl(path, params);
  let response;
  try {
    response = await fetch(url, { signal, ...init });
  } catch (cause) {
    if (cause?.name === 'AbortError') throw cause;
    // A network-level failure is nearly always "the backend isn't running",
    // which no amount of retrying fixes — say so instead of guessing.
    throw new ApiError(
      `Could not reach the API at ${API_BASE}. Is the backend running? (\`make api\`)`,
      { url },
    );
  }

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body?.detail) {
        detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
      }
    } catch {
      /* body was not JSON; the status line is the best message available */
    }
    throw new ApiError(detail, { status: response.status, url });
  }

  return response.json();
}

export const api = {
  health: (opts) => request('/health', opts),
  services: (opts) => request('/api/services', opts),
  servicesHealth: (params, opts) => request('/api/services/health', { params, ...opts }),
  metrics: (params, opts) => request('/api/metrics', { params, ...opts }),
  logs: (params, opts) => request('/api/logs', { params, ...opts }),
  incidents: (params, opts) => request('/api/incidents', { params, ...opts }),
  incident: (id, opts) => request(`/api/incidents/${id}`, opts),
  investigation: (id, opts) => request(`/api/incidents/${id}/investigation`, opts),
  evalResults: (opts) => request('/api/eval/results', opts),
  groundTruth: (opts) => request('/api/ground-truth', opts),
  demoTour: (opts) => request('/api/eval/demo-tour', opts),
};
