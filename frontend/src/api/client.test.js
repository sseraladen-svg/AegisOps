import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { api, ApiError } from './client';

function jsonResponse(body, { ok = true, status = 200, statusText = 'OK' } = {}) {
  return { ok, status, statusText, json: async () => body };
}

describe('api client', () => {
  beforeEach(() => {
    globalThis.fetch = vi.fn();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('requests the right path with no query string when there are no params', async () => {
    globalThis.fetch.mockResolvedValue(jsonResponse([{ id: 1, name: 'auth-service' }]));
    await api.services();
    const url = new URL(globalThis.fetch.mock.calls[0][0]);
    expect(url.pathname).toBe('/api/services');
    expect(url.search).toBe('');
  });

  it('includes query params and skips undefined/null/empty ones', async () => {
    globalThis.fetch.mockResolvedValue(jsonResponse({ points: [] }));
    await api.metrics({ service_id: 1, metric_name: 'cpu_usage', ts_start: undefined, ts_end: null, max_points: '' });
    const url = new URL(globalThis.fetch.mock.calls[0][0]);
    expect(url.searchParams.get('service_id')).toBe('1');
    expect(url.searchParams.get('metric_name')).toBe('cpu_usage');
    expect(url.searchParams.has('ts_start')).toBe(false);
    expect(url.searchParams.has('ts_end')).toBe(false);
    expect(url.searchParams.has('max_points')).toBe(false);
  });

  it('throws an ApiError carrying the backend\'s detail string on a non-2xx response', async () => {
    globalThis.fetch.mockResolvedValue(
      jsonResponse({ detail: 'no service with id 999' }, { ok: false, status: 404, statusText: 'Not Found' }));

    await expect(api.services()).rejects.toMatchObject({
      name: 'ApiError',
      message: 'no service with id 999',
      status: 404,
    });
  });

  it('falls back to the status line when the error body is not JSON', async () => {
    globalThis.fetch.mockResolvedValue({
      ok: false, status: 500, statusText: 'Internal Server Error',
      json: async () => { throw new Error('not json'); },
    });

    await expect(api.services()).rejects.toMatchObject({
      message: '500 Internal Server Error',
    });
  });

  it('wraps a network failure in a friendly ApiError instead of the raw fetch error', async () => {
    globalThis.fetch.mockRejectedValue(new TypeError('Failed to fetch'));

    await expect(api.services()).rejects.toMatchObject({
      name: 'ApiError',
    });
    await expect(api.services()).rejects.toThrow(/Could not reach the API/);
  });

  it('re-throws an AbortError as-is rather than wrapping it', async () => {
    const abortError = new DOMException('aborted', 'AbortError');
    globalThis.fetch.mockRejectedValue(abortError);

    await expect(api.services()).rejects.toBe(abortError);
  });

  it('returns the parsed JSON body on success', async () => {
    const body = [{ id: 1, name: 'auth-service' }];
    globalThis.fetch.mockResolvedValue(jsonResponse(body));
    await expect(api.services()).resolves.toEqual(body);
  });
});

describe('ApiError', () => {
  it('carries status and url', () => {
    const error = new ApiError('boom', { status: 503, url: 'http://x/y' });
    expect(error).toBeInstanceOf(Error);
    expect(error.name).toBe('ApiError');
    expect(error.status).toBe(503);
    expect(error.url).toBe('http://x/y');
  });
});
