import { describe, expect, it } from 'vitest';
import { formatDuration, formatNumber, formatTs, toApiTs } from './format';

describe('toApiTs', () => {
  it('formats a local Date as naive ISO-8601, no zone suffix', () => {
    const date = new Date(2026, 0, 1, 8, 20, 5); // month is 0-indexed
    expect(toApiTs(date)).toBe('2026-01-01T08:20:05');
  });

  it('zero-pads every field', () => {
    const date = new Date(2026, 8, 3, 4, 5, 6);
    expect(toApiTs(date)).toBe('2026-09-03T04:05:06');
  });
});

describe('formatTs', () => {
  it('returns an em dash for a falsy value', () => {
    expect(formatTs(null)).toBe('—');
    expect(formatTs(undefined)).toBe('—');
    expect(formatTs('')).toBe('—');
  });

  it('falls back to the raw string for an unparseable value', () => {
    expect(formatTs('not-a-date')).toBe('not-a-date');
  });

  it('includes a date by default and can omit it', () => {
    const withDate = formatTs('2026-01-01T08:20:00');
    const withoutDate = formatTs('2026-01-01T08:20:00', { withDate: false });
    expect(withDate.length).toBeGreaterThan(withoutDate.length);
  });
});

describe('formatNumber', () => {
  it('returns an em dash for null/undefined/NaN', () => {
    expect(formatNumber(null)).toBe('—');
    expect(formatNumber(undefined)).toBe('—');
    expect(formatNumber(NaN)).toBe('—');
  });

  it('fixes to the requested number of digits', () => {
    expect(formatNumber(3.14159, 2)).toBe('3.14');
    expect(formatNumber(3, 0)).toBe('3');
  });

  it('defaults to 2 digits', () => {
    expect(formatNumber(1)).toBe('1.00');
  });
});

describe('formatDuration', () => {
  it('returns an em dash for null/undefined', () => {
    expect(formatDuration(null)).toBe('—');
    expect(formatDuration(undefined)).toBe('—');
  });

  it('renders a zero-width window as "1 sample", not "0m"', () => {
    expect(formatDuration(0)).toBe('1 sample');
  });

  it('renders sub-hour durations in minutes', () => {
    expect(formatDuration(45)).toBe('45m');
  });

  it('renders sub-day durations in hours', () => {
    expect(formatDuration(150)).toBe('2.5h');
  });

  it('renders multi-day durations in days', () => {
    expect(formatDuration(60 * 24 * 3)).toBe('3.0d');
  });
});
