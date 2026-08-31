import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { AsyncBoundary } from './ui';

describe('AsyncBoundary', () => {
  it('shows the loading state while loading and not refetching', () => {
    render(<AsyncBoundary loading data={null}>{() => <div>data</div>}</AsyncBoundary>);
    expect(screen.getByRole('status')).toBeInTheDocument();
    expect(screen.queryByText('data')).not.toBeInTheDocument();
  });

  it('prefers the error state even while loading', () => {
    render(
      <AsyncBoundary loading error={new Error('kaboom')} data={null}>
        {() => <div>data</div>}
      </AsyncBoundary>,
    );
    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(screen.getByText('kaboom')).toBeInTheDocument();
  });

  it('calls onRetry when the retry button is clicked', async () => {
    const onRetry = vi.fn();
    render(<AsyncBoundary error={new Error('kaboom')} data={null} onRetry={onRetry} />);
    screen.getByText('Try again').click();
    expect(onRetry).toHaveBeenCalledOnce();
  });

  it('renders children(data) once loaded', () => {
    render(
      <AsyncBoundary loading={false} data={{ items: [1, 2] }}>
        {(data) => <div>count: {data.items.length}</div>}
      </AsyncBoundary>,
    );
    expect(screen.getByText('count: 2')).toBeInTheDocument();
  });

  it('renders the empty state when the empty predicate matches', () => {
    const empty = (data) => data.items.length === 0;
    empty.title = 'No incidents';
    render(
      <AsyncBoundary loading={false} data={{ items: [] }} empty={empty}>
        {() => <div>should not render</div>}
      </AsyncBoundary>,
    );
    expect(screen.getByText('No incidents')).toBeInTheDocument();
    expect(screen.queryByText('should not render')).not.toBeInTheDocument();
  });

  it('keeps showing prior children while refetching, without the loading skeleton', () => {
    render(
      <AsyncBoundary loading refetching data={{ items: [1] }}>
        {(data) => <div>count: {data.items.length}</div>}
      </AsyncBoundary>,
    );
    expect(screen.getByText('count: 1')).toBeInTheDocument();
    expect(screen.queryByRole('status')).not.toBeInTheDocument();
  });
});
