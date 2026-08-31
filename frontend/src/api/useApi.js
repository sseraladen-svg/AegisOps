import { useCallback, useEffect, useState } from 'react';

/**
 * Fetch-on-mount hook with the three states every page here needs.
 *
 * Two deliberate choices:
 *
 * `data` is *kept* while a refetch is in flight, and pages render the stale
 * frame at reduced opacity rather than dropping back to a skeleton. A dashboard
 * that blanks every time a filter changes makes comparison impossible, which is
 * the one thing these pages are for.
 *
 * The "a new request started" transition is applied during render by comparing
 * the request key, not by calling setState inside the effect body. A setState
 * in the effect would render once with stale `loading`, then cascade a second
 * render — visible as a flash of the previous state on every filter change.
 */
export function useApi(fetcher, deps = [], { skip = false } = {}) {
  const key = JSON.stringify([deps, skip]);
  const [state, setState] = useState({
    key, data: null, error: null, loading: !skip, hasLoaded: false,
  });
  const [nonce, setNonce] = useState(0);

  if (state.key !== key) {
    // Keep `data` and `hasLoaded`: that pair is what tells the page it is
    // refetching over an existing render rather than loading from nothing.
    setState((prev) => ({ ...prev, key, error: null, loading: !skip }));
  }

  useEffect(() => {
    if (skip) return undefined;

    const controller = new AbortController();
    let active = true;

    fetcher({ signal: controller.signal })
      .then((data) => {
        if (!active) return;
        setState((prev) => ({ ...prev, data, error: null, loading: false, hasLoaded: true }));
      })
      .catch((error) => {
        if (!active || error?.name === 'AbortError') return;
        setState((prev) => ({ ...prev, data: null, error, loading: false }));
      });

    return () => {
      active = false;
      controller.abort();
    };
    // `fetcher` is expected to be a useCallback keyed on the same deps.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, skip, nonce]);

  const refresh = useCallback(() => setNonce((n) => n + 1), []);

  if (skip) {
    return { data: null, error: null, loading: false, refetching: false, refresh };
  }

  return {
    data: state.data,
    error: state.error,
    loading: state.loading,
    // True only for a reload that already has something on screen, which is
    // what distinguishes "hold the frame" from "show a skeleton".
    refetching: state.loading && state.hasLoaded,
    refresh,
  };
}
