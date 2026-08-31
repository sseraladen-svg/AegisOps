import { useEffect, useRef, useState } from 'react';

/**
 * Container width, observed.
 *
 * Charts here render at real pixel coordinates rather than a scaled viewBox:
 * a viewBox stretched to fit would scale the axis text with the plot, so the
 * same chart would have 9px labels on a phone and 20px ones on a wide monitor.
 */
export function useMeasure() {
  const ref = useRef(null);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const node = ref.current;
    if (!node) return undefined;

    const observer = new ResizeObserver(([entry]) => {
      setWidth(Math.round(entry.contentRect.width));
    });
    observer.observe(node);
    setWidth(Math.round(node.getBoundingClientRect().width));

    return () => observer.disconnect();
  }, []);

  return [ref, width];
}
