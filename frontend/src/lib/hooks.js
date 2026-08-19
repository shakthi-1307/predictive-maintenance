import { useEffect, useRef, useState } from "react";

export function usePrefersReducedMotion() {
  const [reduced, setReduced] = useState(
    () =>
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );

  useEffect(() => {
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    const onChange = () => setReduced(query.matches);
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, []);

  return reduced;
}

/** Eases a displayed number towards `value` so readouts tick rather than jump. */
export function useCountUp(value, duration = 700) {
  const reduced = usePrefersReducedMotion();
  const [display, setDisplay] = useState(value ?? 0);
  const fromRef = useRef(value ?? 0);

  useEffect(() => {
    const target = Number(value) || 0;

    if (reduced || duration === 0) {
      fromRef.current = target;
      setDisplay(target);
      return undefined;
    }

    const from = fromRef.current;
    const startedAt = performance.now();
    let frame;

    const tick = (now) => {
      const t = Math.min(1, (now - startedAt) / duration);
      // easeOutCubic
      const eased = 1 - Math.pow(1 - t, 3);
      const current = from + (target - from) * eased;
      setDisplay(current);
      fromRef.current = current;
      if (t < 1) frame = requestAnimationFrame(tick);
      else fromRef.current = target;
    };

    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [value, duration, reduced]);

  return display;
}

/** Container width, so SVG charts can lay out in real pixels instead of
 *  being stretched by a non-uniform viewBox. */
export function useMeasure() {
  const ref = useRef(null);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const node = ref.current;
    if (!node) return undefined;

    const observer = new ResizeObserver((entries) => {
      setWidth(entries[0].contentRect.width);
    });

    observer.observe(node);
    setWidth(node.getBoundingClientRect().width);

    return () => observer.disconnect();
  }, []);

  return [ref, width];
}
