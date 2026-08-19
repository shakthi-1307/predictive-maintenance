import { useMemo, useState } from "react";

import { RUL_CEILING, statusOf } from "../lib/sensors";
import { useMeasure } from "../lib/hooks";

const HEIGHT = 240;
const PAD = { top: 16, right: 16, bottom: 34, left: 46 };

const line = (points) =>
  points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(" ");

/** Life left as the engine ages. Deliberately plain: one line for our estimate,
 *  one dashed line for what really happened. */
export default function TrendChart({ points, showTruth }) {
  const [wrapRef, width] = useMeasure();
  const [hover, setHover] = useState(null);

  const plot = useMemo(() => {
    if (!width || points.length === 0) return null;

    const innerW = Math.max(10, width - PAD.left - PAD.right);
    const innerH = HEIGHT - PAD.top - PAD.bottom;

    const cycles = points.map((p) => p.cycle);
    const minCycle = Math.min(...cycles);
    const span = Math.max(1, Math.max(...cycles) - minCycle);

    const x = (cycle) =>
      points.length === 1
        ? PAD.left + innerW / 2
        : PAD.left + ((cycle - minCycle) / span) * innerW;
    const y = (rul) =>
      PAD.top + innerH - (Math.min(RUL_CEILING, Math.max(0, rul)) / RUL_CEILING) * innerH;

    const predicted = points.map((p) => ({ ...p, x: x(p.cycle), y: y(p.rul) }));
    const truth = showTruth
      ? points
          .filter((p) => Number.isFinite(p.truth))
          .map((p) => ({ ...p, x: x(p.cycle), y: y(p.truth) }))
      : [];

    const area =
      predicted.length > 1
        ? `${line(predicted)} L ${predicted[predicted.length - 1].x} ${PAD.top + innerH} L ${
            predicted[0].x
          } ${PAD.top + innerH} Z`
        : null;

    const tickCount = Math.min(5, Math.max(1, Math.floor(innerW / 140)));
    const ticks = [
      ...new Set(
        Array.from({ length: tickCount + 1 }, (_, i) =>
          points[Math.round((i * (points.length - 1)) / tickCount)].cycle
        )
      ),
    ];

    return { innerW, innerH, x, y, predicted, truth, area, ticks };
  }, [points, width, showTruth]);

  const onMove = (event) => {
    if (!plot) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const px = event.clientX - rect.left;
    let nearest = plot.predicted[0];
    for (const point of plot.predicted) {
      if (Math.abs(point.x - px) < Math.abs(nearest.x - px)) nearest = point;
    }
    setHover(nearest);
  };

  return (
    <div className="chart" ref={wrapRef}>
      {!plot ? (
        <div className="chart-empty">
          <p>Press “Watch it age” and the engine's life story is drawn here.</p>
        </div>
      ) : (
        <>
          <svg
            viewBox={`0 0 ${width} ${HEIGHT}`}
            width={width}
            height={HEIGHT}
            onMouseMove={onMove}
            onMouseLeave={() => setHover(null)}
          >
            <defs>
              <linearGradient id="trend-area" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--blue)" stopOpacity="0.22" />
                <stop offset="100%" stopColor="var(--blue)" stopOpacity="0.02" />
              </linearGradient>
            </defs>

            {[0, 40, 80, 120].map((value) => (
              <g key={value}>
                <line
                  className="chart-grid"
                  x1={PAD.left}
                  x2={PAD.left + plot.innerW}
                  y1={plot.y(value)}
                  y2={plot.y(value)}
                />
                <text className="chart-axis" x={PAD.left - 10} y={plot.y(value) + 4} textAnchor="end">
                  {value}
                </text>
              </g>
            ))}

            {plot.area && <path d={plot.area} fill="url(#trend-area)" />}

            {plot.truth.length > 1 && <path className="chart-truth" d={line(plot.truth)} />}
            {plot.predicted.length > 1 && <path className="chart-line" d={line(plot.predicted)} />}

            {plot.predicted.slice(-1).map((point) => (
              <circle
                key="head"
                cx={point.x}
                cy={point.y}
                r="6"
                fill={statusOf(point.rul).color}
                stroke="#fff"
                strokeWidth="3"
              />
            ))}

            {plot.ticks.map((tick) => (
              <text
                key={tick}
                className="chart-axis"
                x={plot.x(tick)}
                y={HEIGHT - 12}
                textAnchor="middle"
              >
                {tick}
              </text>
            ))}

            {hover && (
              <>
                <line
                  className="chart-crosshair"
                  x1={hover.x}
                  x2={hover.x}
                  y1={PAD.top}
                  y2={PAD.top + plot.innerH}
                />
                <circle cx={hover.x} cy={hover.y} r="5" fill="var(--blue)" stroke="#fff" strokeWidth="2" />
              </>
            )}
          </svg>

          {hover && (
            <div
              className="chart-tooltip"
              style={{
                left: `${Math.min(Math.max(hover.x, 80), width - 80)}px`,
                top: `${Math.max(4, hover.y - 62)}px`,
              }}
            >
              <span>after {hover.cycle} flights</span>
              <strong>{Math.round(hover.rul)} left</strong>
            </div>
          )}
        </>
      )}

      <div className="chart-labels">
        <span className="chart-y-label">Flights left</span>
        <span className="chart-x-label">Flights already flown →</span>
      </div>

      {plot && (
        <div className="chart-legend">
          <span>
            <i className="swatch-line" /> Our estimate
          </span>
          {plot.truth.length > 1 && (
            <span>
              <i className="swatch-dash" /> What actually happened
            </span>
          )}
        </div>
      )}
    </div>
  );
}
