import { RUL_CEILING, statusOf } from "../lib/sensors";
import { useCountUp } from "../lib/hooks";

const SIZE = 240;
const CENTER = SIZE / 2;
const RADIUS = 96;
const WIDTH = 22;
const START_ANGLE = -130;
const SWEEP = 260;

const polar = (radius, angle) => {
  const rad = ((angle - 90) * Math.PI) / 180;
  return [CENTER + radius * Math.cos(rad), CENTER + radius * Math.sin(rad)];
};

const arc = (radius, from, to) => {
  const [x1, y1] = polar(radius, from);
  const [x2, y2] = polar(radius, to);
  return `M ${x1} ${y1} A ${radius} ${radius} 0 ${
    Math.abs(to - from) > 180 ? 1 : 0
  } 1 ${x2} ${y2}`;
};

/** A single dial: how much life is left, as a share of a full-life engine. */
export default function RulGauge({ flightsLeft, pending }) {
  const known = Number.isFinite(flightsLeft);
  const value = known ? flightsLeft : 0;
  const animated = useCountUp(value, 900);
  const status = statusOf(value);

  const fraction = Math.min(1, Math.max(0, animated / RUL_CEILING));
  const endAngle = START_ANGLE + fraction * SWEEP;

  return (
    <div className={`gauge ${pending ? "is-pending" : ""}`}>
      <svg viewBox={`0 0 ${SIZE} ${SIZE}`} role="img"
        aria-label={known ? `About ${Math.round(value)} flights left` : "No result yet"}>
        <path className="gauge-track" d={arc(RADIUS, START_ANGLE, START_ANGLE + SWEEP)} strokeWidth={WIDTH} />
        {known && fraction > 0.005 && (
          <path
            className="gauge-fill"
            d={arc(RADIUS, START_ANGLE, endAngle)}
            stroke={status.color}
            strokeWidth={WIDTH}
          />
        )}
      </svg>

      <div className="gauge-readout">
        {known ? (
          <>
            <span className="gauge-about">about</span>
            <strong style={{ color: status.color }}>{Math.round(animated)}</strong>
            <span className="gauge-unit">flights left</span>
          </>
        ) : (
          <>
            <strong className="gauge-idle">—</strong>
            <span className="gauge-unit">waiting</span>
          </>
        )}
      </div>
    </div>
  );
}
