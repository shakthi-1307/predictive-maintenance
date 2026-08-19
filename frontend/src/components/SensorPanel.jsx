import { describeSensor, wearFraction } from "../lib/sensors";

function wearColor(fraction) {
  if (fraction === null) return "var(--blue)";
  if (fraction < 0.4) return "var(--mint)";
  if (fraction < 0.75) return "var(--amber)";
  return "var(--coral)";
}

function SensorRow({ name, value, stat, onChange, disabled }) {
  const meta = describeSensor(name);
  const numeric = Number(value);
  const wear = wearFraction(numeric, stat);

  return (
    <div className="sensor-row">
      <span className="sensor-name">
        {meta.label}
        {meta.unit && <em> ({meta.unit})</em>}
      </span>

      <span className="sensor-wear">
        <i
          style={{
            width: `${(wear ?? 0) * 100}%`,
            background: wearColor(wear),
          }}
        />
      </span>

      <input
        className="sensor-input"
        type="number"
        step="any"
        value={value ?? ""}
        disabled={disabled}
        aria-label={meta.label}
        onChange={(event) => onChange(name, event.target.value)}
      />
    </div>
  );
}

/** Only shown when someone opens "See the sensor readings". */
export default function SensorPanel({ sensorNames, reading, stats, onChange, disabled }) {
  return (
    <div className="sensor-list">
      <p className="sensor-legend">
        The bar shows how worn each part is — <b>short and green</b> is like new,{" "}
        <b>long and red</b> is worn out.
      </p>

      {sensorNames.map((name) => (
        <SensorRow
          key={name}
          name={name}
          value={reading[name]}
          stat={stats[name]}
          onChange={onChange}
          disabled={disabled}
        />
      ))}
    </div>
  );
}
