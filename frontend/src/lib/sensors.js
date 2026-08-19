// Human-readable names for the FD001 sensor channels the model uses.
// Source: NASA C-MAPSS documentation (Saxena & Goebel, 2008).
export const SENSOR_META = {
  s2: { label: "Low-pressure compressor temperature", unit: "°R" },
  s3: { label: "High-pressure compressor temperature", unit: "°R" },
  s4: { label: "Turbine outlet temperature", unit: "°R" },
  s7: { label: "Compressor outlet pressure", unit: "psi" },
  s8: { label: "Fan speed", unit: "rpm" },
  s9: { label: "Core speed", unit: "rpm" },
  s11: { label: "Compressor static pressure", unit: "psi" },
  s12: { label: "Fuel flow ratio", unit: "" },
  s13: { label: "Corrected fan speed", unit: "rpm" },
  s14: { label: "Corrected core speed", unit: "rpm" },
  s15: { label: "Bypass ratio", unit: "" },
  s17: { label: "Bleed enthalpy", unit: "" },
  s20: { label: "High-pressure turbine coolant bleed", unit: "lb/s" },
  s21: { label: "Low-pressure turbine coolant bleed", unit: "lb/s" },
};

export const describeSensor = (name) =>
  SENSOR_META[name] ?? { label: name, unit: "" };

export const RUL_CEILING = 125;

/**
 * Plain-language status. Thresholds mirror classify_risk() in src/api.py, but
 * the wording is written for someone who has never heard of "RUL".
 */
export const STATUSES = [
  {
    key: "critical",
    max: 20,
    title: "Service now",
    line: "This engine is at the end of its life. Take it off the schedule and service it.",
    emoji: "🛑",
    color: "var(--coral)",
    soft: "var(--coral-soft)",
  },
  {
    key: "high",
    max: 50,
    title: "Book a service",
    line: "Wear is clear. Get maintenance on the calendar in the next few weeks.",
    emoji: "🟠",
    color: "var(--amber)",
    soft: "var(--amber-soft)",
  },
  {
    key: "medium",
    max: 80,
    title: "Keep an eye on it",
    line: "Still fine to fly, but the engine is past its prime. Check again soon.",
    emoji: "👀",
    color: "var(--blue)",
    soft: "var(--blue-soft)",
  },
  {
    key: "low",
    max: Infinity,
    title: "Healthy",
    line: "Plenty of life left. Nothing to do right now.",
    emoji: "✅",
    color: "var(--mint)",
    soft: "var(--mint-soft)",
  },
];

export const statusOf = (flightsLeft) =>
  STATUSES.find((status) => flightsLeft <= status.max) ?? STATUSES[STATUSES.length - 1];

/** How worn a reading is: 0 = like new, 1 = end of life. */
export function wearFraction(value, stat) {
  if (!stat || !Number.isFinite(value)) return null;
  const span = stat.worn - stat.healthy;
  if (Math.abs(span) < 1e-9) return null;
  return Math.min(1, Math.max(0, (value - stat.healthy) / span));
}
