import { useCallback, useEffect, useRef, useState } from "react";

import "./App.css";
import RulGauge from "./components/RulGauge";
import SensorPanel from "./components/SensorPanel";
import Toast from "./components/Toast";
import TrendChart from "./components/TrendChart";
import engineData from "./data/engines.json";
import { API_BASE, fetchHealth, predict } from "./lib/api";
import { RUL_CEILING, statusOf } from "./lib/sensors";

const SENSORS = engineData.sensors;
const STATS = engineData.stats;

// Four real engines from NASA's test set, picked to span the range from
// nearly-new to almost-failed. Their true remaining life is known.
const ENGINES = [
  { band: "low", emoji: "🟢", name: "Fresh engine", blurb: "Recently installed" },
  { band: "medium", emoji: "🔵", name: "Middle-aged", blurb: "Halfway through life" },
  { band: "high", emoji: "🟠", name: "Getting tired", blurb: "Starting to show wear" },
  { band: "critical", emoji: "🔴", name: "Almost done", blurb: "Close to failing" },
]
  .map((card) => ({ ...card, engine: engineData.engines.find((e) => e.band === card.band) }))
  .filter((card) => card.engine);

const REPLAY_DELAY = 45;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function readingAt(engine, index) {
  const values = engine.readings[index];
  const reading = { cycle: String(engine.cycles[index]) };
  SENSORS.forEach((name, i) => {
    reading[name] = String(values[i]);
  });
  return reading;
}

/** What really happened: the data set records the life left at the final
 *  flight, so earlier flights have that plus the flights in between. */
const truthAt = (engine, index) =>
  Math.min(
    RUL_CEILING,
    engine.trueRul + (engine.cycles[engine.cycles.length - 1] - engine.cycles[index])
  );

const DEFAULT_CARD = ENGINES[0];

export default function App() {
  const [sensorNames, setSensorNames] = useState(SENSORS);
  const [reading, setReading] = useState(() =>
    readingAt(DEFAULT_CARD.engine, DEFAULT_CARD.engine.cycles.length - 1)
  );
  const [connected, setConnected] = useState(null);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [toast, setToast] = useState(null);
  const [series, setSeries] = useState([]);
  const [selected, setSelected] = useState(DEFAULT_CARD.band);
  const [replay, setReplay] = useState({ running: false, paused: false, progress: 0 });

  const replayRef = useRef({ cancelled: true, paused: false });

  useEffect(() => {
    let cancelled = false;

    const poll = () =>
      fetchHealth()
        .then((data) => {
          if (cancelled) return;
          setConnected(data.model_loaded === true);
          if (Array.isArray(data.expected_sensors) && data.expected_sensors.length) {
            setSensorNames(data.expected_sensors);
          }
        })
        .catch(() => {
          if (!cancelled) setConnected(false);
        });

    poll();
    const timer = setInterval(poll, 20000);

    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  useEffect(() => () => {
    replayRef.current.cancelled = true;
  }, []);

  const score = useCallback(
    async (source) => {
      const data = await predict({
        cycle: Number(source.cycle),
        sensors: sensorNames.map((name) => Number(source[name])),
      });
      setResult({ flightsLeft: data.predicted_rul, cycle: Number(source.cycle) });
      return data;
    },
    [sensorNames]
  );

  const checkNow = useCallback(async () => {
    if (busy) return;
    replayRef.current.cancelled = true;
    setReplay({ running: false, paused: false, progress: 0 });
    setBusy(true);
    try {
      const data = await score(reading);
      setSeries([{ cycle: Number(reading.cycle), rul: data.predicted_rul }]);
    } catch (err) {
      setToast({ tone: "error", message: err.message });
    } finally {
      setBusy(false);
    }
  }, [busy, reading, score]);

  const pickEngine = useCallback(
    async (card) => {
      replayRef.current.cancelled = true;
      setReplay({ running: false, paused: false, progress: 0 });

      const engine = card.engine;
      const last = engine.cycles.length - 1;
      const next = readingAt(engine, last);

      setReading(next);
      setSelected(card.band);
      setSeries([]);
      setBusy(true);

      try {
        const data = await score(next);
        setSeries([
          { cycle: engine.cycles[last], rul: data.predicted_rul, truth: truthAt(engine, last) },
        ]);
      } catch (err) {
        setToast({ tone: "error", message: err.message });
      } finally {
        setBusy(false);
      }
    },
    [score]
  );

  // Check the first engine as soon as the backend answers, so the page opens
  // on a real answer instead of an empty dial.
  const started = useRef(false);
  useEffect(() => {
    if (started.current || !connected) return;
    started.current = true;
    pickEngine(DEFAULT_CARD);
  }, [connected, pickEngine]);

  const stopReplay = useCallback(() => {
    replayRef.current.cancelled = true;
    setReplay({ running: false, paused: false, progress: 0 });
  }, []);

  const togglePause = useCallback(() => {
    replayRef.current.paused = !replayRef.current.paused;
    setReplay((current) => ({ ...current, paused: replayRef.current.paused }));
  }, []);

  const watchItAge = useCallback(async () => {
    const card = ENGINES.find((item) => item.band === selected) ?? DEFAULT_CARD;
    const engine = card.engine;

    replayRef.current.cancelled = true;
    await sleep(0);

    const token = { cancelled: false, paused: false };
    replayRef.current = token;

    setSeries([]);
    setResult(null);
    setReplay({ running: true, paused: false, progress: 0 });

    const total = engine.cycles.length;
    const stride = Math.max(1, Math.ceil(total / 110));

    try {
      for (let index = 0; index < total; index += stride) {
        if (token.cancelled) return;
        while (token.paused && !token.cancelled) {
          // eslint-disable-next-line no-await-in-loop
          await sleep(120);
        }
        if (token.cancelled) return;

        const frame = readingAt(engine, index);
        setReading(frame);

        // eslint-disable-next-line no-await-in-loop
        const data = await score(frame);
        if (token.cancelled) return;

        setSeries((current) => [
          ...current,
          {
            cycle: engine.cycles[index],
            rul: data.predicted_rul,
            truth: truthAt(engine, index),
          },
        ]);
        setReplay((current) => ({
          ...current,
          progress: Math.min(1, (index + stride) / total),
        }));

        // eslint-disable-next-line no-await-in-loop
        await sleep(REPLAY_DELAY);
      }

      if (!token.cancelled) {
        setToast({
          tone: "ok",
          message: `This engine really had ${engine.trueRul} flights left at the end.`,
        });
      }
    } catch (err) {
      if (!token.cancelled) setToast({ tone: "error", message: err.message });
    } finally {
      if (!token.cancelled) setReplay({ running: false, paused: false, progress: 1 });
    }
  }, [selected, score]);

  const status = result ? statusOf(result.flightsLeft) : null;
  const activeCard = ENGINES.find((item) => item.band === selected) ?? null;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            ✈️
          </span>
          <div>
            <strong>Engine Health Check</strong>
            <span>Know when a jet engine needs servicing</span>
          </div>
        </div>

        <span className={`status-pill ${connected ? "is-on" : "is-off"}`}>
          <i />
          {connected === null ? "Connecting…" : connected ? "Connected" : "Not connected"}
        </span>
      </header>

      {connected === false && (
        <div className="banner">
          The prediction service isn't running. Open a terminal and run{" "}
          <code>.venv\Scripts\activate</code> then <code>python -m src.api</code>{" "}
          — this page will reconnect on its own.
        </div>
      )}

      <main className="container">
        <section className="hero">
          <h1>
            How much life is left in <span>this engine?</span>
          </h1>
          <p>
            Pick an aircraft engine below. We read its sensors and tell you how many more
            flights it can safely make before it needs maintenance.
          </p>
        </section>

        <section className="step">
          <h2>
            <span className="step-number">1</span> Pick an engine
          </h2>

          <div className="engine-cards">
            {ENGINES.map((card) => (
              <button
                key={card.band}
                type="button"
                className={`engine-card ${selected === card.band ? "is-selected" : ""}`}
                onClick={() => pickEngine(card)}
                disabled={replay.running}
              >
                <span className="engine-emoji">{card.emoji}</span>
                <span className="engine-name">{card.name}</span>
                <span className="engine-blurb">{card.blurb}</span>
              </button>
            ))}
          </div>
        </section>

        <section className="step">
          <h2>
            <span className="step-number">2</span> See the answer
          </h2>

          <div
            className={`answer-card ${status ? `tone-${status.key}` : ""}`}
            style={status ? { background: status.soft, borderColor: status.color } : undefined}
          >
            <RulGauge
              flightsLeft={result ? result.flightsLeft : NaN}
              pending={busy || replay.running}
            />

            <div className="answer-text">
              {status ? (
                <>
                  <span className="answer-emoji">{status.emoji}</span>
                  <h3 style={{ color: status.color }}>{status.title}</h3>
                  <p>{status.line}</p>
                  {activeCard && (
                    <p className="answer-caption">
                      {activeCard.name} · checked after {result.cycle} flights flown
                    </p>
                  )}
                </>
              ) : (
                <>
                  <h3>Checking…</h3>
                  <p>Reading the engine's sensors and comparing them with 100 real engines.</p>
                </>
              )}

              <div className="answer-actions">
                {!replay.running ? (
                  <button
                    type="button"
                    className="btn btn-primary"
                    onClick={watchItAge}
                    disabled={busy || !connected}
                  >
                    ▶ Watch it age
                  </button>
                ) : (
                  <>
                    <button type="button" className="btn btn-primary" onClick={togglePause}>
                      {replay.paused ? "▶ Continue" : "❚❚ Pause"}
                    </button>
                    <button type="button" className="btn btn-soft" onClick={stopReplay}>
                      Stop
                    </button>
                  </>
                )}
                <span className="answer-hint">
                  Replays this engine's whole life, flight by flight.
                </span>
              </div>

              {replay.running && (
                <div className="progress">
                  <i style={{ width: `${replay.progress * 100}%` }} />
                </div>
              )}
            </div>
          </div>
        </section>

        <section className="step">
          <h2>
            <span className="step-number">3</span> Watch it wear out
          </h2>

          <div className="card chart-card">
            <p className="card-lead">
              The line falls as the engine ages. When it reaches the bottom, the engine is
              done.
            </p>
            <TrendChart points={series} showTruth={series.length > 1} />
          </div>
        </section>

        <section className="how">
          <div className="how-item">
            <span>📡</span>
            <h4>Sensors are recorded</h4>
            <p>Temperature, pressure and speed are logged on every single flight.</p>
          </div>
          <div className="how-item">
            <span>🤖</span>
            <h4>The AI compares</h4>
            <p>It matches the readings against 100 real engines that were run until failure.</p>
          </div>
          <div className="how-item">
            <span>🔧</span>
            <h4>You get a clear call</h4>
            <p>Flights left, and whether to fly on, keep watching, or book a service.</p>
          </div>
        </section>

        <details className="details">
          <summary>See the sensor readings</summary>

          <div className="details-body">
            <label className="cycle-field">
              <span>Flights flown so far</span>
              <input
                type="number"
                min="1"
                value={reading.cycle}
                disabled={replay.running}
                onChange={(event) =>
                  setReading((current) => ({ ...current, cycle: event.target.value }))
                }
              />
            </label>

            <SensorPanel
              sensorNames={sensorNames}
              reading={reading}
              stats={STATS}
              onChange={(name, value) =>
                setReading((current) => ({ ...current, [name]: value }))
              }
              disabled={replay.running}
            />

            <button
              type="button"
              className="btn btn-primary"
              onClick={checkNow}
              disabled={busy || replay.running || !connected}
            >
              {busy ? "Checking…" : "Check these readings"}
            </button>

            <p className="details-note">
              Data: NASA C-MAPSS FD001. The estimate is usually within about 21 flights of
              what actually happened.
            </p>
          </div>
        </details>

        <footer className="footer">
          Built on NASA's turbofan engine data · predictions are capped at {RUL_CEILING}{" "}
          flights · <span className="footer-api">{API_BASE.replace(/^https?:\/\//, "")}</span>
        </footer>
      </main>

      <Toast toast={toast} onDismiss={() => setToast(null)} />
    </div>
  );
}
