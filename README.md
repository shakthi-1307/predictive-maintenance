# Predictive Maintenance — NASA C-MAPSS FD001

Remaining Useful Life (RUL) prediction for turbofan engines, with a
FastAPI service and a React dashboard.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows;  source .venv/bin/activate on Unix
pip install -r requirements.txt
```

The pinned versions matter: `pandas`/`scikit-learn`/`matplotlib` wheels are
built against a specific NumPy ABI, and mixing a NumPy 2.x install with
packages compiled for 1.x fails at import with
`numpy.dtype size changed`.

## Pipeline

```bash
python run.py
```

runs, in order:

| Step | Module | Output |
| --- | --- | --- |
| Validate raw data | `src.validation` | — |
| Build RUL targets | `src.preprocessing` | `data/processed/train_rul.csv` |
| Engineer features | `src.features` | `data/processed/train_features.csv` |
| Train Random Forest | `src.train_baseline` | `models/random_forest_rul.joblib` |
| Score official test set | `src.evaluate_test` | `reports/metrics/official_test_*` |

Optional extras: `python -m src.eda`, `python -m src.train_xgboost`,
`python -m src.train_lstm`, `python -m src.explainability`.

Current scores live in `reports/metrics/` — read them from there rather
than copying numbers into docs or UI.

### Truncated trajectories

`src.preprocessing` cuts each engine's run-to-failure history at 60/70/80/90/100%
of its life. Each cut becomes its own trajectory with a distinct `unit`
id, while `source_unit` records the physical engine it came from.

Two things depend on this:

- **`cycle_ratio`** is `cycle / last observed cycle`, so it reaches 1.0 at
  a wide spread of true RULs — the same thing the official test set looks
  like, where each engine is observed up to some arbitrary cycle. Training
  on full histories alone would teach the model that `cycle_ratio == 1.0`
  means failure is imminent.
- **Train/validation splits must group on `source_unit`**, never `unit`.
  Trajectories cut from one engine overlap almost completely, so splitting
  on `unit` leaks the validation set into training.

## API

```bash
python -m src.api            # http://localhost:8000  (docs at /docs)
```

`POST /predict` takes one reading:

```json
{ "cycle": 31, "sensors": [642.58, 1581.22, "... 14 values total"] }
```

The `sensors` array is ordered by `src.config.DEFAULT_USEFUL_SENSORS`;
`GET /health` reports that list as `expected_sensors` along with whether a
model is loaded.

Because a single reading carries no history, difference features are 0 and
rolling statistics collapse to the current value. Columns the reading
cannot supply fall back to training medians stored in the model checkpoint.
For accuracy comparable to `src.evaluate_test`, feed full engine histories
through `src.features` instead.

## Dashboard

```bash
cd frontend
npm install
npm run dev                  # http://localhost:5173
```

Point it at a different backend with `VITE_API_BASE`. The API allows CORS
from ports 5173 and 4173 only — add your origin in `src/api.py` if you
serve it elsewhere.

## Tests

```bash
pytest
```

## Docker

```bash
docker compose up --build    # serves src.api on :8000
```

The image needs `models/random_forest_rul.joblib` to exist, so run the
pipeline before building. Without it the service still starts and reports
the problem on `/health`.
