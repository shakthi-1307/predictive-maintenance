import joblib
import numpy as np
import pandas as pd

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import (
    RF_MODEL_PATH,
    DEFAULT_USEFUL_SENSORS,
)
from src.features import WINDOWS


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Predictive Maintenance API",
    description=(
        "Remaining Useful Life prediction API "
        "for NASA C-MAPSS FD001 turbofan engines."
    ),
    version="1.0.0",
)


# The dashboard in frontend/ is served from a different origin
# (Vite dev server on :5173), so the browser needs CORS headers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD MODEL
# ============================================================
#
# A missing model must not stop the process from starting --
# otherwise /health can never report the problem, and the
# container just crash-loops with no diagnosis.

model = None

FEATURE_COLUMNS: List[str] = []

FEATURE_DEFAULTS: dict = {}

MODEL_ERROR: Optional[str] = None

if not RF_MODEL_PATH.exists():

    MODEL_ERROR = (
        f"Model not found: {RF_MODEL_PATH}. "
        "Run `python run.py` to train it."
    )

else:

    checkpoint = joblib.load(
        RF_MODEL_PATH
    )

    model = checkpoint["model"]

    FEATURE_COLUMNS = checkpoint["features"]

    # Training medians, used for the columns a single reading
    # cannot supply. Older checkpoints predate this key.
    FEATURE_DEFAULTS = checkpoint.get(
        "feature_defaults",
        {},
    )


# ============================================================
# USEFUL SENSOR CONFIGURATION
# ============================================================
#
# These are the sensors src/features.py derives diff/rolling
# features from. A sensor the request cannot supply is filled in
# below and the model reads it as a flat signal, so the list here
# and the list the model was fitted on must not drift apart.

USEFUL_SENSORS = DEFAULT_USEFUL_SENSORS

SENSOR_COUNT = len(USEFUL_SENSORS)


if model is not None:

    # Every configured sensor must appear in the trained feature
    # set. Otherwise the API would keep answering, confidently, on
    # a feature vector the model never saw.
    unknown_sensors = [
        sensor
        for sensor in USEFUL_SENSORS
        if sensor not in FEATURE_COLUMNS
    ]

    if unknown_sensors:

        model = None

        MODEL_ERROR = (
            f"Model at {RF_MODEL_PATH} was not trained on "
            f"{unknown_sensors}. It is out of step with "
            f"DEFAULT_USEFUL_SENSORS; retrain with `python run.py`."
        )


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class SensorReading(BaseModel):

    cycle: int = Field(
        ...,
        ge=1,
        description="Current engine cycle",
    )

    sensors: List[float] = Field(
        ...,
        min_length=SENSOR_COUNT,
        max_length=SENSOR_COUNT,
        description=(
            "Sensor values ordered as "
            + ",".join(USEFUL_SENSORS)
        ),
    )


class PredictionResponse(BaseModel):

    predicted_rul: float

    risk_level: str

    maintenance_recommendation: str

    model: str

    features_used: int


# ============================================================
# FEATURE CONSTRUCTION
# ============================================================

def build_single_prediction_features(
    reading: SensorReading,
):
    """
    Construct the feature vector expected by the trained
    Random Forest model.

    Because the API receives a single observation rather than
    a full engine history, rolling statistics are approximated
    using the current sensor state.
    """

    values = reading.sensors

    if len(values) != SENSOR_COUNT:

        raise ValueError(
            f"Exactly {SENSOR_COUNT} sensor values are "
            f"required, ordered as "
            f"{','.join(USEFUL_SENSORS)}."
        )

    row = {}

    # --------------------------------------------------------
    # Current sensor values
    # --------------------------------------------------------

    sensor_map = dict(
        zip(
            USEFUL_SENSORS,
            values,
        )
    )

    for sensor in USEFUL_SENSORS:

        row[sensor] = (
            sensor_map[sensor]
        )

    # --------------------------------------------------------
    # First-order differences
    #
    # With only one observation available through the API,
    # difference features are initialized to zero.
    # --------------------------------------------------------

    for sensor in USEFUL_SENSORS:

        row[
            f"{sensor}_diff"
        ] = 0.0

    # --------------------------------------------------------
    # Rolling statistics
    #
    # Single-observation inference approximation.
    # --------------------------------------------------------

    for sensor in USEFUL_SENSORS:

        value = sensor_map[sensor]

        for window in WINDOWS:

            row[
                f"{sensor}_rolling_mean_{window}"
            ] = value

            row[
                f"{sensor}_rolling_std_{window}"
            ] = 0.0

    # --------------------------------------------------------
    # Cycle features
    #
    # Training defines cycle_ratio as cycle / last observed
    # cycle of the trajectory (src/features.py). The reading
    # being scored IS the last observed cycle, so the ratio is
    # 1.0 -- the same value src/evaluate_test.py produces for
    # the final row of every official test engine.
    # --------------------------------------------------------

    row["cycle_ratio"] = 1.0

    # --------------------------------------------------------
    # Cross-sensor degradation features
    # --------------------------------------------------------

    row["thermal_load"] = (
        sensor_map["s4"]
        * sensor_map["s15"]
    )

    row["mechanical_stress"] = (
        sensor_map["s11"]
        * sensor_map["s20"]
    )

    row["pressure_ratio"] = (
        sensor_map["s4"]
        / (
            sensor_map["s3"]
            + 1e-8
        )
    )

    row["sensor_mean"] = float(
        np.mean(values)
    )

    # ddof=1 to match pandas DataFrame.std(axis=1), which is what
    # src/features.py uses. np.std defaults to ddof=0 and would
    # hand the model a systematically smaller number than it saw
    # during training.
    row["sensor_std"] = float(
        np.std(
            values,
            ddof=1,
        )
    )

    # --------------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------------

    features = pd.DataFrame(
        [row]
    )

    # --------------------------------------------------------
    # Match EXACT training feature order
    # --------------------------------------------------------

    # Columns a single reading cannot produce (operating settings
    # and the FD001 sensors that carry no signal) fall back to the
    # training median rather than to zero.
    for feature in FEATURE_COLUMNS:

        if feature not in features.columns:

            features[
                feature
            ] = FEATURE_DEFAULTS.get(
                feature,
                0.0,
            )

    features = features[
        FEATURE_COLUMNS
    ]

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    features = features.fillna(
        0.0
    )

    return features


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def classify_risk(
    rul: float,
):

    if rul <= 20:

        return (
            "CRITICAL",
            "Immediate maintenance recommended.",
        )

    if rul <= 50:

        return (
            "HIGH",
            "Schedule maintenance soon.",
        )

    if rul <= 80:

        return (
            "MEDIUM",
            "Continue monitoring engine health.",
        )

    return (
        "LOW",
        "Engine appears healthy; continue monitoring.",
    )


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "service":
            "Predictive Maintenance API",

        "status":
            "running",

        "model":
            "Random Forest RUL",

        "dataset":
            "NASA C-MAPSS FD001",

        "features":
            len(FEATURE_COLUMNS),
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": (
            "healthy"
            if model is not None
            else "degraded"
        ),
        "model_loaded": model is not None,
        "error": MODEL_ERROR,
        "features": len(
            FEATURE_COLUMNS
        ),
        "expected_sensors": USEFUL_SENSORS,
    }


# ============================================================
# RUL PREDICTION
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    reading: SensorReading,
):

    if model is None:

        raise HTTPException(
            status_code=503,
            detail=MODEL_ERROR,
        )

    try:

        features = (
            build_single_prediction_features(
                reading
            )
        )

        prediction = model.predict(
            features
        )[0]

        # ----------------------------------------------------
        # Keep RUL within the training range.
        # ----------------------------------------------------

        predicted_rul = float(
            np.clip(
                prediction,
                0,
                125,
            )
        )

        risk_level, recommendation = (
            classify_risk(
                predicted_rul
            )
        )

        return PredictionResponse(
            predicted_rul=round(
                predicted_rul,
                2,
            ),

            risk_level=risk_level,

            maintenance_recommendation=(
                recommendation
            ),

            model=(
                "Random Forest Regressor"
            ),

            features_used=len(
                FEATURE_COLUMNS
            ),
        )

    except ValueError as exc:

        # Bad input, not a server fault.
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )