import pytest

from fastapi.testclient import TestClient

from src.api import (
    app,
    model,
    build_single_prediction_features,
    FEATURE_COLUMNS,
    FEATURE_DEFAULTS,
    SensorReading,
    USEFUL_SENSORS,
)


client = TestClient(app)


needs_model = pytest.mark.skipif(
    model is None,
    reason=(
        "random_forest_rul.joblib is missing; "
        "run `python run.py` to train it."
    ),
)


# Final observed reading of engine 1 in test_FD001.txt (cycle 31),
# ordered to match USEFUL_SENSORS.
SAMPLE_CYCLE = 31

SAMPLE_SENSORS = [
    642.58,    # s2
    1581.22,   # s3
    1398.91,   # s4
    554.42,    # s7
    2388.08,   # s8
    9056.40,   # s9
    47.23,     # s11
    521.79,    # s12
    2388.06,   # s13
    8130.11,   # s14
    8.4024,    # s15
    393.0,     # s17
    38.81,     # s20
    23.3552,   # s21
]


def sample_payload(cycle=SAMPLE_CYCLE):

    return {
        "cycle": cycle,
        "sensors": SAMPLE_SENSORS,
    }


def test_sample_matches_expected_sensor_count():
    """Guards the fixture against drift in USEFUL_SENSORS."""

    assert len(SAMPLE_SENSORS) == len(USEFUL_SENSORS)


def test_root():

    response = client.get("/")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "running"

    assert body["dataset"] == "NASA C-MAPSS FD001"


def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["model_loaded"] is (model is not None)

    assert body["expected_sensors"] == USEFUL_SENSORS


@needs_model
def test_prediction():

    response = client.post(
        "/predict",
        json=sample_payload(),
    )

    assert response.status_code == 200

    data = response.json()

    assert "predicted_rul" in data

    assert "risk_level" in data

    assert "maintenance_recommendation" in data

    # The model is trained against RUL capped at 125.
    assert 0 <= data["predicted_rul"] <= 125

    assert data["risk_level"] in {
        "CRITICAL",
        "HIGH",
        "MEDIUM",
        "LOW",
    }


@needs_model
def test_every_sensor_reaches_the_model():
    """
    Regression test for the bug where the API declared 13 sensors
    while the model was trained on 14. The extra sensor and all of
    its derived columns were silently filled in, so one real
    degradation signal never reached the model at all.
    """

    features = build_single_prediction_features(
        SensorReading(**sample_payload())
    )

    row = features.iloc[0]

    for index, name in enumerate(USEFUL_SENSORS):

        assert name in FEATURE_COLUMNS, (
            f"{name} is offered by the API but the model was "
            f"never trained on it"
        )

        assert row[name] == pytest.approx(SAMPLE_SENSORS[index]), (
            f"{name} did not survive into the feature vector"
        )


@needs_model
def test_derived_columns_are_computed_not_filled():
    """
    Every column derived from a supplied sensor must be computed.
    Only the operating settings and the flat FD001 sensors may fall
    back to a stored default.
    """

    features = build_single_prediction_features(
        SensorReading(**sample_payload())
    )

    computed = set(features.columns) - set(FEATURE_DEFAULTS)

    for name in USEFUL_SENSORS:

        derived = [
            column
            for column in FEATURE_COLUMNS
            if column == name or column.startswith(f"{name}_")
        ]

        assert derived, f"no features derived from {name}"

        missing = [
            column
            for column in derived
            if column not in features.columns
        ]

        assert not missing, (
            f"{name} features never built: {missing}"
        )

    assert computed, "nothing was computed from the request at all"


@needs_model
def test_unsupplied_columns_use_training_defaults():
    """
    A single reading cannot supply the operating settings or the
    flat FD001 sensors. Those must fall back to the training
    median, not to zero -- setting_3 is always 100.0 in training,
    so a zero there is far outside the range the model ever saw.
    """

    features = build_single_prediction_features(
        SensorReading(**sample_payload())
    )

    row = features.iloc[0]

    assert FEATURE_DEFAULTS, (
        "checkpoint carries no feature_defaults; retrain with "
        "`python -m src.train_baseline`"
    )

    assert row["setting_3"] == pytest.approx(
        FEATURE_DEFAULTS["setting_3"]
    )

    assert row["s1"] == pytest.approx(FEATURE_DEFAULTS["s1"])


def test_rejects_wrong_sensor_count():

    response = client.post(
        "/predict",
        json={
            "cycle": 100,
            "sensors": SAMPLE_SENSORS[:-1],
        },
    )

    assert response.status_code == 422


def test_rejects_invalid_cycle():

    response = client.post(
        "/predict",
        json=sample_payload(cycle=0),
    )

    assert response.status_code == 422
