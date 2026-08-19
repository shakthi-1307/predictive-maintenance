from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health():

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    assert (
        response.json()["status"]
        == "healthy"
    )


def test_prediction():

    payload = {
        "machine_age": 7.5,
        "temperature": 75,
        "pressure": 90,
        "vibration": 3.5,
        "rotational_speed": 1350,
        "torque": 52,
        "voltage": 210,
        "current": 16,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        "failure_probability"
        in data
    )

    assert (
        "failure_prediction"
        in data
    )

    assert "status" in data