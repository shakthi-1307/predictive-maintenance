from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import Predictor


app = FastAPI(
    title="Predictive Maintenance API",
    description=(
        "Machine failure prediction API "
        "powered by a trained Random Forest model."
    ),
    version="1.0.0",
)


predictor = Predictor()


class SensorInput(BaseModel):

    machine_age: float = Field(
        ...,
        ge=0,
        le=100,
    )

    temperature: float
    pressure: float
    vibration: float = Field(
        ...,
        ge=0,
    )

    rotational_speed: float = Field(
        ...,
        ge=0,
    )

    torque: float
    voltage: float
    current: float = Field(
        ...,
        ge=0,
    )


class PredictionResponse(BaseModel):

    failure_probability: float
    failure_prediction: int
    status: str


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "predictive-maintenance",
        "model": "random-forest",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    sensor_data: SensorInput,
):

    try:

        result = predictor.predict(
            sensor_data.model_dump()
        )[0]

        return result

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@app.post(
    "/predict/batch",
    response_model=List[
        PredictionResponse
    ],
)
def batch_predict(
    sensor_data: List[SensorInput],
):

    if not sensor_data:

        raise HTTPException(
            status_code=400,
            detail="Empty batch.",
        )

    try:

        records = [
            item.model_dump()
            for item in sensor_data
        ]

        return predictor.predict(
            records
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )