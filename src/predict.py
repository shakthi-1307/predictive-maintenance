import joblib
import pandas as pd

from src.config import (
    RF_MODEL_PATH,
    SCALER_PATH,
)


class Predictor:

    def __init__(self):

        self.model = joblib.load(
            RF_MODEL_PATH
        )

        self.scaler = joblib.load(
            SCALER_PATH
        )

    def predict(self, data):

        if isinstance(data, dict):

            df = pd.DataFrame(
                [data]
            )

        else:

            df = pd.DataFrame(data)

        features = self._prepare_features(
            df
        )

        scaled = self.scaler.transform(
            features
        )

        probability = (
            self.model.predict_proba(
                scaled
            )[:, 1]
        )

        prediction = (
            probability >= 0.5
        ).astype(int)

        return [
            {
                "failure_probability": float(
                    probability[i]
                ),
                "failure_prediction": int(
                    prediction[i]
                ),
                "status": (
                    "FAILURE_RISK"
                    if prediction[i] == 1
                    else "NORMAL"
                ),
            }
            for i in range(len(df))
        ]

    def _prepare_features(self, df):

        sensors = [
            "temperature",
            "pressure",
            "vibration",
            "rotational_speed",
            "torque",
            "voltage",
            "current",
        ]

        features = df[
            sensors
            + ["machine_age"]
        ].copy()

        # For online inference we don't have
        # historical windows, so initialize
        # temporal statistics from current
        # sensor values.
        for sensor in sensors:

            features[
                f"{sensor}_rolling_mean"
            ] = df[sensor]

            features[
                f"{sensor}_rolling_std"
            ] = 0.0

            features[
                f"{sensor}_diff"
            ] = 0.0

        features[
            "age_temperature_interaction"
        ] = (
            df["machine_age"]
            * df["temperature"]
        )

        features[
            "vibration_torque_interaction"
        ] = (
            df["vibration"]
            * df["torque"]
        )

        ordered_columns = sensors + [
            "machine_age",
            "age_temperature_interaction",
            "vibration_torque_interaction",
        ]

        for sensor in sensors:

            ordered_columns.extend(
                [
                    f"{sensor}_rolling_mean",
                    f"{sensor}_rolling_std",
                    f"{sensor}_diff",
                ]
            )

        return features[
            ordered_columns
        ]