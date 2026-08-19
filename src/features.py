import pandas as pd
import numpy as np

from src.config import (
    RAW_DATA_PATH,
    FEATURE_DATA_PATH,
    SENSOR_COLUMNS,
    create_directories,
)


def create_features(df: pd.DataFrame):

    df = df.copy()

    # Always sort before time-series feature engineering.
    df = df.sort_values(
        ["machine_id", "timestamp"]
    ).reset_index(drop=True)

    grouped = df.groupby("machine_id")

    feature_columns = []

    for sensor in SENSOR_COLUMNS:

        # Rolling statistics
        rolling_mean = (
            grouped[sensor]
            .rolling(window=12, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        rolling_std = (
            grouped[sensor]
            .rolling(window=12, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
            .fillna(0)
        )

        df[f"{sensor}_rolling_mean"] = rolling_mean
        df[f"{sensor}_rolling_std"] = rolling_std

        # Rate of change
        df[f"{sensor}_diff"] = (
            grouped[sensor]
            .diff()
            .fillna(0)
        )

        feature_columns.extend(
            [
                f"{sensor}_rolling_mean",
                f"{sensor}_rolling_std",
                f"{sensor}_diff",
            ]
        )

    # Machine age interactions
    df["age_temperature_interaction"] = (
        df["machine_age"]
        * df["temperature"]
    )

    df["vibration_torque_interaction"] = (
        df["vibration"]
        * df["torque"]
    )

    feature_columns.extend(
        [
            "machine_age",
            "age_temperature_interaction",
            "vibration_torque_interaction",
        ]
    )

    feature_columns.extend(SENSOR_COLUMNS)

    return df, feature_columns


def main():

    create_directories()

    df = pd.read_csv(RAW_DATA_PATH)

    df, feature_columns = create_features(df)

    df.to_csv(FEATURE_DATA_PATH, index=False)

    print("=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 60)
    print(f"Rows: {len(df):,}")
    print(f"Features: {len(feature_columns)}")

    print("\nFeature list:")
    for feature in feature_columns:
        print(f"  - {feature}")

    print(f"\nSaved to: {FEATURE_DATA_PATH}")


if __name__ == "__main__":
    main()