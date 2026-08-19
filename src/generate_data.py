import numpy as np
import pandas as pd

from src.config import (
    NUM_MACHINES,
    TIMESTEPS_PER_MACHINE,
    RAW_DATA_PATH,
    RANDOM_STATE,
    create_directories,
)


def generate_machine_data(
    machine_id: int,
    timesteps: int,
    rng: np.random.Generator,
):
    rows = []

    machine_age = rng.uniform(1, 10)

    # Each machine has its own degradation behavior
    degradation_start = rng.integers(
        low=int(timesteps * 0.55),
        high=int(timesteps * 0.80),
    )

    degradation_rate = rng.uniform(0.015, 0.035)

    for t in range(timesteps):

        degradation = max(
            0.0,
            (t - degradation_start) * degradation_rate
        )

        degradation = min(degradation, 1.0)

        temperature = (
            65
            + 8 * degradation
            + rng.normal(0, 2)
        )

        pressure = (
            100
            - 12 * degradation
            + rng.normal(0, 2)
        )

        vibration = (
            1.5
            + 2.5 * degradation
            + rng.normal(0, 0.25)
        )

        rotational_speed = (
            1500
            - 120 * degradation
            + rng.normal(0, 20)
        )

        torque = (
            45
            + 10 * degradation
            + rng.normal(0, 2)
        )

        voltage = (
            220
            - 8 * degradation
            + rng.normal(0, 2)
        )

        current = (
            12
            + 5 * degradation
            + rng.normal(0, 0.5)
        )

        rows.append(
            {
                "machine_id": machine_id,
                "timestamp": t,
                "machine_age": machine_age,
                "temperature": temperature,
                "pressure": pressure,
                "vibration": vibration,
                "rotational_speed": rotational_speed,
                "torque": torque,
                "voltage": voltage,
                "current": current,
                "degradation": degradation,
            }
        )

    return rows


def create_failure_labels(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # Failure risk based on degradation.
    # We deliberately create failures only near the end
    # of the degradation cycle.
    df["failure"] = (
        (df["degradation"] > 0.78)
        & (
            df.groupby("machine_id")["degradation"]
            .transform("max") > 0.90
        )
    ).astype(int)

    # Remove latent variable from final modeling dataset.
    df.drop(columns=["degradation"], inplace=True)

    return df


def main():

    create_directories()

    rng = np.random.default_rng(RANDOM_STATE)

    all_rows = []

    for machine_id in range(1, NUM_MACHINES + 1):

        machine_rows = generate_machine_data(
            machine_id=machine_id,
            timesteps=TIMESTEPS_PER_MACHINE,
            rng=rng,
        )

        all_rows.extend(machine_rows)

    df = pd.DataFrame(all_rows)

    df = create_failure_labels(df)

    # Shuffle only after generating time-series data.
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    df.to_csv(RAW_DATA_PATH, index=False)

    print("=" * 60)
    print("DATASET GENERATED")
    print("=" * 60)
    print(f"Rows: {len(df):,}")
    print(f"Machines: {df['machine_id'].nunique()}")
    print(f"Failure samples: {df['failure'].sum():,}")
    print(
        f"Failure rate: {df['failure'].mean() * 100:.2f}%"
    )
    print(f"Saved to: {RAW_DATA_PATH}")


if __name__ == "__main__":
    main()