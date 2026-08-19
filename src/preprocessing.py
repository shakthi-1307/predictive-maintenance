import pandas as pd
import numpy as np

from src.config import (
    PROCESSED_DATA_DIR,
    create_directories,
)
from src.data_loader import load_train_data


def add_rul_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Remaining Useful Life for every training observation.

    For each engine:

        RUL = maximum_cycle - current_cycle

    Therefore the final cycle before failure has RUL = 0.
    """

    df = df.copy()

    max_cycles = (
        df.groupby("unit")["cycle"]
        .max()
        .rename("max_cycle")
    )

    df = df.merge(
        max_cycles,
        on="unit",
        how="left",
    )

    df["RUL"] = (
        df["max_cycle"]
        - df["cycle"]
    )

    df.drop(
        columns=["max_cycle"],
        inplace=True,
    )

    return df


def cap_rul(
    df: pd.DataFrame,
    upper_limit: int = 125,
) -> pd.DataFrame:
    """
    Cap very large RUL values.

    Early-life observations often contain little
    degradation information. Capping prevents the
    model from spending disproportionate capacity
    fitting very large RUL values.
    """

    df = df.copy()

    df["RUL"] = np.minimum(
        df["RUL"],
        upper_limit,
    )

    return df


def preprocess():

    create_directories()

    print("=" * 60)
    print("RUL PREPROCESSING")
    print("=" * 60)

    train = load_train_data()

    print(
        f"Original rows: {len(train):,}"
    )

    # Step 1: calculate RUL
    train = add_rul_target(train)

    print(
        "\nRUL calculated successfully."
    )

    print(
        f"Minimum RUL: "
        f"{train['RUL'].min()}"
    )

    print(
        f"Maximum RUL: "
        f"{train['RUL'].max()}"
    )

    # Step 2: cap RUL
    train = cap_rul(
        train,
        upper_limit=125,
    )

    print(
        "\nRUL capped at 125 cycles."
    )

    print(
        f"New maximum RUL: "
        f"{train['RUL'].max()}"
    )

    # Save
    output_path = (
        PROCESSED_DATA_DIR
        / "train_rul.csv"
    )

    train.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved to:\n{output_path}"
    )

    return train


if __name__ == "__main__":

    preprocess()