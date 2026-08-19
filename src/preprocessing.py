import numpy as np
import pandas as pd

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    create_directories,
)


# ============================================================
# CONFIGURATION
# ============================================================

RUL_CAP = 125

# Create multiple truncated observations from each engine.
# These represent different points at which the engine could
# have been observed before failure.
TRUNCATION_POINTS = [
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
]


# ============================================================
# DATASET COLUMNS
# ============================================================

COLUMNS = [
    "unit",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    "s1",
    "s2",
    "s3",
    "s4",
    "s5",
    "s6",
    "s7",
    "s8",
    "s9",
    "s10",
    "s11",
    "s12",
    "s13",
    "s14",
    "s15",
    "s16",
    "s17",
    "s18",
    "s19",
    "s20",
    "s21",
]


# ============================================================
# LOAD TRAINING DATA
# ============================================================

def load_training_data():

    train_file = (
        RAW_DATA_DIR
        / "train_FD001.txt"
    )

    df = pd.read_csv(
        train_file,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
    )

    return df


# ============================================================
# CALCULATE TRUE RUL
# ============================================================

def calculate_rul(df):

    df = df.copy()

    max_cycles = (
        df.groupby("unit")["cycle"]
        .transform("max")
    )

    df["RUL"] = (
        max_cycles
        - df["cycle"]
    )

    return df


# ============================================================
# CAP RUL
# ============================================================

def cap_rul(
    df,
    cap=RUL_CAP,
):

    df = df.copy()

    df["RUL"] = np.minimum(
        df["RUL"],
        cap,
    )

    return df


# ============================================================
# CREATE TRUNCATED TRAINING EXAMPLES
# ============================================================

def create_truncated_examples(
    df,
):
    """
    Create training examples that mimic the official
    C-MAPSS test setup.

    For every engine:

        Full trajectory
              ↓
        choose observation cutoff
              ↓
        retain observations up to cutoff
              ↓
        calculate RUL relative to the TRUE failure cycle

    This prevents the model from learning that the last
    observed cycle necessarily means imminent failure.
    """

    all_examples = []

    engine_ids = (
        df["unit"]
        .unique()
    )

    print(
        "\nCreating truncated trajectories..."
    )

    print(
        f"Engines: {len(engine_ids)}"
    )

    print(
        f"Truncation points: "
        f"{TRUNCATION_POINTS}"
    )

    for unit in engine_ids:

        engine = (
            df[
                df["unit"] == unit
            ]
            .sort_values("cycle")
            .copy()
        )

        total_cycles = (
            engine["cycle"].max()
        )

        # ----------------------------------------------------
        # Generate multiple observation windows
        # ----------------------------------------------------

        for fraction in TRUNCATION_POINTS:

            cutoff_cycle = int(
                np.floor(
                    total_cycles
                    * fraction
                )
            )

            cutoff_cycle = max(
                cutoff_cycle,
                1,
            )

            truncated = (
                engine[
                    engine["cycle"]
                    <= cutoff_cycle
                ]
                .copy()
            )

            if len(truncated) == 0:
                continue

            # ------------------------------------------------
            # IMPORTANT:
            #
            # RUL is calculated relative to the ORIGINAL
            # engine failure cycle, NOT the truncation point.
            # ------------------------------------------------

            truncated["RUL"] = (
                total_cycles
                - truncated["cycle"]
            )

            all_examples.append(
                truncated
            )

    result = pd.concat(
        all_examples,
        ignore_index=True,
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    create_directories()

    print("=" * 60)
    print("C-MAPSS RUL PREPROCESSING")
    print("=" * 60)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_training_data()

    print(
        f"\nOriginal rows: "
        f"{len(df):,}"
    )

    print(
        f"Original engines: "
        f"{df['unit'].nunique()}"
    )

    # --------------------------------------------------------
    # Calculate RUL
    # --------------------------------------------------------

    df = calculate_rul(
        df
    )

    print(
        "\nRUL calculated successfully."
    )

    print(
        f"Minimum RUL: "
        f"{df['RUL'].min()}"
    )

    print(
        f"Maximum RUL: "
        f"{df['RUL'].max()}"
    )

    # --------------------------------------------------------
    # Cap RUL
    # --------------------------------------------------------

    df = cap_rul(
        df
    )

    print(
        f"\nRUL capped at "
        f"{RUL_CAP} cycles."
    )

    print(
        f"New maximum RUL: "
        f"{df['RUL'].max()}"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Build truncated trajectories from the ORIGINAL
    # uncapped RUL information.
    #
    # We calculate RUL again inside this function from
    # each engine's total life.
    # --------------------------------------------------------

    print(
        "\nBuilding test-compatible "
        "training trajectories..."
    )

    original_with_rul = (
        calculate_rul(
            load_training_data()
        )
    )

    truncated = (
        create_truncated_examples(
            original_with_rul
        )
    )

    # --------------------------------------------------------
    # Cap RUL after truncation
    # --------------------------------------------------------

    truncated = cap_rul(
        truncated
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    truncated.sort_values(
        ["unit", "cycle"],
        inplace=True,
    )

    truncated.reset_index(
        drop=True,
        inplace=True,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        PROCESSED_DATA_DIR
        / "train_rul.csv"
    )

    truncated.to_csv(
        output_path,
        index=False,
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "TRUNCATED RUL DATASET"
    )

    print(
        "=" * 60
    )

    print(
        f"Rows: "
        f"{len(truncated):,}"
    )

    print(
        f"Engines: "
        f"{truncated['unit'].nunique()}"
    )

    print(
        f"Maximum RUL: "
        f"{truncated['RUL'].max()}"
    )

    print(
        f"Minimum RUL: "
        f"{truncated['RUL'].min()}"
    )

    print(
        f"\nSaved to:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()