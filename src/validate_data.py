import pandas as pd

from src.config import RAW_DATA_PATH, SENSOR_COLUMNS


REQUIRED_COLUMNS = [
    "machine_id",
    "timestamp",
    "machine_age",
    *SENSOR_COLUMNS,
    "failure",
]


def validate_dataset(df: pd.DataFrame):

    errors = []

    # Schema validation
    missing_columns = [
        col
        for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        errors.append(
            f"Missing columns: {missing_columns}"
        )

    # Empty dataset
    if df.empty:
        errors.append("Dataset is empty.")

    # Duplicate detection
    duplicates = df.duplicated(
        subset=["machine_id", "timestamp"]
    ).sum()

    if duplicates > 0:
        errors.append(
            f"Found {duplicates} duplicate machine/timestamp rows."
        )

    # Null validation
    null_counts = df[REQUIRED_COLUMNS].isnull().sum()

    null_columns = {
        col: int(count)
        for col, count in null_counts.items()
        if count > 0
    }

    if null_columns:
        errors.append(
            f"Null values found: {null_columns}"
        )

    # Numeric range checks
    if (df["temperature"] < -50).any():
        errors.append("Invalid temperature values.")

    if (df["vibration"] < 0).any():
        errors.append("Negative vibration values.")

    if (df["rotational_speed"] < 0).any():
        errors.append("Negative rotational speed values.")

    if not set(df["failure"].unique()).issubset({0, 1}):
        errors.append(
            "Failure column must contain only 0 and 1."
        )

    if errors:
        raise ValueError(
            "\n".join(errors)
        )

    return True


def main():

    df = pd.read_csv(RAW_DATA_PATH)

    validate_dataset(df)

    print("=" * 60)
    print("DATA VALIDATION PASSED")
    print("=" * 60)
    print(f"Rows validated: {len(df):,}")
    print(f"Columns validated: {len(df.columns)}")
    print("No schema, null, duplicate, or range errors.")


if __name__ == "__main__":
    main()