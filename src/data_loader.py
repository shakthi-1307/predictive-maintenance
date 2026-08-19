import pandas as pd

from src.config import (
    TRAIN_FILE,
    TEST_FILE,
    RUL_FILE,
    ALL_COLUMNS,
)


def load_train_data():

    df = pd.read_csv(
        TRAIN_FILE,
        sep=r"\s+",
        header=None,
        names=ALL_COLUMNS,
    )

    return df


def load_test_data():

    df = pd.read_csv(
        TEST_FILE,
        sep=r"\s+",
        header=None,
        names=ALL_COLUMNS,
    )

    return df


def load_test_rul():

    rul = pd.read_csv(
        RUL_FILE,
        header=None,
        names=["RUL"],
    )

    return rul["RUL"].values


def load_all():

    train = load_train_data()

    test = load_test_data()

    rul = load_test_rul()

    return train, test, rul


if __name__ == "__main__":

    train, test, rul = load_all()

    print("=" * 60)
    print("NASA C-MAPSS FD001")
    print("=" * 60)

    print(
        f"Train rows: {len(train):,}"
    )

    print(
        f"Test rows: {len(test):,}"
    )

    print(
        f"Training engines: "
        f"{train['unit'].nunique()}"
    )

    print(
        f"Test engines: "
        f"{test['unit'].nunique()}"
    )

    print(
        f"RUL values: {len(rul)}"
    )

    print("\nColumns:")

    for column in train.columns:

        print(f"  {column}")