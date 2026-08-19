from src.data_loader import load_all


def validate_dataset(
    train,
    test,
    rul,
):

    errors = []

    # -------------------------
    # Basic checks
    # -------------------------

    if train.empty:

        errors.append(
            "Training dataset is empty."
        )

    if test.empty:

        errors.append(
            "Test dataset is empty."
        )

    if len(rul) != test["unit"].nunique():

        errors.append(
            "RUL count does not match "
            "number of test engines."
        )

    # -------------------------
    # Schema
    # -------------------------

    if list(train.columns) != list(test.columns):

        errors.append(
            "Train/test schemas do not match."
        )

    # -------------------------
    # Missing values
    # -------------------------

    train_missing = train.isnull().sum().sum()

    test_missing = test.isnull().sum().sum()

    if train_missing > 0:

        errors.append(
            f"Training set contains "
            f"{train_missing} missing values."
        )

    if test_missing > 0:

        errors.append(
            f"Test set contains "
            f"{test_missing} missing values."
        )

    # -------------------------
    # Duplicate rows
    # -------------------------

    duplicate_train = train.duplicated().sum()

    duplicate_test = test.duplicated().sum()

    if duplicate_train > 0:

        errors.append(
            f"Training set contains "
            f"{duplicate_train} duplicates."
        )

    if duplicate_test > 0:

        errors.append(
            f"Test set contains "
            f"{duplicate_test} duplicates."
        )

    # -------------------------
    # Engine consistency
    # -------------------------

    train_engines = set(
        train["unit"].unique()
    )

    test_engines = set(
        test["unit"].unique()
    )

    if not train_engines:

        errors.append(
            "No training engines found."
        )

    if not test_engines:

        errors.append(
            "No test engines found."
        )

    # -------------------------
    # Cycle validation
    # -------------------------

    for df, name in [
        (train, "train"),
        (test, "test"),
    ]:

        invalid_cycles = (
            df.groupby("unit")["cycle"]
            .apply(
                lambda x:
                not x.is_monotonic_increasing
            )
        )

        if invalid_cycles.any():

            errors.append(
                f"{name} contains engines "
                "with non-monotonic cycles."
            )

    if errors:

        raise ValueError(
            "\n".join(errors)
        )

    return True


def main():

    train, test, rul = load_all()

    validate_dataset(
        train,
        test,
        rul,
    )

    print("=" * 60)
    print("DATA VALIDATION PASSED")
    print("=" * 60)

    print(
        f"Training rows: {len(train):,}"
    )

    print(
        f"Testing rows: {len(test):,}"
    )

    print(
        f"Training engines: "
        f"{train['unit'].nunique()}"
    )

    print(
        f"Testing engines: "
        f"{test['unit'].nunique()}"
    )


if __name__ == "__main__":

    main()