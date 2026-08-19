import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.config import (
    RAW_DATA_DIR,
    MODEL_DIR,
    METRIC_DIR,
    create_directories,
)

from src.features import (
    add_temporal_features,
    add_cycle_features,
    add_degradation_features,
)


# ============================================================
# FILE PATHS
# ============================================================

TEST_FILE = (
    RAW_DATA_DIR / "test_FD001.txt"
)

RUL_FILE = (
    RAW_DATA_DIR / "RUL_FD001.txt"
)

MODEL_FILE = (
    MODEL_DIR / "random_forest_rul.joblib"
)

OUTPUT_FILE = (
    METRIC_DIR / "official_test_metrics.json"
)

PREDICTIONS_FILE = (
    METRIC_DIR / "official_test_predictions.csv"
)


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
# NASA C-MAPSS SCORING FUNCTION
# ============================================================

def nasa_score(
    y_true,
    y_pred,
):
    """
    Calculate the asymmetric NASA C-MAPSS scoring function.

    Under-prediction and over-prediction have different
    penalties.
    """

    errors = (
        y_pred - y_true
    )

    score = 0.0

    for error in errors:

        if error < 0:

            # Under-prediction
            score += (
                np.exp(
                    -error / 13
                ) - 1
            )

        else:

            # Over-prediction
            score += (
                np.exp(
                    error / 10
                ) - 1
            )

    return float(score)


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():

    test_df = pd.read_csv(
        TEST_FILE,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
    )

    rul_df = pd.read_csv(
        RUL_FILE,
        header=None,
        names=["RUL"],
    )

    return test_df, rul_df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df):
    """
    Apply EXACTLY the same feature-engineering pipeline
    used during training.

    Training pipeline:

        add_temporal_features()
                ↓
        add_cycle_features()
                ↓
        add_degradation_features()
                ↓
        replace infinities
                ↓
        forward fill
                ↓
        backward fill
    """

    print(
        "\nApplying training feature pipeline..."
    )

    # --------------------------------------------------------
    # 1. Temporal features
    # --------------------------------------------------------

    df = add_temporal_features(
        df
    )

    # --------------------------------------------------------
    # 2. Cycle features
    # --------------------------------------------------------

    df = add_cycle_features(
        df
    )

    # --------------------------------------------------------
    # 3. Cross-sensor degradation features
    # --------------------------------------------------------

    df = add_degradation_features(
        df
    )

    # --------------------------------------------------------
    # 4. Match training preprocessing exactly
    # --------------------------------------------------------

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    df.ffill(
        inplace=True
    )

    df.bfill(
        inplace=True
    )

    return df


# ============================================================
# GET FINAL OBSERVATION OF EACH ENGINE
# ============================================================

def get_final_engine_rows(
    df,
):
    """
    C-MAPSS FD001 test evaluation uses the final
    observed cycle of each engine.
    """

    final_rows = (
        df
        .sort_values(
            ["unit", "cycle"]
        )
        .groupby(
            "unit"
        )
        .tail(1)
        .copy()
    )

    return final_rows


# ============================================================
# MAIN
# ============================================================

def main():

    create_directories()

    print("=" * 70)
    print(
        "OFFICIAL NASA C-MAPSS FD001 TEST EVALUATION"
    )
    print("=" * 70)

    # ========================================================
    # LOAD DATA
    # ========================================================

    test_df, rul_df = (
        load_test_data()
    )

    print(
        f"\nTest rows: "
        f"{len(test_df):,}"
    )

    print(
        f"Test engines: "
        f"{test_df['unit'].nunique()}"
    )

    print(
        f"RUL labels: "
        f"{len(rul_df)}"
    )

    # ========================================================
    # VALIDATE DATA
    # ========================================================

    test_engines = (
        test_df["unit"]
        .nunique()
    )

    rul_labels = (
        len(rul_df)
    )

    if test_engines != rul_labels:

        raise ValueError(
            "Number of test engines does not "
            "match number of RUL labels."
        )

    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    print(
        "\nGenerating test features..."
    )

    featured_test = (
        create_features(
            test_df
        )
    )

    print(
        f"Generated test features: "
        f"{len(featured_test.columns)} columns"
    )

    # ========================================================
    # FINAL OBSERVATION PER ENGINE
    # ========================================================

    final_featured = (
        get_final_engine_rows(
            featured_test
        )
    )

    print(
        "\nFinal engine observations:"
    )

    print(
        len(final_featured)
    )

    # ========================================================
    # LOAD OFFICIAL RUL LABELS
    # ========================================================

    # RUL_FD001.txt contains one RUL value
    # for each test engine.

    y_test = (
        rul_df["RUL"]
        .to_numpy(
            dtype=float
        )
    )

    # ========================================================
    # LOAD TRAINED MODEL
    # ========================================================

    print(
        "\nLoading Random Forest model..."
    )

    checkpoint = joblib.load(
        MODEL_FILE
    )

    # The training script saved:
    #
    # {
    #     "model": pipeline,
    #     "features": feature_columns
    # }

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise ValueError(
            "Unexpected model checkpoint format."
        )

    if "model" not in checkpoint:

        raise ValueError(
            "Model checkpoint does not contain "
            "'model'."
        )

    if "features" not in checkpoint:

        raise ValueError(
            "Model checkpoint does not contain "
            "'features'."
        )

    pipeline = checkpoint[
        "model"
    ]

    feature_columns = checkpoint[
        "features"
    ]

    print(
        f"Model features: "
        f"{len(feature_columns)}"
    )

    # ========================================================
    # VERIFY FEATURE CONSISTENCY
    # ========================================================

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in final_featured.columns
    ]

    if missing_features:

        print(
            "\nMissing features:"
        )

        for feature in missing_features:

            print(
                f"  - {feature}"
            )

        raise ValueError(
            "Test feature set does not match "
            "training feature set."
        )

    # Check for unexpected NaNs.

    X_test = final_featured[
        feature_columns
    ].copy()

    nan_columns = (
        X_test.columns[
            X_test.isna()
            .any()
        ]
        .tolist()
    )

    if nan_columns:

        print(
            "\nColumns containing NaN values:"
        )

        for column in nan_columns:

            print(
                f"  - {column}"
            )

        raise ValueError(
            "NaN values found in test features."
        )

    # ========================================================
    # FINAL SHAPE VALIDATION
    # ========================================================

    if len(X_test) != len(y_test):

        raise ValueError(
            f"Feature rows ({len(X_test)}) "
            f"do not match RUL labels ({len(y_test)})."
        )

    print(
        f"\nEvaluation matrix:"
    )

    print(
        f"  Samples: {len(X_test)}"
    )

    print(
        f"  Features: {X_test.shape[1]}"
    )

    # ========================================================
    # PREDICTION
    # ========================================================

    print(
        "\nRunning predictions..."
    )

    predictions = (
        pipeline.predict(
            X_test
        )
    )

    # ========================================================
    # METRICS
    # ========================================================

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    nasa = nasa_score(
        y_test,
        predictions,
    )

    # ========================================================
    # PREDICTION DATAFRAME
    # ========================================================

    prediction_df = pd.DataFrame(
        {
            "unit":
                final_featured[
                    "unit"
                ].to_numpy(),

            "actual_rul":
                y_test,

            "predicted_rul":
                predictions,

            "error":
                predictions - y_test,

            "absolute_error":
                np.abs(
                    predictions - y_test
                ),
        }
    )

    # Sort by engine ID.

    prediction_df.sort_values(
        "unit",
        inplace=True,
    )

    prediction_df.reset_index(
        drop=True,
        inplace=True,
    )

    prediction_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    # ========================================================
    # METRICS JSON
    # ========================================================

    results = {
        "dataset":
            "NASA C-MAPSS FD001",

        "evaluation":
            "Official test set",

        "engines":
            int(len(y_test)),

        "features":
            int(X_test.shape[1]),

        "MAE":
            float(mae),

        "RMSE":
            float(rmse),

        "R2":
            float(r2),

        "NASA_score":
            float(nasa),
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
        )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n")

    print("=" * 70)
    print(
        "OFFICIAL TEST RESULT"
    )
    print("=" * 70)

    print(
        f"MAE: "
        f"{mae:.4f}"
    )

    print(
        f"RMSE: "
        f"{rmse:.4f}"
    )

    print(
        f"R²: "
        f"{r2:.4f}"
    )

    print(
        f"NASA Score: "
        f"{nasa:.2f}"
    )

    print(
        "\nPredictions saved to:"
    )

    print(
        PREDICTIONS_FILE
    )

    print(
        "\nMetrics saved to:"
    )

    print(
        OUTPUT_FILE
    )

    # ========================================================
    # SAMPLE PREDICTIONS
    # ========================================================

    print(
        "\nSample predictions:"
    )

    print(
        prediction_df.head(
            10
        ).to_string(
            index=False
        )
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()