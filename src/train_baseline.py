import json
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    ParameterGrid,
)
from sklearn.pipeline import Pipeline

from src.config import (
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    METRIC_DIR,
    NON_FEATURE_COLUMNS,
    create_directories,
)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    PROCESSED_DATA_DIR
    / "train_features.csv"
)

MODEL_FILE = (
    MODEL_DIR
    / "random_forest_rul.joblib"
)

METRICS_FILE = (
    METRIC_DIR
    / "random_forest_metrics.json"
)

RANDOM_STATE = 42

VALIDATION_SIZE = 0.20


# ============================================================
# NASA C-MAPSS SCORE
# ============================================================

def nasa_score(
    y_true,
    y_pred,
):
    """
    NASA C-MAPSS asymmetric scoring function.

    Under-prediction and over-prediction receive
    different penalties.
    """

    errors = (
        y_pred - y_true
    )

    score = 0.0

    for error in errors:

        if error < 0:

            score += (
                np.exp(
                    -error / 13
                ) - 1
            )

        else:

            score += (
                np.exp(
                    error / 10
                ) - 1
            )

    return float(score)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    df = pd.read_csv(
        INPUT_FILE
    )

    return df


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

def split_by_engine(
    df,
):
    """
    Split by physical engine rather than by individual rows.

    This is critical because each engine contributes many
    time-series observations, and several truncated
    trajectories that overlap heavily with one another.
    Grouping on `source_unit` keeps every trajectory cut
    from the same engine on the same side of the split.
    """

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
    )

    groups = df["source_unit"]

    train_idx, val_idx = next(
        splitter.split(
            df,
            groups=groups,
        )
    )

    train_df = (
        df.iloc[train_idx]
        .copy()
    )

    val_df = (
        df.iloc[val_idx]
        .copy()
    )

    return (
        train_df,
        val_df,
    )


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_data(
    train_df,
    val_df,
):
    """
    Remove identifiers and target from the feature matrix.
    """

    feature_columns = [
        column
        for column in train_df.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    X_train = (
        train_df[
            feature_columns
        ]
        .copy()
    )

    y_train = (
        train_df["RUL"]
        .to_numpy()
    )

    X_val = (
        val_df[
            feature_columns
        ]
        .copy()
    )

    y_val = (
        val_df["RUL"]
        .to_numpy()
    )

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        feature_columns,
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_val,
    y_val,
):
    """
    Evaluate a trained model using MAE, RMSE,
    R² and NASA score.
    """

    predictions = (
        model.predict(
            X_val
        )
    )

    mae = mean_absolute_error(
        y_val,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_val,
            predictions,
        )
    )

    r2 = r2_score(
        y_val,
        predictions,
    )

    nasa = nasa_score(
        y_val,
        predictions,
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "nasa_score": float(nasa),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    create_directories()

    print("=" * 70)
    print(
        "RANDOM FOREST RUL BASELINE"
    )
    print("=" * 70)

    # ========================================================
    # LOAD
    # ========================================================

    df = load_data()

    print(
        f"\nDataset rows: "
        f"{len(df):,}"
    )

    # ========================================================
    # VERIFY TRUNCATED DATASET
    # ========================================================

    if "source_unit" not in df.columns:

        raise ValueError(
            "The dataset appears to be the old "
            "non-truncated dataset. Run "
            "`python -m src.preprocessing` and "
            "`python -m src.features` first."
        )

    print(
        f"Trajectories: "
        f"{df['unit'].nunique()}"
    )

    print(
        f"Source engines: "
        f"{df['source_unit'].nunique()}"
    )

    print(
        "\nTraining on truncated trajectories."
    )

    # ========================================================
    # ENGINE-LEVEL SPLIT
    # ========================================================

    (
        train_df,
        val_df,
    ) = split_by_engine(
        df
    )

    print(
        f"\nTraining engines: "
        f"{train_df['source_unit'].nunique()}"
    )

    print(
        f"Validation engines: "
        f"{val_df['source_unit'].nunique()}"
    )

    print(
        f"Training rows: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation rows: "
        f"{len(val_df):,}"
    )

    # ========================================================
    # PREPARE DATA
    # ========================================================

    (
        X_train,
        y_train,
        X_val,
        y_val,
        feature_columns,
    ) = prepare_data(
        train_df,
        val_df,
    )

    print(
        f"\nNumber of features: "
        f"{len(feature_columns)}"
    )

    # ========================================================
    # HYPERPARAMETER GRID
    # ========================================================

    param_grid = {

        "model__n_estimators": [
            100,
            200,
        ],

        "model__max_depth": [
            10,
            20,
            None,
        ],

        "model__min_samples_leaf": [
            1,
            2,
            4,
        ],

        "model__max_features": [
            "sqrt",
            0.7,
        ],
    }

    configurations = list(
        ParameterGrid(
            param_grid
        )
    )

    print(
        f"\nTesting "
        f"{len(configurations)} "
        f"hyperparameter configurations..."
    )

    # ========================================================
    # SEARCH
    # ========================================================

    best_score = float(
        "inf"
    )

    best_result = None
    best_pipeline = None

    for index, params in enumerate(
        configurations,
        start=1,
    ):

        print(
            f"\n[{index}/{len(configurations)}]"
        )

        print(
            f"Parameters: {params}"
        )

        # ----------------------------------------------------
        # Pipeline
        # ----------------------------------------------------

        pipeline = Pipeline(
            steps=[
                (
                    "model",
                    RandomForestRegressor(
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                )
            ]
        )

        pipeline.set_params(
            **params
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        start_time = time.time()

        pipeline.fit(
            X_train,
            y_train,
        )

        training_time = (
            time.time()
            - start_time
        )

        # ----------------------------------------------------
        # Evaluate
        # ----------------------------------------------------

        metrics = evaluate_model(
            pipeline,
            X_val,
            y_val,
        )

        print(
            f"MAE: "
            f"{metrics['mae']:.4f}"
        )

        print(
            f"RMSE: "
            f"{metrics['rmse']:.4f}"
        )

        print(
            f"R²: "
            f"{metrics['r2']:.4f}"
        )

        print(
            f"NASA Score: "
            f"{metrics['nasa_score']:.2f}"
        )

        print(
            f"Training time: "
            f"{training_time:.2f}s"
        )

        # ----------------------------------------------------
        # Select best model
        #
        # NASA score is the primary metric because it is
        # specifically designed for C-MAPSS RUL prediction.
        # ----------------------------------------------------

        if (
            metrics["nasa_score"]
            < best_score
        ):

            best_score = (
                metrics["nasa_score"]
            )

            best_result = {
                "parameters": params,
                "metrics": metrics,
                "training_time": (
                    training_time
                ),
            }

            best_pipeline = pipeline

            print(
                "✓ New best model"
            )

    # ========================================================
    # BEST RESULT
    # ========================================================

    print("\n")

    print("=" * 70)
    print(
        "BEST RANDOM FOREST RESULT"
    )
    print("=" * 70)

    print(
        f"MAE: "
        f"{best_result['metrics']['mae']:.4f}"
    )

    print(
        f"RMSE: "
        f"{best_result['metrics']['rmse']:.4f}"
    )

    print(
        f"R²: "
        f"{best_result['metrics']['r2']:.4f}"
    )

    print(
        f"NASA Score: "
        f"{best_result['metrics']['nasa_score']:.2f}"
    )

    print(
        "\nBest parameters:"
    )

    for key, value in (
        best_result[
            "parameters"
        ].items()
    ):

        print(
            f"  {key}: {value}"
        )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    # Per-feature training medians. The API is handed a single
    # sensor reading and cannot compute every column the model was
    # fitted on (the near-constant settings and dead sensors); it
    # substitutes these instead of zeros, which for a column like
    # setting_3 (always 100.0) would be far outside the training
    # range.
    feature_defaults = {
        column: float(value)
        for column, value in (
            X_train
            .median()
            .items()
        )
    }

    checkpoint = {
        "model":
            best_pipeline,

        "features":
            feature_columns,

        "feature_defaults":
            feature_defaults,

        "training_type":
            "truncated_trajectories",

        "rul_cap":
            125,

        "random_state":
            RANDOM_STATE,
    }

    joblib.dump(
        checkpoint,
        MODEL_FILE,
    )

    print(
        "\nModel saved to:"
    )

    print(
        MODEL_FILE
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics_output = {

        "dataset":
            "NASA C-MAPSS FD001",

        "training_strategy":
            "truncated_trajectories",

        "training_engines":
            int(
                train_df["source_unit"]
                .nunique()
            ),

        "validation_engines":
            int(
                val_df["source_unit"]
                .nunique()
            ),

        "training_rows":
            int(
                len(train_df)
            ),

        "validation_rows":
            int(
                len(val_df)
            ),

        "features":
            int(
                len(feature_columns)
            ),

        "best_parameters":
            best_result[
                "parameters"
            ],

        "MAE":
            best_result[
                "metrics"
            ]["mae"],

        "RMSE":
            best_result[
                "metrics"
            ]["rmse"],

        "R2":
            best_result[
                "metrics"
            ]["r2"],

        "NASA_score":
            best_result[
                "metrics"
            ]["nasa_score"],

        "training_time":
            best_result[
                "training_time"
            ],
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics_output,
            f,
            indent=4,
        )

    print(
        "\nMetrics saved to:"
    )

    print(
        METRICS_FILE
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()