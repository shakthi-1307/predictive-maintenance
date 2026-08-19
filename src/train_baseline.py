import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

from src.config import (
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    METRIC_DIR,
    RANDOM_STATE,
    create_directories,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

FEATURE_FILE = (
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


TARGET = "RUL"

# Columns that must never be used as model features.
EXCLUDED_COLUMNS = {
    "unit",
    "cycle",
    "RUL",
}


# ---------------------------------------------------------
# NASA C-MAPSS scoring function
# ---------------------------------------------------------

def nasa_score(
    y_true,
    y_pred,
):
    """
    NASA C-MAPSS asymmetric scoring function.

    Early predictions are penalized differently
    from late predictions.

    d = prediction - actual

    d < 0:
        model predicted too little RUL

    d > 0:
        model predicted too much RUL
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


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def calculate_metrics(
    y_true,
    y_pred,
):

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    score = nasa_score(
        y_true,
        y_pred,
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "NASA_score": float(score),
    }


# ---------------------------------------------------------
# Engine-level split
# ---------------------------------------------------------

def split_by_engine(
    df,
    validation_fraction=0.2,
):

    engines = (
        df["unit"]
        .unique()
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    rng.shuffle(engines)

    split_index = int(
        len(engines)
        * (1 - validation_fraction)
    )

    train_engines = engines[
        :split_index
    ]

    validation_engines = engines[
        split_index:
    ]

    train_df = df[
        df["unit"].isin(
            train_engines
        )
    ].copy()

    validation_df = df[
        df["unit"].isin(
            validation_engines
        )
    ].copy()

    return (
        train_df,
        validation_df,
        train_engines,
        validation_engines,
    )


# ---------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------

def prepare_features(df):

    feature_columns = [
        column
        for column in df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    X = df[
        feature_columns
    ].copy()

    y = df[
        TARGET
    ].copy()

    return (
        X,
        y,
        feature_columns,
    )


# ---------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------

def tune_random_forest(
    X_train,
    y_train,
    X_validation,
    y_validation,
):

    parameter_grid = {

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

    combinations = list(
        ParameterGrid(
            parameter_grid
        )
    )

    print(
        f"\nTesting "
        f"{len(combinations)} "
        f"hyperparameter configurations..."
    )

    best_pipeline = None
    best_metrics = None
    best_params = None

    best_mae = float(
        "inf"
    )

    for index, params in enumerate(
        combinations,
        start=1,
    ):

        print(
            f"\n[{index}/{len(combinations)}]"
        )

        print(
            f"Parameters: {params}"
        )

        pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),

                (
                    "model",
                    RandomForestRegressor(
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        **{
                            key.replace(
                                "model__",
                                ""
                            ): value
                            for key, value
                            in params.items()
                        },
                    ),
                ),
            ]
        )

        start = time.perf_counter()

        pipeline.fit(
            X_train,
            y_train,
        )

        training_time = (
            time.perf_counter()
            - start
        )

        start = time.perf_counter()

        predictions = pipeline.predict(
            X_validation
        )

        inference_time = (
            time.perf_counter()
            - start
        )

        current_metrics = calculate_metrics(
            y_validation,
            predictions,
        )

        current_metrics[
            "training_time_seconds"
        ] = float(
            training_time
        )

        current_metrics[
            "validation_inference_seconds"
        ] = float(
            inference_time
        )

        print(
            f"MAE: "
            f"{current_metrics['MAE']:.4f}"
        )

        print(
            f"RMSE: "
            f"{current_metrics['RMSE']:.4f}"
        )

        print(
            f"R²: "
            f"{current_metrics['R2']:.4f}"
        )

        print(
            f"NASA Score: "
            f"{current_metrics['NASA_score']:.2f}"
        )

        # Lower MAE is better.
        if (
            current_metrics["MAE"]
            < best_mae
        ):

            best_mae = (
                current_metrics["MAE"]
            )

            best_pipeline = pipeline

            best_metrics = (
                current_metrics
            )

            best_params = params

    return (
        best_pipeline,
        best_metrics,
        best_params,
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    create_directories()

    print("=" * 70)
    print("RANDOM FOREST RUL BASELINE")
    print("=" * 70)

    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    df = pd.read_csv(
        FEATURE_FILE
    )

    print(
        f"\nDataset rows: "
        f"{len(df):,}"
    )

    print(
        f"Total engines: "
        f"{df['unit'].nunique()}"
    )

    # -----------------------------------------------------
    # Split by engine
    # -----------------------------------------------------

    (
        train_df,
        validation_df,
        train_engines,
        validation_engines,
    ) = split_by_engine(
        df
    )

    print(
        f"\nTraining engines: "
        f"{len(train_engines)}"
    )

    print(
        f"Validation engines: "
        f"{len(validation_engines)}"
    )

    print(
        f"Training rows: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_df):,}"
    )

    # -----------------------------------------------------
    # Features
    # -----------------------------------------------------

    (
        X_train,
        y_train,
        feature_columns,
    ) = prepare_features(
        train_df
    )

    (
        X_validation,
        y_validation,
        _,
    ) = prepare_features(
        validation_df
    )

    print(
        f"\nNumber of features: "
        f"{len(feature_columns)}"
    )

    # -----------------------------------------------------
    # Train / tune
    # -----------------------------------------------------

    (
        best_model,
        best_metrics,
        best_params,
    ) = tune_random_forest(
        X_train,
        y_train,
        X_validation,
        y_validation,
    )

    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------

    joblib.dump(
        {
            "model": best_model,
            "features": feature_columns,
        },
        MODEL_FILE,
    )

    # -----------------------------------------------------
    # Save metrics
    # -----------------------------------------------------

    results = {

        "model": "RandomForestRegressor",

        "dataset": "NASA C-MAPSS FD001",

        "training_engines": int(
            len(train_engines)
        ),

        "validation_engines": int(
            len(validation_engines)
        ),

        "training_rows": int(
            len(train_df)
        ),

        "validation_rows": int(
            len(validation_df)
        ),

        "feature_count": int(
            len(feature_columns)
        ),

        "best_parameters": best_params,

        "metrics": best_metrics,
    }

    with open(
        METRICS_FILE,
        "w",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
        )

    # -----------------------------------------------------
    # Final output
    # -----------------------------------------------------

    print("\n")
    print("=" * 70)
    print("BEST RANDOM FOREST RESULT")
    print("=" * 70)

    print(
        f"MAE: "
        f"{best_metrics['MAE']:.4f}"
    )

    print(
        f"RMSE: "
        f"{best_metrics['RMSE']:.4f}"
    )

    print(
        f"R²: "
        f"{best_metrics['R2']:.4f}"
    )

    print(
        f"NASA Score: "
        f"{best_metrics['NASA_score']:.2f}"
    )

    print(
        f"\nBest parameters:"
    )

    for key, value in (
        best_params.items()
    ):

        print(
            f"  {key}: {value}"
        )

    print(
        f"\nModel saved to:"
        f"\n{MODEL_FILE}"
    )

    print(
        f"\nMetrics saved to:"
        f"\n{METRICS_FILE}"
    )


if __name__ == "__main__":

    main()