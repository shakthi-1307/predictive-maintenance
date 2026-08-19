import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from xgboost import XGBRegressor

from src.config import (
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    METRIC_DIR,
    RANDOM_STATE,
    create_directories,
)


FEATURE_FILE = (
    PROCESSED_DATA_DIR
    / "train_features.csv"
)

MODEL_FILE = (
    MODEL_DIR
    / "xgboost_rul.json"
)

METRICS_FILE = (
    METRIC_DIR
    / "xgboost_metrics.json"
)


TARGET = "RUL"

EXCLUDED_COLUMNS = {
    "unit",
    "cycle",
    "RUL",
}


def nasa_score(y_true, y_pred):

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

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "NASA_score": nasa_score(
            y_true,
            y_pred,
        ),
    }


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


def main():

    create_directories()

    print("=" * 70)
    print("XGBOOST RUL MODEL")
    print("=" * 70)

    df = pd.read_csv(
        FEATURE_FILE
    )

    (
        train_df,
        validation_df,
        train_engines,
        validation_engines,
    ) = split_by_engine(
        df
    )

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
        f"\nTraining engines: "
        f"{len(train_engines)}"
    )

    print(
        f"Validation engines: "
        f"{len(validation_engines)}"
    )

    print(
        f"Training rows: "
        f"{len(X_train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(X_validation):,}"
    )

    print(
        f"Features: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = XGBRegressor(

        n_estimators=1000,

        learning_rate=0.03,

        max_depth=6,

        min_child_weight=3,

        subsample=0.8,

        colsample_bytree=0.8,

        reg_alpha=0.05,

        reg_lambda=1.0,

        objective="reg:squarederror",

        eval_metric="mae",

        random_state=RANDOM_STATE,

        n_jobs=-1,

        tree_method="hist",
    )

    print(
        "\nTraining XGBoost..."
    )

    start = time.perf_counter()

    model.fit(
        X_train,
        y_train,

        eval_set=[
            (
                X_validation,
                y_validation,
            )
        ],

        verbose=False,
    )

    training_time = (
        time.perf_counter()
        - start
    )

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    start = time.perf_counter()

    predictions = model.predict(
        X_validation
    )

    inference_time = (
        time.perf_counter()
        - start
    )

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    metrics = calculate_metrics(
        y_validation,
        predictions,
    )

    metrics[
        "training_time_seconds"
    ] = float(
        training_time
    )

    metrics[
        "validation_inference_seconds"
    ] = float(
        inference_time
    )

    metrics[
        "samples_per_second"
    ] = float(
        len(X_validation)
        / inference_time
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    model.save_model(
        MODEL_FILE
    )

    results = {

        "model": "XGBRegressor",

        "dataset":
            "NASA C-MAPSS FD001",

        "training_engines":
            int(len(train_engines)),

        "validation_engines":
            int(len(validation_engines)),

        "training_rows":
            int(len(X_train)),

        "validation_rows":
            int(len(X_validation)),

        "feature_count":
            int(len(feature_columns)),

        "parameters": {

            "n_estimators": 1000,

            "learning_rate": 0.03,

            "max_depth": 6,

            "min_child_weight": 3,

            "subsample": 0.8,

            "colsample_bytree": 0.8,

            "reg_alpha": 0.05,

            "reg_lambda": 1.0,
        },

        "metrics": metrics,
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

    # --------------------------------------------------
    # Output
    # --------------------------------------------------

    print("\n")
    print("=" * 70)
    print("XGBOOST RESULT")
    print("=" * 70)

    print(
        f"MAE: "
        f"{metrics['MAE']:.4f}"
    )

    print(
        f"RMSE: "
        f"{metrics['RMSE']:.4f}"
    )

    print(
        f"R²: "
        f"{metrics['R2']:.4f}"
    )

    print(
        f"NASA Score: "
        f"{metrics['NASA_score']:.2f}"
    )

    print(
        f"Training time: "
        f"{training_time:.2f}s"
    )

    print(
        f"Inference time: "
        f"{inference_time:.4f}s"
    )

    print(
        f"Throughput: "
        f"{metrics['samples_per_second']:.2f} samples/sec"
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