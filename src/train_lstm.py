import json
import time

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.config import (
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    METRIC_DIR,
    RANDOM_STATE,
    create_directories,
)

from src.datasets import RULSequenceDataset
from src.models import RULLSTM


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_FILE = (
    PROCESSED_DATA_DIR
    / "train_features.csv"
)

MODEL_FILE = (
    MODEL_DIR
    / "lstm_rul.pt"
)

SCALER_FILE = (
    MODEL_DIR
    / "lstm_scaler.joblib"
)

METRICS_FILE = (
    METRIC_DIR
    / "lstm_metrics.json"
)


SEQUENCE_LENGTH = 30

BATCH_SIZE = 128

EPOCHS = 30

LEARNING_RATE = 0.001

PATIENCE = 5

HIDDEN_SIZE = 64

NUM_LAYERS = 2

DROPOUT = 0.2


# These are the sensors identified during EDA
# as useful degradation indicators.

SENSOR_COLUMNS = [
    "s2",
    "s3",
    "s4",
    "s7",
    "s8",
    "s11",
    "s12",
    "s13",
    "s14",
    "s15",
    "s17",
    "s20",
    "s21",
]


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed=RANDOM_STATE):

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# NASA SCORE
# ============================================================

def nasa_score(
    y_true,
    y_pred,
):

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
# METRICS
# ============================================================

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


# ============================================================
# ENGINE-LEVEL SPLIT
# ============================================================

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


# ============================================================
# SCALE FEATURES
# ============================================================

def scale_features(
    train_df,
    validation_df,
    feature_columns,
):

    scaler = StandardScaler()

    train_df = train_df.copy()

    validation_df = (
        validation_df.copy()
    )

    scaler.fit(
        train_df[
            feature_columns
        ]
    )

    train_df[
        feature_columns
    ] = scaler.transform(
        train_df[
            feature_columns
        ]
    )

    validation_df[
        feature_columns
    ] = scaler.transform(
        validation_df[
            feature_columns
        ]
    )

    return (
        train_df,
        validation_df,
        scaler,
    )


# ============================================================
# CREATE DATALOADERS
# ============================================================

def create_dataloaders(
    train_df,
    validation_df,
    feature_columns,
):

    train_dataset = (
        RULSequenceDataset(
            train_df,
            feature_columns,
            sequence_length=SEQUENCE_LENGTH,
        )
    )

    validation_dataset = (
        RULSequenceDataset(
            validation_df,
            feature_columns,
            sequence_length=SEQUENCE_LENGTH,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return (
        train_loader,
        validation_loader,
        train_dataset,
        validation_dataset,
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):

    model.train()

    total_loss = 0.0

    total_samples = 0

    for sequences, targets in loader:

        sequences = (
            sequences.to(device)
        )

        targets = (
            targets.to(device)
        )

        optimizer.zero_grad()

        predictions = model(
            sequences
        )

        loss = criterion(
            predictions,
            targets,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        batch_size = (
            sequences.size(0)
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += (
            batch_size
        )

    return (
        total_loss
        / total_samples
    )


# ============================================================
# VALIDATION
# ============================================================

def evaluate(
    model,
    loader,
    device,
):

    model.eval()

    predictions = []

    actuals = []

    with torch.no_grad():

        for sequences, targets in loader:

            sequences = (
                sequences.to(device)
            )

            outputs = model(
                sequences
            )

            predictions.extend(
                outputs.cpu()
                .numpy()
            )

            actuals.extend(
                targets.numpy()
            )

    predictions = np.array(
        predictions
    )

    actuals = np.array(
        actuals
    )

    metrics = calculate_metrics(
        actuals,
        predictions,
    )

    return (
        metrics,
        actuals,
        predictions,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    create_directories()

    set_seed()

    print("=" * 70)
    print("LSTM RUL MODEL")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = pd.read_csv(
        FEATURE_FILE
    )

    print(
        f"\nDataset rows: "
        f"{len(df):,}"
    )

    print(
        f"Engines: "
        f"{df['unit'].nunique()}"
    )

    # --------------------------------------------------------
    # Engine split
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Feature selection
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in SENSOR_COLUMNS
        if column in df.columns
    ]

    print(
        f"\nSequence features: "
        f"{len(feature_columns)}"
    )

    print(
        feature_columns
    )

    # --------------------------------------------------------
    # Scale
    # --------------------------------------------------------

    print(
        "\nFitting scaler on training data..."
    )

    (
        train_df,
        validation_df,
        scaler,
    ) = scale_features(
        train_df,
        validation_df,
        feature_columns,
    )

    # --------------------------------------------------------
    # Save scaler
    # --------------------------------------------------------

    joblib.dump(
        scaler,
        SCALER_FILE,
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    print(
        "\nCreating sequences..."
    )

    (
        train_loader,
        validation_loader,
        train_dataset,
        validation_dataset,
    ) = create_dataloaders(
        train_df,
        validation_df,
        feature_columns,
    )

    print(
        f"Training sequences: "
        f"{len(train_dataset):,}"
    )

    print(
        f"Validation sequences: "
        f"{len(validation_dataset):,}"
    )

    print(
        f"Sequence length: "
        f"{SEQUENCE_LENGTH}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = RULLSTM(
        input_size=len(
            feature_columns
        ),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    print(
        "\nModel:"
    )

    print(model)

    # --------------------------------------------------------
    # Loss / optimizer
    # --------------------------------------------------------

    criterion = torch.nn.HuberLoss(
        delta=1.0
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
        )
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_mae = float("inf")

    best_state = None

    epochs_without_improvement = 0

    history = []

    training_start = (
        time.perf_counter()
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        epoch_start = (
            time.perf_counter()
        )

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        (
            metrics,
            _,
            _,
        ) = evaluate(
            model,
            validation_loader,
            device,
        )

        scheduler.step(
            metrics["MAE"]
        )

        epoch_time = (
            time.perf_counter()
            - epoch_start
        )

        current_lr = (
            optimizer.param_groups[
                0
            ]["lr"]
        )

        print(
            f"\nEpoch "
            f"{epoch:02d}/{EPOCHS}"
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Val MAE: "
            f"{metrics['MAE']:.4f}"
        )

        print(
            f"Val RMSE: "
            f"{metrics['RMSE']:.4f}"
        )

        print(
            f"Val R²: "
            f"{metrics['R2']:.4f}"
        )

        print(
            f"Val NASA Score: "
            f"{metrics['NASA_score']:.2f}"
        )

        print(
            f"Learning rate: "
            f"{current_lr:.6f}"
        )

        print(
            f"Time: "
            f"{epoch_time:.2f}s"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(
                    train_loss
                ),
                **metrics,
                "learning_rate":
                    float(
                        current_lr
                    ),
            }
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if (
            metrics["MAE"]
            < best_mae
        ):

            best_mae = (
                metrics["MAE"]
            )

            best_state = {
                key: value.cpu().clone()
                for key, value
                in model.state_dict()
                .items()
            }

            epochs_without_improvement = 0

            print(
                "✓ New best model"
            )

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                "\nEarly stopping triggered."
            )

            break

    total_training_time = (
        time.perf_counter()
        - training_start
    )

    # --------------------------------------------------------
    # Restore best model
    # --------------------------------------------------------

    model.load_state_dict(
        best_state
    )

    model.to(device)

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    (
        final_metrics,
        actuals,
        predictions,
    ) = evaluate(
        model,
        validation_loader,
        device,
    )

    # --------------------------------------------------------
    # Inference benchmark
    # --------------------------------------------------------

    inference_start = (
        time.perf_counter()
    )

    _ = evaluate(
        model,
        validation_loader,
        device,
    )

    inference_time = (
        time.perf_counter()
        - inference_start
    )

    # --------------------------------------------------------
    # Save checkpoint
    # --------------------------------------------------------

    checkpoint = {

        "model_state_dict":
            model.state_dict(),

        "input_size":
            len(feature_columns),

        "hidden_size":
            HIDDEN_SIZE,

        "num_layers":
            NUM_LAYERS,

        "dropout":
            DROPOUT,

        "sequence_length":
            SEQUENCE_LENGTH,

        "feature_columns":
            feature_columns,
    }

    torch.save(
        checkpoint,
        MODEL_FILE,
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    results = {

        "model":
            "LSTM",

        "dataset":
            "NASA C-MAPSS FD001",

        "training_engines":
            int(len(train_engines)),

        "validation_engines":
            int(len(validation_engines)),

        "training_sequences":
            int(len(train_dataset)),

        "validation_sequences":
            int(len(validation_dataset)),

        "sequence_length":
            SEQUENCE_LENGTH,

        "feature_count":
            len(feature_columns),

        "metrics":
            final_metrics,

        "training_time_seconds":
            float(
                total_training_time
            ),

        "inference_time_seconds":
            float(
                inference_time
            ),

        "history":
            history,
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

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n")

    print("=" * 70)

    print(
        "FINAL LSTM RESULT"
    )

    print("=" * 70)

    print(
        f"MAE: "
        f"{final_metrics['MAE']:.4f}"
    )

    print(
        f"RMSE: "
        f"{final_metrics['RMSE']:.4f}"
    )

    print(
        f"R²: "
        f"{final_metrics['R2']:.4f}"
    )

    print(
        f"NASA Score: "
        f"{final_metrics['NASA_score']:.2f}"
    )

    print(
        f"Training time: "
        f"{total_training_time:.2f}s"
    )

    print(
        f"Inference time: "
        f"{inference_time:.4f}s"
    )

    print(
        f"\nModel saved to:"
        f"\n{MODEL_FILE}"
    )

    print(
        f"\nScaler saved to:"
        f"\n{SCALER_FILE}"
    )

    print(
        f"\nMetrics saved to:"
        f"\n{METRICS_FILE}"
    )


if __name__ == "__main__":

    main()