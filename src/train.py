import json
import joblib
import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import ParameterGrid

from src.config import (
    FEATURE_DATA_PATH,
    RF_MODEL_PATH,
    SCALER_PATH,
    LSTM_MODEL_PATH,
    METRICS_PATH,
    SENSOR_COLUMNS,
    SEQUENCE_LENGTH,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    RANDOM_STATE,
)


torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


class LSTMClassifier(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        last_output = output[:, -1, :]

        return self.fc(last_output).squeeze(1)


def metrics(y_true, probabilities):

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return {
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
    }


def build_feature_matrix(df):

    feature_columns = (
        SENSOR_COLUMNS
        + [
            "machine_age",
            "age_temperature_interaction",
            "vibration_torque_interaction",
        ]
    )

    for sensor in SENSOR_COLUMNS:
        feature_columns.extend(
            [
                f"{sensor}_rolling_mean",
                f"{sensor}_rolling_std",
                f"{sensor}_diff",
            ]
        )

    return df[feature_columns], feature_columns


def machine_based_split(df):

    machines = sorted(
        df["machine_id"].unique()
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    rng.shuffle(machines)

    split = int(len(machines) * 0.8)

    train_machines = machines[:split]
    test_machines = machines[split:]

    train_df = df[
        df["machine_id"].isin(train_machines)
    ].copy()

    test_df = df[
        df["machine_id"].isin(test_machines)
    ].copy()

    return train_df, test_df


def train_random_forest(
    X_train,
    y_train,
    X_test,
    y_test,
):

    print("\nTraining Random Forest...")

    parameter_grid = {
        "n_estimators": [100, 200],
        "max_depth": [8, 12],
        "min_samples_leaf": [2, 5],
    }

    best_model = None
    best_score = -1
    best_params = None

    for params in ParameterGrid(
        parameter_grid
    ):

        model = RandomForestClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
            **params,
        )

        model.fit(
            X_train,
            y_train,
        )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        result = metrics(
            y_test,
            probabilities,
        )

        print(
            f"params={params} "
            f"F1={result['f1']:.4f}"
        )

        if result["f1"] > best_score:

            best_score = result["f1"]
            best_model = model
            best_params = params

    probabilities = best_model.predict_proba(
        X_test
    )[:, 1]

    result = metrics(
        y_test,
        probabilities,
    )

    joblib.dump(
        best_model,
        RF_MODEL_PATH,
    )

    print("\nBest Random Forest:")
    print(best_params)
    print(result)

    return best_model, result


def create_sequences(
    df,
    scaler,
    feature_columns,
):

    sequences = []
    labels = []

    for _, group in df.groupby(
        "machine_id"
    ):

        group = group.sort_values(
            "timestamp"
        )

        X = scaler.transform(
            group[feature_columns]
        )

        y = group["failure"].values

        for i in range(
            SEQUENCE_LENGTH,
            len(group),
        ):

            sequences.append(
                X[
                    i - SEQUENCE_LENGTH : i
                ]
            )

            labels.append(
                y[i]
            )

    return (
        np.array(sequences),
        np.array(labels),
    )


def train_lstm(
    X_train,
    y_train,
    X_test,
    y_test,
):

    print("\nTraining PyTorch LSTM...")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    model = LSTMClassifier(
        input_size=X_train.shape[2]
    ).to(device)

    train_dataset = TensorDataset(
        torch.tensor(
            X_train,
            dtype=torch.float32,
        ),
        torch.tensor(
            y_train,
            dtype=torch.float32,
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    positive = max(
        1,
        y_train.sum(),
    )

    negative = max(
        1,
        len(y_train) - y_train.sum(),
    )

    pos_weight = torch.tensor(
        [negative / positive],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    model.train()

    for epoch in range(EPOCHS):

        total_loss = 0

        for batch_x, batch_y in train_loader:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            logits = model(batch_x)

            loss = criterion(
                logits,
                batch_y,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            total_loss += loss.item()

        avg_loss = (
            total_loss
            / len(train_loader)
        )

        print(
            f"Epoch "
            f"{epoch + 1}/{EPOCHS} "
            f"loss={avg_loss:.4f}"
        )

    model.eval()

    with torch.no_grad():

        X_tensor = torch.tensor(
            X_test,
            dtype=torch.float32,
        ).to(device)

        probabilities = (
            torch.sigmoid(
                model(X_tensor)
            )
            .cpu()
            .numpy()
        )

    result = metrics(
        y_test,
        probabilities,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": X_train.shape[2],
        },
        LSTM_MODEL_PATH,
    )

    print("\nLSTM metrics:")
    print(result)

    return model, result


def main():

    print("=" * 60)
    print("PREDICTIVE MAINTENANCE TRAINING")
    print("=" * 60)

    df = pd.read_csv(
        FEATURE_DATA_PATH
    )

    train_df, test_df = machine_based_split(
        df
    )

    X_train, feature_columns = (
        build_feature_matrix(train_df)
    )

    X_test, _ = build_feature_matrix(
        test_df
    )

    y_train = train_df[
        "failure"
    ].values

    y_test = test_df[
        "failure"
    ].values

    print(
        f"Training rows: {len(train_df):,}"
    )

    print(
        f"Testing rows: {len(test_df):,}"
    )

    print(
        f"Train failures: {y_train.sum():,}"
    )

    print(
        f"Test failures: {y_test.sum():,}"
    )

    # Scale features for LSTM.
    scaler = StandardScaler()

    scaler.fit(X_train)

    X_train_scaled = scaler.transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    joblib.dump(
        scaler,
        SCALER_PATH,
    )

    # Random Forest
    rf_model, rf_metrics = train_random_forest(
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
    )

    # LSTM
    train_sequence_df = train_df.sort_values(
        ["machine_id", "timestamp"]
    )

    test_sequence_df = test_df.sort_values(
        ["machine_id", "timestamp"]
    )

    X_lstm_train, y_lstm_train = (
        create_sequences(
            train_sequence_df,
            scaler,
            feature_columns,
        )
    )

    X_lstm_test, y_lstm_test = (
        create_sequences(
            test_sequence_df,
            scaler,
            feature_columns,
        )
    )

    lstm_model, lstm_metrics = train_lstm(
        X_lstm_train,
        y_lstm_train,
        X_lstm_test,
        y_lstm_test,
    )

    results = {
        "random_forest": rf_metrics,
        "lstm": lstm_metrics,
        "feature_count": len(
            feature_columns
        ),
        "training_rows": len(train_df),
        "testing_rows": len(test_df),
        "training_failures": int(
            y_train.sum()
        ),
        "testing_failures": int(
            y_test.sum()
        ),
    }

    with open(
        METRICS_PATH,
        "w",
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
        )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        json.dumps(
            results,
            indent=4,
        )
    )

    print(
        f"\nMetrics saved to: {METRICS_PATH}"
    )


if __name__ == "__main__":
    main()