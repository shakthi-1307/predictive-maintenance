import json
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from src.config import (
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    METRIC_DIR,
    create_directories,
)


FEATURE_FILE = (
    PROCESSED_DATA_DIR / "train_features.csv"
)

MODEL_FILE = (
    MODEL_DIR / "random_forest_rul.joblib"
)

OUTPUT_DIR = (
    METRIC_DIR.parent / "figures"
)

IMPORTANCE_FILE = (
    METRIC_DIR / "feature_importance.json"
)


def main():

    create_directories()
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("RANDOM FOREST SHAP EXPLAINABILITY")
    print("=" * 70)

    df = pd.read_csv(
        FEATURE_FILE
    )

    model_data = joblib.load(
        MODEL_FILE
    )

    print("\nLoaded model checkpoint:")
    print(
        "Keys:",
        list(model_data.keys())
    )

    # Extract the sklearn pipeline
    pipeline = model_data["model"]

    print(
        "Pipeline:",
        type(pipeline)
    )

    # Extract the actual Random Forest
    model = pipeline.named_steps["model"]

    print(
        "Underlying model:",
        type(model)
    )
    # Take the column list from the checkpoint rather than
    # re-deriving it, so SHAP is guaranteed to line up with the
    # matrix the model was actually fitted on.
    feature_columns = model_data["features"]

    missing = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"train_features.csv is missing columns the model "
            f"was trained on: {missing}. Re-run "
            f"`python -m src.features`."
        )

    X = df[
        feature_columns
    ].copy()

    # Sample for SHAP to keep computation fast
    sample_size = min(
        3000,
        len(X)
    )

    X_sample = X.sample(
        sample_size,
        random_state=42,
    )

    print(
        f"\nFeatures: "
        f"{len(feature_columns)}"
    )

    print(
        f"SHAP samples: "
        f"{len(X_sample):,}"
    )

    print(
        "\nCalculating SHAP values..."
    )

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        X_sample
    )

    # --------------------------------------------------
    # Global importance
    # --------------------------------------------------

    importance = np.abs(
        shap_values
    ).mean(axis=0)

    importance_df = pd.DataFrame({

        "feature": feature_columns,

        "mean_abs_shap":
            importance,

    }).sort_values(
        "mean_abs_shap",
        ascending=False,
    )

    importance_df.to_csv(
        OUTPUT_DIR
        / "shap_feature_importance.csv",
        index=False,
    )

    # Save JSON
    top_features = (
        importance_df
        .head(20)
        .to_dict(
            orient="records"
        )
    )

    with open(
        IMPORTANCE_FILE,
        "w",
    ) as f:

        json.dump(
            top_features,
            f,
            indent=4,
        )

    # --------------------------------------------------
    # SHAP summary
    # --------------------------------------------------

    print(
        "\nGenerating SHAP summary..."
    )

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_sample,
        show=False,
        max_display=20,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "shap_summary.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------
    # Bar importance
    # --------------------------------------------------

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        show=False,
        max_display=20,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "shap_importance.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TOP 15 PREDICTIVE FEATURES")
    print("=" * 70)

    print(
        importance_df
        .head(15)
        .to_string(
            index=False
        )
    )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_DIR
        / "shap_summary.png"
    )

    print(
        OUTPUT_DIR
        / "shap_importance.png"
    )

    print(
        OUTPUT_DIR
        / "shap_feature_importance.csv"
    )


if __name__ == "__main__":
    main()