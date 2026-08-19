import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.config import (
    PROCESSED_DATA_DIR,
    FIGURE_DIR,
    SENSOR_COLUMNS,
    create_directories,
)


def load_data():

    path = (
        PROCESSED_DATA_DIR
        / "train_rul.csv"
    )

    return pd.read_csv(path)


def sensor_statistics(df):

    rows = []

    for sensor in SENSOR_COLUMNS:

        series = df[sensor]

        rows.append(
            {
                "sensor": sensor,
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "max": series.max(),
                "unique_values": series.nunique(),
            }
        )

    stats = pd.DataFrame(rows)

    stats.to_csv(
        FIGURE_DIR.parent
        / "sensor_statistics.csv",
        index=False,
    )

    return stats


def correlation_analysis(df):

    correlations = (
        df[
            SENSOR_COLUMNS + ["RUL"]
        ]
        .corr()["RUL"]
        .drop("RUL")
        .sort_values()
    )

    print("\nSensor/RUL correlations:")
    print(
        correlations.to_string()
    )

    return correlations


def plot_rul_distribution(df):

    plt.figure(figsize=(10, 6))

    plt.hist(
        df["RUL"],
        bins=40,
    )

    plt.xlabel(
        "Remaining Useful Life (cycles)"
    )

    plt.ylabel(
        "Number of observations"
    )

    plt.title(
        "RUL Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "rul_distribution.png",
        dpi=150,
    )

    plt.close()


def plot_engine_degradation(
    df,
    engine_id,
):

    engine = (
        df[
            df["unit"] == engine_id
        ]
        .sort_values("cycle")
    )

    sensors = [
        "s2",
        "s3",
        "s4",
        "s7",
        "s11",
        "s12",
        "s15",
    ]

    fig, axes = plt.subplots(
        len(sensors),
        1,
        figsize=(12, 18),
    )

    for ax, sensor in zip(
        axes,
        sensors,
    ):

        ax.plot(
            engine["cycle"],
            engine[sensor],
        )

        ax.set_ylabel(sensor)

        ax.grid(
            alpha=0.3
        )

    axes[-1].set_xlabel(
        "Cycle"
    )

    fig.suptitle(
        f"Engine {engine_id} Sensor Degradation",
        fontsize=16,
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / f"engine_{engine_id}_degradation.png",
        dpi=150,
    )

    plt.close()


def plot_sensor_correlations(
    correlations,
):

    plt.figure(figsize=(10, 7))

    correlations.plot(
        kind="barh"
    )

    plt.xlabel(
        "Pearson correlation with RUL"
    )

    plt.title(
        "Sensor Correlation with Remaining Useful Life"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "sensor_rul_correlation.png",
        dpi=150,
    )

    plt.close()


def main():

    create_directories()

    df = load_data()

    print("=" * 60)
    print("C-MAPSS EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Engines: {df['unit'].nunique()}"
    )

    stats = sensor_statistics(df)

    print("\nSensor statistics:")
    print(stats.to_string(index=False))

    correlations = correlation_analysis(
        df
    )

    plot_rul_distribution(df)

    plot_sensor_correlations(
        correlations
    )

    # Plot several representative engines
    for engine_id in [
        1,
        10,
        50,
        100,
    ]:

        plot_engine_degradation(
            df,
            engine_id,
        )

    print(
        "\nEDA completed."
    )

    print(
        f"Figures saved to:\n{FIGURE_DIR}"
    )


if __name__ == "__main__":

    main()