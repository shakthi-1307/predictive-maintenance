from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "CMAPSSData"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = BASE_DIR / "models"

REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
METRIC_DIR = REPORT_DIR / "metrics"


TRAIN_FILE = RAW_DATA_DIR / "train_FD001.txt"
TEST_FILE = RAW_DATA_DIR / "test_FD001.txt"
RUL_FILE = RAW_DATA_DIR / "RUL_FD001.txt"


# These must match the filenames the training scripts write.
RF_MODEL_PATH = MODEL_DIR / "random_forest_rul.joblib"
XGB_MODEL_PATH = MODEL_DIR / "xgboost_rul.json"
LSTM_MODEL_PATH = MODEL_DIR / "lstm_rul.pt"
LSTM_SCALER_PATH = MODEL_DIR / "lstm_scaler.joblib"


RUL_DATA_PATH = (
    PROCESSED_DATA_DIR / "train_rul.csv"
)

FEATURE_DATA_PATH = (
    PROCESSED_DATA_DIR / "train_features.csv"
)


RANDOM_STATE = 42


# NASA C-MAPSS columns
COLUMNS = [
    "unit",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
]

SENSOR_COLUMNS = [
    f"s{i}"
    for i in range(1, 22)
]


ALL_COLUMNS = COLUMNS + SENSOR_COLUMNS


# Sensors with useful variance in FD001.
# We'll confirm these empirically during EDA.
DEFAULT_USEFUL_SENSORS = [
    "s2",
    "s3",
    "s4",
    "s7",
    "s8",
    "s9",
    "s11",
    "s12",
    "s13",
    "s14",
    "s15",
    "s17",
    "s20",
    "s21",
]


# Columns that identify a row rather than describe its condition.
# These must never enter the feature matrix.
#
#   unit        -> id of a single truncated trajectory
#   source_unit -> id of the physical engine the trajectory came from
#
# `source_unit` is what train/validation splits must group on, otherwise
# truncations of the same engine leak across the split.
NON_FEATURE_COLUMNS = [
    "unit",
    "source_unit",
    "cycle",
    "RUL",
]


SEQUENCE_LENGTH = 30

BATCH_SIZE = 64

EPOCHS = 30

LEARNING_RATE = 1e-3


def create_directories():

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRIC_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )