from pathlib import Path


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

# Files
RAW_DATA_PATH = RAW_DATA_DIR / "sensor_data.csv"
FEATURE_DATA_PATH = PROCESSED_DATA_DIR / "features.csv"

RF_MODEL_PATH = MODEL_DIR / "random_forest.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"

LSTM_MODEL_PATH = MODEL_DIR / "lstm_model.pt"

METRICS_PATH = REPORT_DIR / "metrics.json"

# Reproducibility
RANDOM_STATE = 42

# Dataset
NUM_MACHINES = 50
TIMESTEPS_PER_MACHINE = 300

# Sensors
SENSOR_COLUMNS = [
    "temperature",
    "pressure",
    "vibration",
    "rotational_speed",
    "torque",
    "voltage",
    "current",
]

# Sequence configuration
SEQUENCE_LENGTH = 24

# Training
BATCH_SIZE = 64
EPOCHS = 12
LEARNING_RATE = 0.001

# Failure horizon
FAILURE_HORIZON = 24


def create_directories():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)