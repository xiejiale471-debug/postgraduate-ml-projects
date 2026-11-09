from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "playground-series-s6e11"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"
LOG_DIR = OUTPUT_DIR / "logs"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"

ID_COL = "id"
TARGET_COL = "is_fraud"

CATEGORICAL_COLUMNS = [
    "merchant_category",
    "card_type",
    "auth_method",
    "channel",
    "device_type",
]

BOOLEAN_COLUMNS = [
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]

MODEL_NAMES = ("catboost", "lightgbm", "xgboost", "random_forest")


def create_output_dirs() -> None:
    for path in (OUTPUT_DIR, MODEL_DIR, SUBMISSION_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)
