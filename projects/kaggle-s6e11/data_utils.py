from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    BOOLEAN_COLUMNS,
    CATEGORICAL_COLUMNS,
    ID_COL,
    SAMPLE_SUBMISSION_PATH,
    TARGET_COL,
    TEST_PATH,
    TRAIN_PATH,
)


def add_common_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add row-wise features only; no target or test-set statistics are used."""
    result = frame.copy()

    result["amount_balance_ratio"] = (
        result["amount_usd"] / (result["account_balance_usd"].abs() + 1.0)
    )
    result["distance_per_hour"] = result["distance_from_home_km"] / (
        result["hours_since_last_txn"] + 0.1
    )
    result["txn_velocity_interaction"] = (
        result["txn_count_last_24h"] * result["velocity_score"]
    )
    result["risk_flag_count"] = (
        result[
            [
                "used_vpn",
                "ip_country_mismatch",
                "billing_shipping_mismatch",
                "is_new_merchant",
                "is_foreign_transaction",
                "is_ai_generated_scam_attempt",
            ]
        ]
        .astype(np.int8)
        .sum(axis=1)
    )

    for column in BOOLEAN_COLUMNS:
        result[column] = result[column].astype(np.int8)

    return result


def add_engineered_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add row-wise v2 features and low-cardinality category combinations."""
    result = add_common_features(frame)
    result["balance_after_amount"] = (
        result["account_balance_usd"] - result["amount_usd"]
    )
    result["balance_shortfall_ratio"] = (
        np.maximum(result["amount_usd"] - result["account_balance_usd"], 0.0)
        / (result["account_balance_usd"].abs() + 1.0)
    )
    result["amount_to_age_ratio"] = (
        result["amount_usd"] / (result["customer_age"] + 1.0)
    )
    result["amount_per_txn_24h"] = (
        result["amount_usd"] / (result["txn_count_last_24h"] + 1.0)
    )
    result["log_amount_usd"] = np.log1p(
        result["amount_usd"].clip(lower=0.0)
    )
    result["log_distance_from_home_km"] = np.log1p(
        result["distance_from_home_km"].clip(lower=0.0)
    )
    result["log_hours_since_last_txn"] = np.log1p(
        result["hours_since_last_txn"].clip(lower=0.0)
    )
    result["txn_rate_per_hour"] = (
        result["txn_count_last_24h"]
        / (result["hours_since_last_txn"].abs() + 1.0)
    )
    result["customer_age_to_card_age"] = (
        result["customer_age"] / (result["card_age_months"] + 1.0)
    )
    result["risk_velocity_product"] = (
        result["merchant_risk_score"] * result["velocity_score"]
    )
    result["risk_flag_score"] = (
        result["merchant_risk_score"] * result["risk_flag_count"]
    )
    result["has_cvv_retry"] = (
        result["cvv_retry_count"] > 0
    ).astype(np.int8)
    result["hour_sin"] = np.sin(
        2.0 * np.pi * result["time_of_day_hour"] / 24.0
    )
    result["hour_cos"] = np.cos(
        2.0 * np.pi * result["time_of_day_hour"] / 24.0
    )
    result["is_night"] = result["time_of_day_hour"].between(0, 5).astype(
        np.int8
    )

    combo_columns = [
        "merchant_category_x_auth_method",
        "device_type_x_channel",
        "card_type_x_merchant_category",
    ]
    result[combo_columns[0]] = (
        result["merchant_category"].astype(str)
        + "__"
        + result["auth_method"].astype(str)
    )
    result[combo_columns[1]] = (
        result["device_type"].astype(str)
        + "__"
        + result["channel"].astype(str)
    )
    result[combo_columns[2]] = (
        result["card_type"].astype(str)
        + "__"
        + result["merchant_category"].astype(str)
    )

    numeric_columns = result.select_dtypes(include=[np.number]).columns
    result[numeric_columns] = result[numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    return result, combo_columns


def add_conservative_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add only six low-risk row-wise interactions to the v1 feature set."""
    result = add_common_features(frame)
    result["balance_shortfall_ratio"] = (
        np.maximum(result["amount_usd"] - result["account_balance_usd"], 0.0)
        / (result["account_balance_usd"].abs() + 1.0)
    )
    result["amount_per_txn_24h"] = (
        result["amount_usd"] / (result["txn_count_last_24h"] + 1.0)
    )
    result["txn_rate_per_hour"] = (
        result["txn_count_last_24h"]
        / (result["hours_since_last_txn"].abs() + 1.0)
    )
    result["risk_velocity_product"] = (
        result["merchant_risk_score"] * result["velocity_score"]
    )
    result["risk_flag_score"] = (
        result["merchant_risk_score"] * result["risk_flag_count"]
    )
    result["has_cvv_retry"] = (
        result["cvv_retry_count"] > 0
    ).astype(np.int8)
    return result


def load_data(feature_set: str = "v1") -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    list[str],
]:
    for path in (TRAIN_PATH, TEST_PATH, SAMPLE_SUBMISSION_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing data file: {path}")

    train = pd.read_csv(TRAIN_PATH).sort_values(ID_COL).reset_index(drop=True)
    test = pd.read_csv(TEST_PATH)
    sample = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    if TARGET_COL not in train.columns:
        raise ValueError(f"Training data must contain target column: {TARGET_COL}")
    if TARGET_COL in test.columns:
        raise ValueError("Test data unexpectedly contains the target column.")
    if not train[ID_COL].is_unique or not test[ID_COL].is_unique:
        raise ValueError("IDs must be unique in both training and test data.")
    if set(train[ID_COL]).intersection(set(test[ID_COL])):
        raise ValueError("Training and test IDs overlap.")
    if set(test[ID_COL]) != set(sample[ID_COL]):
        raise ValueError("sample_submission IDs do not match test.csv IDs.")

    y = train[TARGET_COL].astype(np.int8).copy()
    train_ids = train[ID_COL].copy()
    test_ids = test[ID_COL].copy()

    raw_train = train.drop(columns=[ID_COL, TARGET_COL])
    raw_test = test.drop(columns=[ID_COL])
    if feature_set == "v1":
        X_train = add_common_features(raw_train)
        X_test = add_common_features(raw_test)
        extra_categorical_columns: list[str] = []
    elif feature_set == "v2":
        X_train, extra_train_categorical = add_engineered_features(raw_train)
        X_test, extra_test_categorical = add_engineered_features(raw_test)
        if extra_train_categorical != extra_test_categorical:
            raise ValueError("Engineered categorical columns differ by dataset.")
        extra_categorical_columns = extra_train_categorical
    elif feature_set == "v3":
        X_train = add_conservative_features(raw_train)
        X_test = add_conservative_features(raw_test)
        extra_categorical_columns = []
    elif feature_set == "v4":
        removed_columns = ["card_type", "channel", "device_type"]
        X_train = add_common_features(raw_train).drop(columns=removed_columns)
        X_test = add_common_features(raw_test).drop(columns=removed_columns)
        extra_categorical_columns = []
    else:
        raise ValueError(f"Unsupported feature set: {feature_set}")

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Feature columns differ between train and test.")
    if not np.isfinite(X_train.select_dtypes(include=[np.number])).all().all():
        raise ValueError("Non-finite values found in training features.")
    if not np.isfinite(X_test.select_dtypes(include=[np.number])).all().all():
        raise ValueError("Non-finite values found in test features.")

    categorical_columns = [
        column for column in CATEGORICAL_COLUMNS if column in X_train.columns
    ]
    categorical_columns.extend(extra_categorical_columns)
    return X_train, y, train_ids, test_ids, X_test, categorical_columns


def time_ordered_split(
    X: pd.DataFrame,
    y: pd.Series,
    holdout_fraction: float,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, np.ndarray]:
    if not 0.05 <= holdout_fraction <= 0.4:
        raise ValueError("holdout_fraction must be between 0.05 and 0.4")

    split_index = int(len(X) * (1.0 - holdout_fraction))
    fit_indices = np.arange(split_index)
    valid_indices = np.arange(split_index, len(X))

    X_fit = X.iloc[fit_indices].reset_index(drop=True)
    y_fit = y.iloc[fit_indices].reset_index(drop=True)
    X_valid = X.iloc[valid_indices].reset_index(drop=True)
    y_valid = y.iloc[valid_indices].reset_index(drop=True)

    if y_fit.nunique() < 2 or y_valid.nunique() < 2:
        raise ValueError("Both training and validation parts must contain both classes.")
    return X_fit, y_fit, X_valid, y_valid, valid_indices


def apply_train_categories(
    X_fit: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_full: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use categories learned only from the training partition."""
    fit = X_fit.copy()
    valid = X_valid.copy()
    full = X_full.copy()
    test = X_test.copy()

    for column in categorical_columns:
        categories = pd.Index(fit[column].dropna().unique()).sort_values()
        dtype = pd.CategoricalDtype(categories=categories)
        fit[column] = fit[column].astype(dtype)
        valid[column] = valid[column].astype(dtype)
        full[column] = full[column].astype(dtype)
        test[column] = test[column].astype(dtype)

    return fit, valid, full, test


def validate_submission(submission: pd.DataFrame, expected_ids: pd.Series) -> None:
    expected_columns = [ID_COL, TARGET_COL]
    if list(submission.columns) != expected_columns:
        raise ValueError(f"Submission columns must be exactly {expected_columns}")
    if len(submission) != len(expected_ids):
        raise ValueError("Submission row count does not match test.csv.")
    if not submission[ID_COL].reset_index(drop=True).equals(
        expected_ids.reset_index(drop=True)
    ):
        raise ValueError("Submission IDs do not match the order in test.csv.")
    if submission[TARGET_COL].isna().any():
        raise ValueError("Submission contains missing predictions.")
    if not submission[TARGET_COL].between(0.0, 1.0).all():
        raise ValueError("Probabilities must be in [0, 1].")


def write_submission(
    output_path: Path,
    expected_ids: pd.Series,
    probabilities: np.ndarray,
) -> None:
    submission = pd.DataFrame(
        {
            ID_COL: expected_ids.to_numpy(),
            TARGET_COL: probabilities.astype(np.float64),
        }
    )
    validate_submission(submission, expected_ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
