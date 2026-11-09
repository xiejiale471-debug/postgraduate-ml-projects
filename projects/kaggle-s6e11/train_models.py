from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import (
    MODEL_DIR,
    MODEL_NAMES,
    OUTPUT_DIR,
    SUBMISSION_DIR,
    TARGET_COL,
    create_output_dirs,
)
from data_utils import (
    apply_train_categories,
    load_data,
    time_ordered_split,
    write_submission,
)


def predict_positive_probability(model, frame: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(frame)[:, 1]
    return np.asarray(probabilities, dtype=np.float64)


def evaluate(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "validation_auc": float(roc_auc_score(y_true, probabilities)),
        "validation_logloss": float(log_loss(y_true, probabilities)),
    }


def save_model(name: str, model, run_dir: Path) -> Path:
    path_map = {
        "catboost": run_dir / "model.cbm",
        "lightgbm": run_dir / "model.txt",
        "xgboost": run_dir / "model.json",
        "random_forest": run_dir / "model.joblib",
    }
    model_path = path_map[name]

    if name == "catboost":
        model.save_model(str(model_path))
    elif name == "lightgbm":
        model.booster_.save_model(str(model_path))
    elif name == "xgboost":
        model.save_model(str(model_path))
    else:
        joblib.dump(model, model_path)
    return model_path


def run_catboost(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_full: pd.DataFrame,
    y_full: pd.Series,
    X_test: pd.DataFrame,
    categorical_columns: list[str],
    args: argparse.Namespace,
) -> tuple[object, dict[str, float], int, np.ndarray]:
    from catboost import CatBoostClassifier

    iterations = args.quick_iterations if args.quick else 3000
    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=4.0,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=args.seed,
        early_stopping_rounds=min(200, max(20, iterations // 5)),
        verbose=100,
        thread_count=args.n_jobs,
        allow_writing_files=False,
    )
    model.fit(
        X_fit,
        y_fit,
        cat_features=categorical_columns,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
    )
    best_iterations = int(model.get_best_iteration()) + 1
    valid_probabilities = predict_positive_probability(model, X_valid)
    metrics = evaluate(y_valid, valid_probabilities)
    metrics["best_iteration"] = best_iterations
    if args.validation_only:
        return model, metrics, best_iterations, valid_probabilities

    final_model = CatBoostClassifier(
        iterations=max(1, best_iterations),
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=4.0,
        loss_function="Logloss",
        random_seed=args.seed,
        verbose=100,
        thread_count=args.n_jobs,
        allow_writing_files=False,
    )
    final_model.fit(X_full, y_full, cat_features=categorical_columns)
    return final_model, metrics, best_iterations, valid_probabilities


def run_lightgbm(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_full: pd.DataFrame,
    y_full: pd.Series,
    X_test: pd.DataFrame,
    categorical_columns: list[str],
    args: argparse.Namespace,
) -> tuple[object, dict[str, float], int, np.ndarray]:
    import lightgbm as lgb

    iterations = args.quick_iterations if args.quick else 3000
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=iterations,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        verbosity=-1,
    )
    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        categorical_feature=categorical_columns,
        callbacks=[
            lgb.early_stopping(min(200, max(20, iterations // 5)), verbose=True),
            lgb.log_evaluation(100),
        ],
    )
    best_iterations = int(model.best_iteration_ or iterations)
    valid_probabilities = predict_positive_probability(model, X_valid)
    metrics = evaluate(y_valid, valid_probabilities)
    metrics["best_iteration"] = best_iterations
    if args.validation_only:
        return model, metrics, best_iterations, valid_probabilities

    final_model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=max(1, best_iterations),
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        verbosity=-1,
    )
    final_model.fit(
        X_full,
        y_full,
        categorical_feature=categorical_columns,
    )
    return final_model, metrics, best_iterations, valid_probabilities


def run_xgboost(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_full: pd.DataFrame,
    y_full: pd.Series,
    X_test: pd.DataFrame,
    categorical_columns: list[str],
    args: argparse.Namespace,
) -> tuple[object, dict[str, float], int, np.ndarray]:
    from xgboost import XGBClassifier

    iterations = args.quick_iterations if args.quick else 3000
    early_stopping_rounds = min(200, max(20, iterations // 5))
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=iterations,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        max_cat_to_onehot=16,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        early_stopping_rounds=early_stopping_rounds,
    )
    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_valid, y_valid)],
        verbose=100,
    )
    best_iterations = int(model.best_iteration) + 1
    valid_probabilities = predict_positive_probability(model, X_valid)
    metrics = evaluate(y_valid, valid_probabilities)
    metrics["best_iteration"] = best_iterations
    if args.validation_only:
        return model, metrics, best_iterations, valid_probabilities

    final_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=max(1, best_iterations),
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        max_cat_to_onehot=16,
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )
    final_model.fit(X_full, y_full, verbose=False)
    return final_model, metrics, best_iterations, valid_probabilities


def run_random_forest(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_full: pd.DataFrame,
    y_full: pd.Series,
    X_test: pd.DataFrame,
    categorical_columns: list[str],
    args: argparse.Namespace,
) -> tuple[object, dict[str, float], int, np.ndarray]:
    estimators = 30 if args.quick else 350

    def make_pipeline() -> Pipeline:
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                    categorical_columns,
                )
            ],
            remainder="passthrough",
            sparse_threshold=1.0,
        )
        classifier = RandomForestClassifier(
            n_estimators=estimators,
            max_features="sqrt",
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=args.n_jobs,
            random_state=args.seed,
            verbose=1,
        )
        return Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )

    model = make_pipeline()
    model.fit(X_fit, y_fit)
    valid_probabilities = predict_positive_probability(model, X_valid)
    metrics = evaluate(y_valid, valid_probabilities)
    metrics["best_iteration"] = estimators
    if args.validation_only:
        return model, metrics, estimators, valid_probabilities

    final_model = make_pipeline()
    final_model.fit(X_full, y_full)
    return final_model, metrics, estimators, valid_probabilities


RUNNERS = {
    "catboost": run_catboost,
    "lightgbm": run_lightgbm,
    "xgboost": run_xgboost,
    "random_forest": run_random_forest,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train four models for Kaggle Playground S6E11."
    )
    parser.add_argument(
        "--model",
        choices=[*MODEL_NAMES, "all"],
        default="all",
        help="Model to train, or all models sequentially.",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.2,
        help="Last fraction of ID-sorted training rows used for validation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--feature-set",
        choices=["v1", "v2", "v3", "v4"],
        default="v1",
        help=(
            "v1 keeps the original 28-feature pipeline; v2 enables all "
            "candidate features; v3 enables six conservative interactions; "
            "v4 removes three low-gain category columns."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Thread count passed to model libraries.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run fewer iterations for an end-to-end smoke test.",
    )
    parser.add_argument(
        "--quick-iterations",
        type=int,
        default=80,
        help="Maximum boosting iterations in quick mode.",
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help=(
            "Train only the 80%% fit model, save validation predictions, "
            "and skip full retraining and test submission."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_output_dirs()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    X_train, y_train, train_ids, test_ids, X_test, categorical_columns = load_data(
        feature_set=args.feature_set
    )
    X_fit, y_fit, X_valid, y_valid, valid_indices = time_ordered_split(
        X_train,
        y_train,
        args.holdout_fraction,
    )

    print(
        f"train={len(train_ids):,}, test={len(test_ids):,}, "
        f"fit={len(X_fit):,}, valid={len(X_valid):,}"
    )
    print(f"positive_rate={y_train.mean():.6f}, categories={categorical_columns}")
    print(
        "validation id range: "
        f"{int(train_ids.iloc[valid_indices[0]])} - "
        f"{int(train_ids.iloc[valid_indices[-1]])}"
    )

    selected_models = list(MODEL_NAMES) if args.model == "all" else [args.model]
    summary_path = run_dir / "metrics.csv"
    all_results: list[dict[str, object]] = []

    for model_name in selected_models:
        print("\n" + "=" * 72)
        print(f"Training {model_name}")
        print("=" * 72)

        started_at = time.perf_counter()
        if model_name in {"lightgbm", "xgboost"}:
            fit_data, valid_data, full_data, test_data = apply_train_categories(
                X_fit,
                X_valid,
                X_train,
                X_test,
                categorical_columns,
            )
        else:
            fit_data, valid_data, full_data, test_data = (
                X_fit.copy(),
                X_valid.copy(),
                X_train.copy(),
                X_test.copy(),
            )

        model, metrics, best_iteration, valid_probabilities = RUNNERS[model_name](
            fit_data,
            y_fit,
            valid_data,
            y_valid,
            full_data,
            y_train,
            test_data,
            categorical_columns,
            args,
        )

        elapsed_seconds = time.perf_counter() - started_at

        model_run_dir = MODEL_DIR / f"{run_id}_{model_name}"
        model_run_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_model(model_name, model, model_run_dir)

        valid_predictions = pd.DataFrame(
            {
                "id": train_ids.iloc[valid_indices].to_numpy(),
                "y_true": y_valid.to_numpy(),
                "y_prob": valid_probabilities,
            }
        )
        valid_predictions.to_csv(
            model_run_dir / "valid_predictions.csv",
            index=False,
        )

        submission_path: Path | None = None
        if not args.validation_only:
            # The test set is touched only here, after all fitting is complete.
            test_probabilities = predict_positive_probability(model, test_data)
            submission_path = (
                SUBMISSION_DIR / f"submission_{model_name}_{run_id}.csv"
            )
            write_submission(submission_path, test_ids, test_probabilities)

        metadata = {
            "run_id": run_id,
            "model": model_name,
            "quick": args.quick,
            "feature_set": args.feature_set,
            "validation_only": args.validation_only,
            "holdout_fraction": args.holdout_fraction,
            "validation_auc": metrics["validation_auc"],
            "validation_logloss": metrics["validation_logloss"],
            "best_iteration": best_iteration,
            "elapsed_seconds": elapsed_seconds,
            "model_file": str(model_path),
            "validation_predictions_file": str(
                model_run_dir / "valid_predictions.csv"
            ),
            "submission_file": (
                str(submission_path) if submission_path is not None else None
            ),
            "feature_columns": list(X_train.columns),
            "categorical_columns": categorical_columns,
        }
        with (model_run_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)

        all_results.append(
            {
                "model": model_name,
                "feature_set": args.feature_set,
                "validation_only": args.validation_only,
                "validation_auc": metrics["validation_auc"],
                "validation_logloss": metrics["validation_logloss"],
                "best_iteration": best_iteration,
                "elapsed_seconds": elapsed_seconds,
                "model_file": str(model_path),
                "submission_file": (
                    str(submission_path) if submission_path is not None else None
                ),
            }
        )
        pd.DataFrame(all_results).sort_values(
            "validation_auc", ascending=False
        ).to_csv(summary_path, index=False)

        print(
            f"{model_name}: AUC={metrics['validation_auc']:.6f}, "
            f"LogLoss={metrics['validation_logloss']:.6f}, "
            f"time={elapsed_seconds:.1f}s"
        )
        if submission_path is not None:
            print(f"Submission: {submission_path}")
        else:
            print("Validation-only mode: test submission was skipped.")

    print("\nAll requested models finished.")
    print(f"Metrics summary: {summary_path}")


if __name__ == "__main__":
    main()
