from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import rankdata
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import MODEL_DIR, OUTPUT_DIR, SUBMISSION_DIR, TARGET_COL
from data_utils import load_data, write_submission


BASELINE_RUN = OUTPUT_DIR / "run_20261104_221722"
EXPERIMENTS_DIR = OUTPUT_DIR / "experiments"


def set_categories(
    fit_frame: pd.DataFrame,
    other_frames: list[pd.DataFrame],
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    fit = fit_frame.copy()
    outputs = [frame.copy() for frame in other_frames]
    for column in categorical_columns:
        categories = pd.Index(fit[column].dropna().unique()).sort_values()
        dtype = pd.CategoricalDtype(categories=categories)
        fit[column] = fit[column].astype(dtype)
        for frame in outputs:
            frame[column] = frame[column].astype(dtype)
    return fit, outputs


def fit_lightgbm_fold(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    categorical_columns: list[str],
    seed: int,
    n_jobs: int,
    quick: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    iterations = 100 if quick else 3000
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=iterations,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=n_jobs,
        verbosity=-1,
    )
    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        categorical_feature=categorical_columns,
        callbacks=[
            lgb.early_stopping(20 if quick else 200, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    best_iteration = int(model.best_iteration_ or iterations)
    return model.predict_proba(X_valid)[:, 1], model, best_iteration


def fit_xgboost_fold(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    seed: int,
    n_jobs: int,
    quick: bool,
) -> tuple[np.ndarray, object, int]:
    iterations = 100 if quick else 3000
    model = xgb.XGBClassifier(
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
        random_state=seed,
        n_jobs=n_jobs,
        early_stopping_rounds=20 if quick else 200,
    )
    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )
    best_iteration = int(model.best_iteration) + 1
    return model.predict_proba(X_valid)[:, 1], model, best_iteration


def probability_blend(
    first: np.ndarray,
    second: np.ndarray,
    first_weight: float,
) -> np.ndarray:
    return first_weight * first + (1.0 - first_weight) * second


def rank_blend(
    first: np.ndarray,
    second: np.ndarray,
    first_weight: float,
) -> np.ndarray:
    first_rank = rankdata(first, method="average") / len(first)
    second_rank = rankdata(second, method="average") / len(second)
    return first_weight * first_rank + (1.0 - first_weight) * second_rank


def search_blend_weight(
    first: np.ndarray,
    second: np.ndarray,
    y_true: pd.Series,
    use_rank: bool,
) -> tuple[float, float]:
    best_weight = 0.5
    best_auc = -1.0
    for weight in np.linspace(0.0, 1.0, 101):
        if use_rank:
            prediction = rank_blend(first, second, float(weight))
        else:
            prediction = probability_blend(first, second, float(weight))
        score = roc_auc_score(y_true, prediction)
        if score > best_auc:
            best_auc = float(score)
            best_weight = float(weight)
    return best_weight, best_auc


def fit_final_lightgbm(
    X_full: pd.DataFrame,
    y_full: pd.Series,
    X_test: pd.DataFrame,
    categorical_columns: list[str],
    n_estimators: int,
    seed: int,
    n_jobs: int,
) -> tuple[object, np.ndarray]:
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=max(1, n_estimators),
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=n_jobs,
        verbosity=-1,
    )
    model.fit(X_full, y_full, categorical_feature=categorical_columns)
    return model, model.predict_proba(X_test)[:, 1]


def fit_final_xgboost(
    X_full: pd.DataFrame,
    y_full: pd.Series,
    X_test: pd.DataFrame,
    n_estimators: int,
    seed: int,
    n_jobs: int,
) -> tuple[object, np.ndarray]:
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=max(1, n_estimators),
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
        random_state=seed,
        n_jobs=n_jobs,
    )
    model.fit(X_full, y_full, verbose=False)
    return model, model.predict_proba(X_test)[:, 1]


def load_baseline_rows() -> pd.DataFrame:
    metrics_path = BASELINE_RUN / "metrics.csv"
    if not metrics_path.exists():
        return pd.DataFrame()
    baseline = pd.read_csv(metrics_path)
    return pd.DataFrame(
        {
            "source": "first_run",
            "model": baseline["model"],
            "evaluation": "time_holdout",
            "auc": baseline["validation_auc"],
            "logloss": baseline["validation_logloss"],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a 5-fold OOF ensemble for S6E11."
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = EXPERIMENTS_DIR / f"exp01_oof_ensemble_{run_id}"
    experiment_dir.mkdir(parents=True, exist_ok=False)
    model_output_dir = experiment_dir / "models"
    model_output_dir.mkdir()

    print("Loading data...")
    X_train, y_train, train_ids, test_ids, X_test, categorical_columns = load_data()

    splitter = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )
    oof_lightgbm = np.zeros(len(X_train), dtype=np.float64)
    oof_xgboost = np.zeros(len(X_train), dtype=np.float64)
    lightgbm_iterations: list[int] = []
    xgboost_iterations: list[int] = []

    started_at = time.perf_counter()
    for fold, (fit_index, valid_index) in enumerate(
        splitter.split(X_train, y_train),
        start=1,
    ):
        fold_started = time.perf_counter()
        X_fit = X_train.iloc[fit_index].reset_index(drop=True)
        X_valid = X_train.iloc[valid_index].reset_index(drop=True)
        y_fit = y_train.iloc[fit_index].reset_index(drop=True)
        y_valid = y_train.iloc[valid_index].reset_index(drop=True)

        X_fit_cat, (X_valid_cat,) = set_categories(
            X_fit,
            [X_valid],
            categorical_columns,
        )

        lightgbm_prob, _, lightgbm_best = fit_lightgbm_fold(
            X_fit_cat,
            y_fit,
            X_valid_cat,
            y_valid,
            categorical_columns,
            args.seed + fold,
            args.n_jobs,
            args.quick,
        )
        oof_lightgbm[valid_index] = lightgbm_prob
        lightgbm_iterations.append(lightgbm_best)

        xgboost_prob, _, xgboost_best = fit_xgboost_fold(
            X_fit_cat,
            y_fit,
            X_valid_cat,
            y_valid,
            args.seed + fold,
            args.n_jobs,
            args.quick,
        )
        oof_xgboost[valid_index] = xgboost_prob
        xgboost_iterations.append(xgboost_best)

        print(
            f"fold={fold}/{args.folds} "
            f"lgb_auc={roc_auc_score(y_valid, lightgbm_prob):.6f} "
            f"xgb_auc={roc_auc_score(y_valid, xgboost_prob):.6f} "
            f"time={time.perf_counter() - fold_started:.1f}s"
        )

    oof_lgb_auc = float(roc_auc_score(y_train, oof_lightgbm))
    oof_xgb_auc = float(roc_auc_score(y_train, oof_xgboost))

    prob_weight, prob_auc = search_blend_weight(
        oof_lightgbm,
        oof_xgboost,
        y_train,
        use_rank=False,
    )
    rank_weight, rank_auc = search_blend_weight(
        oof_lightgbm,
        oof_xgboost,
        y_train,
        use_rank=True,
    )

    candidate_predictions = {
        "lightgbm_oof": oof_lightgbm,
        "xgboost_oof": oof_xgboost,
        "simple_probability_blend": probability_blend(
            oof_lightgbm,
            oof_xgboost,
            0.5,
        ),
        "optimized_probability_blend": probability_blend(
            oof_lightgbm,
            oof_xgboost,
            prob_weight,
        ),
        "rank_blend": rank_blend(
            oof_lightgbm,
            oof_xgboost,
            rank_weight,
        ),
    }
    candidates = {
        name: (
            float(roc_auc_score(y_train, prediction)),
            (
                "rank"
                if name == "rank_blend"
                else "probability"
            ),
        )
        for name, prediction in candidate_predictions.items()
    }
    best_method = max(candidates, key=lambda name: candidates[name][0])
    best_auc = candidates[best_method][0]
    best_transform = candidates[best_method][1]
    best_weight = rank_weight if best_transform == "rank" else prob_weight

    if best_transform == "rank":
        oof_blend = rank_blend(oof_lightgbm, oof_xgboost, best_weight)
    else:
        oof_blend = probability_blend(
            oof_lightgbm,
            oof_xgboost,
            best_weight,
        )

    # Final category schema is learned from the complete training set.
    X_full_cat, (X_test_cat,) = set_categories(
        X_train,
        [X_test],
        categorical_columns,
    )
    final_lgb_iterations = max(1, int(round(np.mean(lightgbm_iterations))))
    final_xgb_iterations = max(1, int(round(np.mean(xgboost_iterations))))

    print(
        f"Training final models with "
        f"lgb_iterations={final_lgb_iterations}, "
        f"xgb_iterations={final_xgb_iterations}"
    )
    final_lightgbm, test_lightgbm = fit_final_lightgbm(
        X_full_cat,
        y_train,
        X_test_cat,
        categorical_columns,
        final_lgb_iterations,
        args.seed + 100,
        args.n_jobs,
    )
    final_xgboost, test_xgboost = fit_final_xgboost(
        X_full_cat,
        y_train,
        X_test_cat,
        final_xgb_iterations,
        args.seed + 200,
        args.n_jobs,
    )

    if best_transform == "rank":
        test_blend = rank_blend(test_lightgbm, test_xgboost, best_weight)
    else:
        test_blend = probability_blend(
            test_lightgbm,
            test_xgboost,
            best_weight,
        )

    final_lightgbm.booster_.save_model(str(model_output_dir / "lightgbm.txt"))
    final_xgboost.save_model(str(model_output_dir / "xgboost.json"))

    submission_path = (
        SUBMISSION_DIR / f"submission_oof_ensemble_{run_id}.csv"
    )
    write_submission(submission_path, test_ids, test_blend)

    pd.DataFrame(
        {
            "id": train_ids.to_numpy(),
            "y_true": y_train.to_numpy(),
            "lightgbm_prob": oof_lightgbm,
            "xgboost_prob": oof_xgboost,
            "blend_prob": oof_blend,
        }
    ).to_csv(experiment_dir / "oof_predictions.csv", index=False)

    pd.DataFrame(
        {
            "id": test_ids.to_numpy(),
            "lightgbm_prob": test_lightgbm,
            "xgboost_prob": test_xgboost,
            "blend_prob": test_blend,
        }
    ).to_csv(experiment_dir / "test_predictions.csv", index=False)

    experiment_metrics = pd.DataFrame(
        [
            {
                "source": "exp01_oof_ensemble",
                "model": name,
                "evaluation": "5_fold_oof",
                "auc": score,
                "logloss": (
                    None
                    if name == "rank_blend"
                    else float(
                        log_loss(
                            y_train,
                            np.clip(
                                candidate_predictions[name],
                                1e-12,
                                1 - 1e-12,
                            ),
                        )
                    )
                ),
            }
            for name, (score, _) in candidates.items()
        ]
    )
    baseline_rows = load_baseline_rows()
    comparison = pd.concat(
        [baseline_rows, experiment_metrics],
        ignore_index=True,
    ).sort_values("auc", ascending=False)
    comparison.to_csv(experiment_dir / "comparison.csv", index=False)

    metadata = {
        "run_id": run_id,
        "baseline_run": str(BASELINE_RUN),
        "folds": args.folds,
        "quick": args.quick,
        "categorical_columns": categorical_columns,
        "lightgbm_fold_iterations": lightgbm_iterations,
        "xgboost_fold_iterations": xgboost_iterations,
        "final_lightgbm_iterations": final_lgb_iterations,
        "final_xgboost_iterations": final_xgb_iterations,
        "best_method": best_method,
        "best_oof_auc": best_auc,
        "best_weight_lightgbm": best_weight,
        "best_weight_xgboost": 1.0 - best_weight,
        "blend_transform": best_transform,
        "submission_file": str(submission_path),
        "elapsed_seconds": time.perf_counter() - started_at,
    }
    (experiment_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nOOF results:")
    print(comparison.to_string(index=False))
    print(f"\nBest method: {best_method}")
    print(f"Best OOF AUC: {best_auc:.6f}")
    print(f"LightGBM weight: {best_weight:.2f}")
    print(f"Submission: {submission_path}")


if __name__ == "__main__":
    main()
