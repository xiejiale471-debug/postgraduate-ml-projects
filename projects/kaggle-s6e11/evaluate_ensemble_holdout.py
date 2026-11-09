from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from config import OUTPUT_DIR, SUBMISSION_DIR
from data_utils import load_data, time_ordered_split, write_submission
from train_oof_ensemble import (
    EXPERIMENTS_DIR,
    fit_lightgbm_fold,
    fit_xgboost_fold,
    probability_blend,
    rank_blend,
    search_blend_weight,
    set_categories,
)


BASELINE_RUN = OUTPUT_DIR / "run_20261104_221722"


def latest_oof_metadata() -> tuple[object, dict]:
    candidates = sorted(
        path
        for path in EXPERIMENTS_DIR.glob("exp01_oof_ensemble_*")
        if (path / "metadata.json").exists()
    )
    if not candidates:
        raise FileNotFoundError("No completed OOF ensemble experiment found.")
    path = candidates[-1]
    return path, json.loads((path / "metadata.json").read_text(encoding="utf-8"))


def metrics_row(
    source: str,
    model: str,
    evaluation: str,
    y_true: pd.Series,
    prediction: np.ndarray,
) -> dict[str, object]:
    row = {
        "source": source,
        "model": model,
        "evaluation": evaluation,
        "auc": float(roc_auc_score(y_true, prediction)),
        "logloss": float(
            log_loss(
                y_true,
                np.clip(prediction, 1e-12, 1 - 1e-12),
            )
        ),
    }
    if "rank" in model:
        row["logloss"] = np.nan
    return row


def main() -> None:
    oof_dir, oof_metadata = latest_oof_metadata()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = EXPERIMENTS_DIR / f"exp01_holdout_comparison_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=False)

    X_train, y_train, train_ids, _, _, categorical_columns = load_data()
    X_fit, y_fit, X_valid, y_valid, _ = time_ordered_split(
        X_train,
        y_train,
        0.2,
    )
    X_fit_cat, (X_valid_cat,) = set_categories(
        X_fit,
        [X_valid],
        categorical_columns,
    )

    print("Training LightGBM and XGBoost on the same 80/20 time split...")
    lightgbm_prob, _, lightgbm_iterations = fit_lightgbm_fold(
        X_fit_cat,
        y_fit,
        X_valid_cat,
        y_valid,
        categorical_columns,
        42,
        -1,
        False,
    )
    xgboost_prob, _, xgboost_iterations = fit_xgboost_fold(
        X_fit_cat,
        y_fit,
        X_valid_cat,
        y_valid,
        43,
        -1,
        False,
    )

    oof_weight = float(oof_metadata["best_weight_lightgbm"])
    fixed_probability = probability_blend(
        lightgbm_prob,
        xgboost_prob,
        oof_weight,
    )
    fixed_rank = rank_blend(
        lightgbm_prob,
        xgboost_prob,
        oof_weight,
    )
    optimized_probability_weight, _ = search_blend_weight(
        lightgbm_prob,
        xgboost_prob,
        y_valid,
        use_rank=False,
    )
    optimized_rank_weight, _ = search_blend_weight(
        lightgbm_prob,
        xgboost_prob,
        y_valid,
        use_rank=True,
    )

    candidate_predictions = {
        "lightgbm_holdout": lightgbm_prob,
        "xgboost_holdout": xgboost_prob,
        "oof_weight_probability_blend": fixed_probability,
        "oof_weight_rank_blend": fixed_rank,
        "holdout_optimized_probability_blend": probability_blend(
            lightgbm_prob,
            xgboost_prob,
            optimized_probability_weight,
        ),
        "holdout_optimized_rank_blend": rank_blend(
            lightgbm_prob,
            xgboost_prob,
            optimized_rank_weight,
        ),
    }
    experiment_rows = [
        metrics_row(
            "exp01_holdout_comparison",
            name,
            "time_holdout",
            y_valid,
            prediction,
        )
        for name, prediction in candidate_predictions.items()
    ]

    baseline_metrics_path = BASELINE_RUN / "metrics.csv"
    baseline_rows = []
    if baseline_metrics_path.exists():
        baseline = pd.read_csv(baseline_metrics_path)
        baseline_rows = [
            {
                "source": "first_run",
                "model": row["model"],
                "evaluation": "time_holdout",
                "auc": row["validation_auc"],
                "logloss": row["validation_logloss"],
            }
            for _, row in baseline.iterrows()
        ]

    comparison = pd.concat(
        [pd.DataFrame(baseline_rows), pd.DataFrame(experiment_rows)],
        ignore_index=True,
    ).sort_values("auc", ascending=False)
    comparison.to_csv(output_dir / "comparison.csv", index=False)

    pd.DataFrame(
        {
            "id": train_ids.iloc[len(X_fit) :].reset_index(drop=True),
            "y_true": y_valid,
            "lightgbm_prob": lightgbm_prob,
            "xgboost_prob": xgboost_prob,
            "oof_weight_probability_blend": fixed_probability,
            "oof_weight_rank_blend": fixed_rank,
        }
    ).to_csv(output_dir / "holdout_predictions.csv", index=False)

    metadata = {
        "run_id": run_id,
        "oof_source": str(oof_dir),
        "lightgbm_iterations": lightgbm_iterations,
        "xgboost_iterations": xgboost_iterations,
        "oof_lightgbm_weight": oof_weight,
        "oof_xgboost_weight": 1.0 - oof_weight,
        "holdout_optimized_probability_weight": optimized_probability_weight,
        "holdout_optimized_rank_weight": optimized_rank_weight,
        "best_candidate": comparison.iloc[0].to_dict(),
    }

    test_predictions_path = oof_dir / "test_predictions.csv"
    if test_predictions_path.exists():
        test_predictions = pd.read_csv(test_predictions_path)
        test_rank_blend = rank_blend(
            test_predictions["lightgbm_prob"].to_numpy(),
            test_predictions["xgboost_prob"].to_numpy(),
            optimized_rank_weight,
        )
        test_probability_blend = probability_blend(
            test_predictions["lightgbm_prob"].to_numpy(),
            test_predictions["xgboost_prob"].to_numpy(),
            optimized_probability_weight,
        )
        rank_submission_path = (
            SUBMISSION_DIR / f"submission_ensemble_holdout_rank_{run_id}.csv"
        )
        probability_submission_path = (
            SUBMISSION_DIR
            / f"submission_ensemble_holdout_probability_{run_id}.csv"
        )
        write_submission(
            rank_submission_path,
            test_predictions["id"],
            test_rank_blend,
        )
        write_submission(
            probability_submission_path,
            test_predictions["id"],
            test_probability_blend,
        )
        metadata["rank_submission"] = str(rank_submission_path)
        metadata["probability_submission"] = str(probability_submission_path)

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print("\nSame-split comparison:")
    print(comparison.to_string(index=False))
    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()
