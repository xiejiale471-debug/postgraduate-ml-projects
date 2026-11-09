from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score

from config import OUTPUT_DIR, SUBMISSION_DIR
from data_utils import load_data, time_ordered_split, write_submission


BASELINE_RUN = OUTPUT_DIR / "run_20261104_221722"
EXPERIMENTS_DIR = OUTPUT_DIR / "experiments"
SEEDS = (42, 2024, 3407)


def variants() -> list[dict[str, object]]:
    return [
        {
            "name": "symmetric_depth6",
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 4.0,
            "random_strength": 1.0,
            "grow_policy": "SymmetricTree",
            "one_hot_max_size": 16,
            "border_count": 128,
        },
        {
            "name": "symmetric_depth7",
            "depth": 7,
            "learning_rate": 0.03,
            "l2_leaf_reg": 8.0,
            "random_strength": 2.0,
            "grow_policy": "SymmetricTree",
            "one_hot_max_size": 32,
            "border_count": 128,
        },
        {
            "name": "depthwise_depth6",
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 4.0,
            "random_strength": 2.0,
            "grow_policy": "Depthwise",
            "one_hot_max_size": 16,
            "border_count": 128,
        },
        {
            "name": "lossguide_64",
            "depth": 8,
            "max_leaves": 64,
            "learning_rate": 0.03,
            "l2_leaf_reg": 8.0,
            "random_strength": 1.0,
            "grow_policy": "Lossguide",
            "one_hot_max_size": 32,
            "border_count": 128,
        },
    ]


def baseline_auc() -> float:
    path = BASELINE_RUN / "metrics.csv"
    frame = pd.read_csv(path)
    row = frame.loc[frame["model"] == "catboost"].iloc[0]
    return float(row["validation_auc"])


def train_validation_model(
    params: dict[str, object],
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    categorical_columns: list[str],
    seed: int,
    iterations: int,
    early_stopping: int,
    n_jobs: int,
) -> tuple[CatBoostClassifier, np.ndarray, dict[str, float]]:
    model = CatBoostClassifier(
        iterations=iterations,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        thread_count=n_jobs,
        early_stopping_rounds=early_stopping,
        verbose=False,
        allow_writing_files=False,
        **{key: value for key, value in params.items() if key != "name"},
    )
    model.fit(
        X_fit,
        y_fit,
        cat_features=categorical_columns,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
    )
    probability = model.predict_proba(X_valid)[:, 1]
    metrics = {
        "auc": float(roc_auc_score(y_valid, probability)),
        "logloss": float(log_loss(y_valid, probability)),
        "best_iteration": int(model.get_best_iteration()) + 1,
    }
    return model, probability, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate CatBoost variants and seed averaging."
    )
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--early-stopping", type=int, default=200)
    parser.add_argument("--improvement-threshold", type=float, default=0.0002)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.iterations = min(args.iterations, 500)
        args.early_stopping = min(args.early_stopping, 50)

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = EXPERIMENTS_DIR / f"exp04_catboost_variants_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=False)

    print("Loading original v1 features...")
    X_train, y_train, _, test_ids, X_test, categorical_columns = load_data("v1")
    X_fit, y_fit, X_valid, y_valid, _ = time_ordered_split(
        X_train,
        y_train,
        0.2,
    )
    baseline = baseline_auc()

    variant_rows = []
    variant_models: dict[str, object] = {}
    variant_predictions: dict[str, np.ndarray] = {}
    variant_iterations: dict[str, int] = {}
    for index, params in enumerate(variants(), start=1):
        started_at = time.perf_counter()
        print(f"variant={index}/4 name={params['name']}")
        model, prediction, metrics = train_validation_model(
            params,
            X_fit,
            y_fit,
            X_valid,
            y_valid,
            categorical_columns,
            SEEDS[0],
            args.iterations,
            args.early_stopping,
            args.n_jobs,
        )
        elapsed = time.perf_counter() - started_at
        variant_rows.append(
            {
                "name": params["name"],
                "auc": metrics["auc"],
                "logloss": metrics["logloss"],
                "best_iteration": metrics["best_iteration"],
                "elapsed_seconds": elapsed,
                "grow_policy": params["grow_policy"],
                "depth": params["depth"],
            }
        )
        variant_models[params["name"]] = model
        variant_predictions[params["name"]] = prediction
        variant_iterations[params["name"]] = metrics["best_iteration"]
        print(
            f"{params['name']}: AUC={metrics['auc']:.6f}, "
            f"time={elapsed:.1f}s"
        )

    variant_table = pd.DataFrame(variant_rows).sort_values("auc", ascending=False)
    variant_table.to_csv(output_dir / "variant_results.csv", index=False)
    best_name = str(variant_table.iloc[0]["name"])
    best_params = next(item for item in variants() if item["name"] == best_name)
    print(f"Best variant: {best_name}")

    seed_predictions = [variant_predictions[best_name]]
    seed_metrics = [
        {
            "seed": SEEDS[0],
            "auc": float(variant_table.iloc[0]["auc"]),
            "logloss": float(variant_table.iloc[0]["logloss"]),
            "best_iteration": variant_iterations[best_name],
        }
    ]
    for seed in SEEDS[1:]:
        print(f"Training best variant with seed={seed}")
        _, prediction, metrics = train_validation_model(
            best_params,
            X_fit,
            y_fit,
            X_valid,
            y_valid,
            categorical_columns,
            seed,
            args.iterations,
            args.early_stopping,
            args.n_jobs,
        )
        seed_predictions.append(prediction)
        seed_metrics.append({"seed": seed, **metrics})
        print(
            f"seed={seed}: AUC={metrics['auc']:.6f}, "
            f"iteration={metrics['best_iteration']}"
        )

    seed_average = np.mean(seed_predictions, axis=0)
    seed_average_auc = float(roc_auc_score(y_valid, seed_average))
    seed_average_logloss = float(log_loss(y_valid, seed_average))
    improvement = seed_average_auc - baseline
    accepted = improvement >= args.improvement_threshold

    pd.DataFrame(seed_metrics).to_csv(
        output_dir / "seed_results.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "id": np.arange(len(y_valid)),
            "y_true": y_valid.to_numpy(),
            "seed_average_prob": seed_average,
        }
    ).to_csv(output_dir / "seed_average_validation.csv", index=False)

    results = {
        "run_id": run_id,
        "baseline_auc": baseline,
        "best_variant": best_name,
        "best_variant_params": best_params,
        "variant_results": variant_rows,
        "seed_results": seed_metrics,
        "seed_average_auc": seed_average_auc,
        "seed_average_logloss": seed_average_logloss,
        "improvement_over_baseline": improvement,
        "improvement_threshold": args.improvement_threshold,
        "accepted_for_submission": accepted,
        "feature_set": "v1",
    }

    if accepted:
        full_probabilities = []
        model_paths = []
        for seed, metrics in zip(SEEDS, seed_metrics):
            print(f"Training full model for seed={seed}")
            model = CatBoostClassifier(
                iterations=max(1, int(metrics["best_iteration"])),
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=seed,
                thread_count=args.n_jobs,
                verbose=False,
                allow_writing_files=False,
                **{
                    key: value
                    for key, value in best_params.items()
                    if key != "name"
                },
            )
            model.fit(X_train, y_train, cat_features=categorical_columns)
            model_path = output_dir / f"catboost_{best_name}_seed_{seed}.cbm"
            model.save_model(str(model_path))
            model_paths.append(str(model_path))
            full_probabilities.append(model.predict_proba(X_test)[:, 1])

        test_average = np.mean(full_probabilities, axis=0)
        submission_path = (
            SUBMISSION_DIR
            / f"submission_catboost_variant_seedavg_{run_id}.csv"
        )
        write_submission(submission_path, test_ids, test_average)
        results["model_paths"] = model_paths
        results["submission_file"] = str(submission_path)
    else:
        print(
            "Seed average did not exceed the baseline threshold. "
            "No submission was created."
        )

    (output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
