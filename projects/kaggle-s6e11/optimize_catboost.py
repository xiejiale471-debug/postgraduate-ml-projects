from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from config import OUTPUT_DIR, SUBMISSION_DIR
from data_utils import load_data, time_ordered_split, write_submission


BASELINE_RUN = OUTPUT_DIR / "run_20261104_221722"
EXPERIMENTS_DIR = OUTPUT_DIR / "experiments"


def default_trials() -> list[dict[str, object]]:
    return [
        {
            "name": "symmetric_depth6_lr03",
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 4.0,
            "random_strength": 1.0,
            "grow_policy": "SymmetricTree",
            "one_hot_max_size": 16,
            "border_count": 128,
        },
        {
            "name": "symmetric_depth7_lr03",
            "depth": 7,
            "learning_rate": 0.03,
            "l2_leaf_reg": 8.0,
            "random_strength": 2.0,
            "grow_policy": "SymmetricTree",
            "one_hot_max_size": 32,
            "border_count": 128,
        },
        {
            "name": "symmetric_depth8_lr02",
            "depth": 8,
            "learning_rate": 0.02,
            "l2_leaf_reg": 12.0,
            "random_strength": 1.0,
            "grow_policy": "SymmetricTree",
            "one_hot_max_size": 16,
            "border_count": 254,
        },
        {
            "name": "depthwise_depth6_lr03",
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 4.0,
            "random_strength": 2.0,
            "grow_policy": "Depthwise",
            "one_hot_max_size": 16,
            "border_count": 128,
        },
        {
            "name": "lossguide_64_lr03",
            "depth": 8,
            "max_leaves": 64,
            "learning_rate": 0.03,
            "l2_leaf_reg": 8.0,
            "random_strength": 1.0,
            "grow_policy": "Lossguide",
            "one_hot_max_size": 32,
            "border_count": 128,
        },
        {
            "name": "symmetric_depth5_lr04",
            "depth": 5,
            "learning_rate": 0.04,
            "l2_leaf_reg": 2.0,
            "random_strength": 0.5,
            "grow_policy": "SymmetricTree",
            "one_hot_max_size": 32,
            "border_count": 128,
        },
        {
            "name": "bernoulli_depth7_lr02",
            "depth": 7,
            "learning_rate": 0.02,
            "l2_leaf_reg": 16.0,
            "random_strength": 2.0,
            "bootstrap_type": "Bernoulli",
            "subsample": 0.8,
            "rsm": 0.85,
            "grow_policy": "SymmetricTree",
            "one_hot_max_size": 16,
            "border_count": 128,
        },
        {
            "name": "symmetric_depth6_lr025",
            "depth": 6,
            "learning_rate": 0.025,
            "l2_leaf_reg": 6.0,
            "random_strength": 3.0,
            "grow_policy": "SymmetricTree",
            "one_hot_max_size": 8,
            "border_count": 254,
        },
        {
            "name": "bernoulli_depth8_lr02",
            "depth": 8,
            "learning_rate": 0.02,
            "l2_leaf_reg": 16.0,
            "random_strength": 3.0,
            "bootstrap_type": "Bernoulli",
            "subsample": 0.8,
            "rsm": 0.8,
            "grow_policy": "SymmetricTree",
            "one_hot_max_size": 16,
            "border_count": 128,
        },
        {
            "name": "symmetric_depth6_lr03_border254",
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 10.0,
            "random_strength": 2.0,
            "grow_policy": "SymmetricTree",
            "one_hot_max_size": 16,
            "border_count": 254,
        },
    ]


def sample_time_split(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    sample_size: int,
    valid_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    fit_fraction = min(1.0, sample_size / len(X_fit))
    valid_fraction = min(1.0, valid_size / len(X_valid))
    X_fit_sample, _, y_fit_sample, _ = train_test_split(
        X_fit,
        y_fit,
        train_size=fit_fraction,
        stratify=y_fit,
        random_state=seed,
    )
    X_valid_sample, _, y_valid_sample, _ = train_test_split(
        X_valid,
        y_valid,
        train_size=valid_fraction,
        stratify=y_valid,
        random_state=seed + 1,
    )
    return X_fit_sample, y_fit_sample, X_valid_sample, y_valid_sample


def make_model(
    params: dict[str, object],
    iterations: int,
    seed: int,
    n_jobs: int,
    early_stopping_rounds: int | None,
) -> CatBoostClassifier:
    kwargs = {
        key: value
        for key, value in params.items()
        if key != "name"
    }
    return CatBoostClassifier(
        iterations=iterations,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        thread_count=n_jobs,
        early_stopping_rounds=early_stopping_rounds,
        verbose=False,
        allow_writing_files=False,
        **kwargs,
    )


def evaluate_model(
    model: CatBoostClassifier,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> dict[str, float]:
    probability = model.predict_proba(X_valid)[:, 1]
    return {
        "auc": float(roc_auc_score(y_valid, probability)),
        "logloss": float(log_loss(y_valid, probability)),
        "iterations": int(model.get_best_iteration()) + 1,
    }


def baseline_metrics() -> dict[str, float]:
    path = BASELINE_RUN / "metrics.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    catboost = frame.loc[frame["model"] == "catboost"]
    if catboost.empty:
        return {}
    row = catboost.iloc[0]
    return {
        "auc": float(row["validation_auc"]),
        "logloss": float(row["validation_logloss"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search CatBoost parameters and train a seed-averaged submission."
    )
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--search-sample-size", type=int, default=120000)
    parser.add_argument("--search-valid-size", type=int, default=30000)
    parser.add_argument("--search-iterations", type=int, default=800)
    parser.add_argument("--search-early-stopping", type=int, default=80)
    parser.add_argument("--final-iterations", type=int, default=3000)
    parser.add_argument("--final-early-stopping", type=int, default=200)
    parser.add_argument("--final-seeds", type=int, default=3)
    parser.add_argument("--improvement-threshold", type=float, default=0.0003)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.trials = min(args.trials, 4)
        args.search_sample_size = min(args.search_sample_size, 60000)
        args.search_valid_size = min(args.search_valid_size, 15000)
        args.search_iterations = min(args.search_iterations, 300)
        args.final_iterations = min(args.final_iterations, 500)
        args.final_early_stopping = min(args.final_early_stopping, 50)
        args.final_seeds = 1

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = EXPERIMENTS_DIR / f"exp02_catboost_tuning_{run_id}"
    experiment_dir.mkdir(parents=True, exist_ok=False)

    print("Loading data with the original v1 feature set...")
    X_train, y_train, train_ids, test_ids, X_test, categorical_columns = load_data(
        feature_set="v1"
    )
    X_fit, y_fit, X_valid, y_valid, _ = time_ordered_split(
        X_train,
        y_train,
        0.2,
    )

    X_search, y_search, X_search_valid, y_search_valid = sample_time_split(
        X_fit,
        y_fit,
        X_valid,
        y_valid,
        args.search_sample_size,
        args.search_valid_size,
        args.seed,
    )
    print(
        f"Search sample: fit={len(X_search):,}, valid={len(X_search_valid):,}"
    )

    trials = default_trials()[: args.trials]
    trial_rows = []
    best_trial: dict[str, object] | None = None
    best_search_auc = -1.0
    for index, params in enumerate(trials, start=1):
        started_at = time.perf_counter()
        model = make_model(
            params,
            args.search_iterations,
            args.seed + index,
            args.n_jobs,
            args.search_early_stopping,
        )
        model.fit(
            X_search,
            y_search,
            cat_features=categorical_columns,
            eval_set=(X_search_valid, y_search_valid),
            use_best_model=True,
        )
        metrics = evaluate_model(model, X_search_valid, y_search_valid)
        row = {
            "trial": index,
            "name": params["name"],
            "auc": metrics["auc"],
            "logloss": metrics["logloss"],
            "best_iteration": metrics["iterations"],
            "elapsed_seconds": time.perf_counter() - started_at,
            **{key: value for key, value in params.items() if key != "name"},
        }
        trial_rows.append(row)
        if metrics["auc"] > best_search_auc:
            best_search_auc = metrics["auc"]
            best_trial = dict(params)
        print(
            f"trial={index}/{len(trials)} name={params['name']} "
            f"auc={metrics['auc']:.6f} time={row['elapsed_seconds']:.1f}s"
        )

    trial_frame = pd.DataFrame(trial_rows).sort_values("auc", ascending=False)
    trial_frame.to_csv(experiment_dir / "search_results.csv", index=False)
    if best_trial is None:
        raise RuntimeError("No CatBoost trial completed.")

    print(f"\nValidating best trial on the full 80/20 split: {best_trial['name']}")
    validation_started = time.perf_counter()
    validation_model = make_model(
        best_trial,
        args.final_iterations,
        args.seed + 1000,
        args.n_jobs,
        args.final_early_stopping,
    )
    validation_model.fit(
        X_fit,
        y_fit,
        cat_features=categorical_columns,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
    )
    validation_metrics = evaluate_model(validation_model, X_valid, y_valid)
    baseline = baseline_metrics()
    validation_auc = validation_metrics["auc"]
    baseline_auc = baseline.get("auc")
    improvement = (
        validation_auc - baseline_auc
        if baseline_auc is not None
        else None
    )
    is_improved = (
        improvement is not None
        and improvement >= args.improvement_threshold
    )

    results = {
        "run_id": run_id,
        "best_trial": best_trial,
        "best_search_auc": best_search_auc,
        "validation_auc": validation_auc,
        "validation_logloss": validation_metrics["logloss"],
        "validation_best_iteration": validation_metrics["iterations"],
        "baseline_auc": baseline_auc,
        "improvement_over_baseline": improvement,
        "improvement_threshold": args.improvement_threshold,
        "accepted_for_submission": is_improved,
        "validation_elapsed_seconds": time.perf_counter() - validation_started,
        "feature_set": "v1",
        "categorical_columns": categorical_columns,
    }

    if is_improved:
        seed_probabilities = []
        seed_paths = []
        for seed_index in range(args.final_seeds):
            seed = args.seed + 2000 + seed_index
            print(f"Training final full model with seed={seed}")
            final_model = CatBoostClassifier(
                iterations=max(1, validation_metrics["iterations"]),
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=seed,
                thread_count=args.n_jobs,
                verbose=False,
                allow_writing_files=False,
                **{
                    key: value
                    for key, value in best_trial.items()
                    if key != "name"
                },
            )
            final_model.fit(
                X_train,
                y_train,
                cat_features=categorical_columns,
            )
            model_path = experiment_dir / f"catboost_seed_{seed}.cbm"
            final_model.save_model(str(model_path))
            seed_paths.append(str(model_path))
            seed_probabilities.append(
                final_model.predict_proba(X_test)[:, 1]
            )
        averaged_probability = np.mean(seed_probabilities, axis=0)
        submission_path = (
            SUBMISSION_DIR / f"submission_catboost_tuned_seedavg_{run_id}.csv"
        )
        write_submission(submission_path, test_ids, averaged_probability)
        results["seed_model_paths"] = seed_paths
        results["submission_file"] = str(submission_path)
    else:
        print(
            "Candidate did not exceed the baseline by the required threshold. "
            "No new submission was created."
        )

    (experiment_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
