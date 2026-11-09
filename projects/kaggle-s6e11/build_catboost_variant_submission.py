from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier

from config import SUBMISSION_DIR
from data_utils import load_data, write_submission


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build an exploratory CatBoost seed-averaged submission from "
            "an existing variant experiment."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Path containing results.json from train_catboost_variants.py.",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    experiment_dir = args.experiment_dir.resolve()
    results_path = experiment_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")

    results = json.loads(results_path.read_text(encoding="utf-8"))
    if results.get("feature_set") != "v1":
        raise ValueError("This builder expects a v1 feature experiment.")

    params = {
        key: value
        for key, value in results["best_variant_params"].items()
        if key != "name"
    }
    seed_results = results["seed_results"]

    print("Loading full training and test data with v1 features...")
    X_train, y_train, _, test_ids, X_test, categorical_columns = load_data("v1")
    seed_probabilities = []
    model_paths = []
    final_seed_rows = []

    final_model_dir = experiment_dir / "final_seed_models"
    final_model_dir.mkdir(exist_ok=True)
    for item in seed_results:
        seed = int(item["seed"])
        iterations = max(1, int(item["best_iteration"]))
        print(f"Training full model: seed={seed}, iterations={iterations}")
        model = CatBoostClassifier(
            iterations=iterations,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=seed,
            thread_count=args.n_jobs,
            verbose=False,
            allow_writing_files=False,
            **params,
        )
        model.fit(X_train, y_train, cat_features=categorical_columns)
        probability = model.predict_proba(X_test)[:, 1]
        seed_probabilities.append(probability)
        model_path = final_model_dir / f"catboost_seed_{seed}.cbm"
        model.save_model(str(model_path))
        model_paths.append(str(model_path))
        final_seed_rows.append(
            {
                "seed": seed,
                "iterations": iterations,
                "model_path": str(model_path),
            }
        )

    averaged_probability = np.mean(seed_probabilities, axis=0)
    submission_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = (
        SUBMISSION_DIR
        / f"submission_catboost_variant_seedavg_exploratory_{submission_run_id}.csv"
    )
    write_submission(submission_path, test_ids, averaged_probability)

    metadata = {
        "source_experiment": str(experiment_dir),
        "submission_run_id": submission_run_id,
        "submission_file": str(submission_path),
        "validation_auc": results.get("seed_average_auc"),
        "baseline_auc": results.get("baseline_auc"),
        "validation_delta": results.get("improvement_over_baseline"),
        "advisory": (
            "Exploratory candidate only: validation AUC did not exceed the "
            "first CatBoost baseline."
        ),
        "models": final_seed_rows,
        "categorical_columns": categorical_columns,
    }
    (experiment_dir / "exploratory_submission.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Submission created: {submission_path}")
    print(
        "Validation advisory: "
        f"seed-average AUC={results.get('seed_average_auc'):.6f}, "
        f"baseline={results.get('baseline_auc'):.6f}"
    )


if __name__ == "__main__":
    main()
