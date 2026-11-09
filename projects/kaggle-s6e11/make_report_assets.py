from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import (
    average_precision_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

from config import (
    DATA_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    TARGET_COL,
    TEST_PATH,
    TRAIN_PATH,
)
from data_utils import add_common_features


MODEL_LABELS = {
    "catboost": "CatBoost",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
}

MODEL_COLORS = {
    "catboost": "#1f77b4",
    "lightgbm": "#2ca02c",
    "xgboost": "#d62728",
    "random_forest": "#9467bd",
}

MODEL_CONFIG = [
    {
        "模型": "CatBoost",
        "核心参数": (
            "iterations<=3000, learning_rate=0.03, depth=6, "
            "l2_leaf_reg=4, early_stopping=200"
        ),
        "类别变量处理": "原生类别特征",
    },
    {
        "模型": "LightGBM",
        "核心参数": (
            "n_estimators<=3000, learning_rate=0.03, num_leaves=63, "
            "min_child_samples=80, subsample=0.85"
        ),
        "类别变量处理": "pandas category + 原生类别支持",
    },
    {
        "模型": "XGBoost",
        "核心参数": (
            "n_estimators<=3000, learning_rate=0.03, max_depth=6, "
            "min_child_weight=3, subsample=0.85"
        ),
        "类别变量处理": "pandas category + enable_categorical",
    },
    {
        "模型": "Random Forest",
        "核心参数": (
            "n_estimators=350, max_features=sqrt, "
            "min_samples_leaf=5, class_weight=balanced_subsample"
        ),
        "类别变量处理": "训练折内拟合 One-Hot 编码",
    },
]


def configure_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "SimSun",
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = [
        [str(value).replace("|", "\\|") for value in row]
        for row in frame.itertuples(index=False, name=None)
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def find_latest_run() -> str:
    runs = sorted(
        [
            path.name.removeprefix("run_")
            for path in OUTPUT_DIR.glob("run_*")
            if (path / "metrics.csv").exists()
        ]
    )
    if not runs:
        raise FileNotFoundError("No completed run with metrics.csv was found.")
    return runs[-1]


def save_figure(fig, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_workflow(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    boxes = [
        (0.3, 2.0, 1.45, 1.0, "训练集与测试集"),
        (2.05, 2.0, 1.45, 1.0, "共同特征工程"),
        (3.8, 2.0, 1.45, 1.0, "按 ID 顺序切分"),
        (5.55, 2.0, 1.45, 1.0, "四个模型训练"),
        (7.3, 2.0, 1.45, 1.0, "验证集评估"),
        (8.95, 3.25, 0.9, 0.75, "全量重训"),
        (8.95, 0.85, 0.9, 0.75, "预测并提交"),
    ]
    for x, y, width, height, label in boxes:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.2,
            edgecolor="#4c4c4c",
            facecolor="#f3f6fa",
        )
        ax.add_patch(patch)
        ax.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=11,
        )

    for x1, x2 in [(1.75, 2.05), (3.5, 3.8), (5.25, 5.55), (7.0, 7.3)]:
        ax.add_patch(
            FancyArrowPatch(
                (x1, 2.5),
                (x2, 2.5),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.2,
                color="#4c4c4c",
            )
        )
    ax.add_patch(
        FancyArrowPatch(
            (8.02, 2.65),
            (8.95, 3.55),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.2,
            color="#4c4c4c",
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (8.02, 2.35),
            (8.95, 1.2),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.2,
            color="#4c4c4c",
        )
    )
    ax.text(
        5.0,
        4.35,
        "Kaggle Playground S6E11 实验流程",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )
    save_figure(fig, output_path)


def plot_class_distribution(y: pd.Series, output_path: Path) -> None:
    counts = y.value_counts().reindex([0, 1]).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bars = ax.bar(
        ["正常交易", "诈骗交易"],
        counts.to_numpy(),
        color=["#5b8db8", "#d76a6a"],
        width=0.58,
    )
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}\n({count / len(y):.2%})",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylabel("样本数")
    ax.set_title("训练集类别分布")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, counts.max() * 1.2)
    save_figure(fig, output_path)


def plot_metrics_comparison(metrics: pd.DataFrame, output_path: Path) -> None:
    metrics = metrics.sort_values("validation_auc", ascending=False)
    labels = [MODEL_LABELS.get(name, name) for name in metrics["model"]]
    colors = [MODEL_COLORS.get(name, "#777777") for name in metrics["model"]]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    for ax, column, title in [
        (axes[0], "validation_auc", "验证集 ROC-AUC"),
        (axes[1], "validation_logloss", "验证集 LogLoss"),
    ]:
        bars = ax.bar(labels, metrics[column], color=colors, width=0.62)
        for bar, value in zip(bars, metrics[column]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", rotation=15)
        if column == "validation_auc":
            lower = min(metrics[column]) - 0.01
            ax.set_ylim(max(0.0, lower), 1.0)
    save_figure(fig, output_path)


def catboost_feature_importance(
    run_id: str,
    output_path: Path,
) -> pd.DataFrame | None:
    metadata_path = MODEL_DIR / f"{run_id}_catboost" / "metadata.json"
    model_path = MODEL_DIR / f"{run_id}_catboost" / "model.cbm"
    if not metadata_path.exists() or not model_path.exists():
        return None

    import json

    from catboost import CatBoostClassifier

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_names = metadata.get("feature_columns", [])
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    importances = np.asarray(model.get_feature_importance(), dtype=float)
    if not feature_names or len(feature_names) != len(importances):
        feature_names = list(model.feature_names_)

    importance_table = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
            }
        )
        .sort_values("importance", ascending=False)
        .head(15)
    )

    plot_data = importance_table.sort_values("importance")
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.barh(
        plot_data["feature"],
        plot_data["importance"],
        color="#4f7da8",
    )
    ax.set_xlabel("特征重要性")
    ax.set_title("CatBoost 前 15 个重要特征")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure(fig, output_path)
    return importance_table


def load_validation_predictions(run_id: str, model_name: str) -> pd.DataFrame | None:
    path = MODEL_DIR / f"{run_id}_{model_name}" / "valid_predictions.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    required = {"id", "y_true", "y_prob"}
    if not required.issubset(frame.columns):
        return None
    return frame


def plot_roc_curves(
    predictions: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.5))
    for model_name, frame in predictions.items():
        fpr, tpr, _ = roc_curve(frame["y_true"], frame["y_prob"])
        score = auc(fpr, tpr)
        ax.plot(
            fpr,
            tpr,
            linewidth=2.0,
            color=MODEL_COLORS.get(model_name),
            label=f"{MODEL_LABELS.get(model_name, model_name)} (AUC={score:.4f})",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1)
    ax.set_xlabel("假阳性率 FPR")
    ax.set_ylabel("真阳性率 TPR")
    ax.set_title("四模型验证集 ROC 曲线")
    ax.legend(loc="lower right")
    ax.grid(linestyle="--", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure(fig, output_path)


def plot_pr_curves(
    predictions: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.5))
    for model_name, frame in predictions.items():
        precision, recall, _ = precision_recall_curve(
            frame["y_true"],
            frame["y_prob"],
        )
        average_precision = average_precision_score(
            frame["y_true"],
            frame["y_prob"],
        )
        ax.plot(
            recall,
            precision,
            linewidth=2.0,
            color=MODEL_COLORS.get(model_name),
            label=(
                f"{MODEL_LABELS.get(model_name, model_name)} "
                f"(AP={average_precision:.4f})"
            ),
        )
    ax.set_xlabel("召回率 Recall")
    ax.set_ylabel("精确率 Precision")
    ax.set_title("四模型验证集 Precision-Recall 曲线")
    ax.legend(loc="upper right")
    ax.grid(linestyle="--", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure(fig, output_path)


def threshold_analysis(
    frame: pd.DataFrame,
    output_table: Path,
) -> tuple[float, dict[str, float]]:
    precision, recall, thresholds = precision_recall_curve(
        frame["y_true"],
        frame["y_prob"],
    )
    f1_values = 2 * precision * recall / np.clip(
        precision + recall,
        a_min=1e-12,
        a_max=None,
    )
    best_index = int(np.argmax(f1_values[:-1]))
    best_threshold = float(thresholds[best_index])

    candidate_thresholds = sorted(
        {
            0.01,
            0.02,
            0.05,
            0.10,
            0.20,
            0.30,
            0.50,
            best_threshold,
        }
    )
    rows = []
    for threshold in candidate_thresholds:
        y_pred = (frame["y_prob"] >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(
                    frame["y_true"], y_pred, zero_division=0
                ),
                "recall": recall_score(
                    frame["y_true"], y_pred, zero_division=0
                ),
                "f1": f1_score(frame["y_true"], y_pred, zero_division=0),
            }
        )
    table = pd.DataFrame(rows).sort_values("threshold")
    table.to_csv(output_table, index=False, encoding="utf-8-sig")

    best_y_pred = (frame["y_prob"] >= best_threshold).astype(int)
    scores = {
        "threshold": best_threshold,
        "precision": float(precision_score(frame["y_true"], best_y_pred)),
        "recall": float(recall_score(frame["y_true"], best_y_pred)),
        "f1": float(f1_score(frame["y_true"], best_y_pred)),
    }
    return best_threshold, scores


def plot_confusion_matrix(
    frame: pd.DataFrame,
    threshold: float,
    output_path: Path,
) -> None:
    y_pred = (frame["y_prob"] >= threshold).astype(int)
    matrix = confusion_matrix(frame["y_true"], y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                fontsize=12,
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
            )
    ax.set_xticks([0, 1], ["预测正常", "预测诈骗"])
    ax.set_yticks([0, 1], ["真实正常", "真实诈骗"])
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(f"最优阈值 {threshold:.4f} 下的混淆矩阵")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, output_path)


def plot_probability_distribution(
    frame: pd.DataFrame,
    model_name: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bins = np.linspace(0, 1, 60)
    ax.hist(
        frame.loc[frame["y_true"] == 0, "y_prob"],
        bins=bins,
        density=True,
        alpha=0.55,
        label="真实正常交易",
        color="#5b8db8",
    )
    ax.hist(
        frame.loc[frame["y_true"] == 1, "y_prob"],
        bins=bins,
        density=True,
        alpha=0.65,
        label="真实诈骗交易",
        color="#d76a6a",
    )
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_xlabel("预测诈骗概率")
    ax.set_ylabel("密度（对数刻度）")
    ax.set_title(f"{MODEL_LABELS.get(model_name, model_name)} 验证集概率分布")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure(fig, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate report tables and figures from a completed run."
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run ID such as 20261104_221722. Defaults to the latest run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR / "report_assets",
    )
    args = parser.parse_args()

    configure_chinese_font()
    run_id = args.run_id or find_latest_run()
    run_dir = OUTPUT_DIR / f"run_{run_id}"
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    engineered_train = add_common_features(train.drop(columns=[TARGET_COL, "id"]))
    engineered_test = add_common_features(test.drop(columns=["id"]))
    metrics = pd.read_csv(metrics_path)
    metrics = metrics.sort_values("validation_auc", ascending=False).reset_index(
        drop=True
    )
    metrics.insert(0, "rank", np.arange(1, len(metrics) + 1))

    dataset_summary = pd.DataFrame(
        [
            ("训练集样本数", f"{len(train):,}"),
            ("测试集样本数", f"{len(test):,}"),
            ("原始预测特征数", f"{engineered_train.shape[1] - 4:,}"),
            ("特征工程后特征数", f"{engineered_train.shape[1]:,}"),
            ("类别特征数", "5"),
            ("正常交易样本数", f"{int((train[TARGET_COL] == 0).sum()):,}"),
            ("诈骗交易样本数", f"{int((train[TARGET_COL] == 1).sum()):,}"),
            ("训练集正类比例", f"{train[TARGET_COL].mean():.4%}"),
            (
                "训练集/测试集缺失值",
                f"{int(train.isna().sum().sum())}/{int(test.isna().sum().sum())}",
            ),
        ],
        columns=["指标", "数值"],
    )

    config_table = pd.DataFrame(MODEL_CONFIG)
    result_table = metrics[
        [
            "rank",
            "model",
            "validation_auc",
            "validation_logloss",
            "best_iteration",
            "elapsed_seconds",
        ]
    ].copy()
    result_table["model"] = result_table["model"].map(
        lambda name: MODEL_LABELS.get(name, name)
    )
    result_table.columns = [
        "排名",
        "模型",
        "验证AUC",
        "验证LogLoss",
        "最佳迭代",
        "训练秒数",
    ]
    result_table["验证AUC"] = result_table["验证AUC"].round(6)
    result_table["验证LogLoss"] = result_table["验证LogLoss"].round(6)
    result_table["训练秒数"] = result_table["训练秒数"].round(1)

    dataset_summary.to_csv(
        output_dir / "table1_dataset_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    config_table.to_csv(
        output_dir / "table2_model_config.csv",
        index=False,
        encoding="utf-8-sig",
    )
    result_table.to_csv(
        output_dir / "table3_validation_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_workflow(output_dir / "fig0_experiment_workflow.png")
    plot_class_distribution(train[TARGET_COL], output_dir / "fig1_class_distribution.png")
    plot_metrics_comparison(metrics, output_dir / "fig2_metrics_comparison.png")

    feature_importance = catboost_feature_importance(
        run_id,
        output_dir / "fig3_catboost_feature_importance.png",
    )
    if feature_importance is not None:
        feature_importance.to_csv(
            output_dir / "table4_catboost_feature_importance.csv",
            index=False,
            encoding="utf-8-sig",
        )

    predictions = {}
    for model_name in metrics["model"]:
        frame = load_validation_predictions(run_id, model_name)
        if frame is not None:
            predictions[model_name] = frame

    generated_validation_assets = False
    threshold_scores = None
    best_model_name = str(metrics.iloc[0]["model"])
    if len(predictions) == len(metrics):
        plot_roc_curves(predictions, output_dir / "fig4_roc_curves.png")
        plot_pr_curves(predictions, output_dir / "fig5_precision_recall_curves.png")
        generated_validation_assets = True

        if best_model_name in predictions:
            threshold, threshold_scores = threshold_analysis(
                predictions[best_model_name],
                output_dir / "table5_threshold_analysis.csv",
            )
            plot_confusion_matrix(
                predictions[best_model_name],
                threshold,
                output_dir / "fig6_confusion_matrix.png",
            )
            plot_probability_distribution(
                predictions[best_model_name],
                best_model_name,
                output_dir / "fig7_probability_distribution.png",
            )

    summary_lines = [
        "# Kaggle Playground S6E11 报告图表说明",
        "",
        f"- 运行编号：`{run_id}`",
        f"- 数据目录：`{DATA_DIR}`",
        f"- 输出目录：`{output_dir}`",
        "",
        "## 表1 数据集概况",
        "",
        dataframe_to_markdown(dataset_summary),
        "",
        "## 表2 模型配置",
        "",
        dataframe_to_markdown(config_table),
        "",
        "## 表3 验证结果",
        "",
        dataframe_to_markdown(result_table),
        "",
        "## 已生成图片",
        "",
        "- `fig0_experiment_workflow.png`：实验流程图",
        "- `fig1_class_distribution.png`：训练集类别分布",
        "- `fig2_metrics_comparison.png`：AUC 与 LogLoss 对比",
        "- `fig3_catboost_feature_importance.png`：CatBoost 特征重要性",
    ]
    if generated_validation_assets:
        summary_lines.extend(
            [
                "- `fig4_roc_curves.png`：四模型 ROC 曲线",
                "- `fig5_precision_recall_curves.png`：四模型 PR 曲线",
                "- `fig6_confusion_matrix.png`：最优模型混淆矩阵",
                "- `fig7_probability_distribution.png`：最优模型概率分布",
            ]
        )
        if threshold_scores is not None:
            summary_lines.extend(
                [
                    "",
                    "## 阈值分析",
                    "",
                    (
                        f"按验证集 F1 最大化选择阈值 "
                        f"`{threshold_scores['threshold']:.6f}`，"
                        f"Precision=`{threshold_scores['precision']:.6f}`，"
                        f"Recall=`{threshold_scores['recall']:.6f}`，"
                        f"F1=`{threshold_scores['f1']:.6f}`。"
                    ),
                ]
            )
    else:
        missing = [
            model_name
            for model_name in metrics["model"]
            if model_name not in predictions
        ]
        summary_lines.extend(
            [
                "",
                "## 缺少验证集预测",
                "",
                (
                    "以下模型缺少 `valid_predictions.csv`，因此未生成 ROC、"
                    "PR、混淆矩阵和概率分布图："
                    + "、".join(MODEL_LABELS.get(name, name) for name in missing)
                ),
                "",
                "使用以下命令重新生成验证集预测，不进行全量重训：",
                "",
                "```powershell",
                ".\\.venv\\Scripts\\python.exe train_models.py "
                "--model all --validation-only",
                "```",
            ]
        )

    (output_dir / "README.md").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )
    print(f"Report assets written to: {output_dir}")
    if not generated_validation_assets:
        print("Validation-only predictions are missing; ROC/PR assets were skipped.")


if __name__ == "__main__":
    main()
