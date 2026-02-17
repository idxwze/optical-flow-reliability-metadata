from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


FEATURE_COLUMNS = [
    "num_instances",
    "camera_translation_speed_mean",
    "camera_rotation_change_mean",
    "instance_speed_mean",
]
TARGET_COLUMN = "reliability_score"


def _to_float(value: str) -> float:
    if value is None:
        return np.nan
    v = value.strip()
    if v == "" or v.lower() == "nan":
        return np.nan
    return float(v)


def load_table(table_path: str | Path):
    scenarios: list[str] = []
    x_rows: list[list[float]] = []
    y_vals: list[float] = []
    y_by_scenario: dict[str, list[float]] = {}

    with Path(table_path).open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"scenario", TARGET_COLUMN, *FEATURE_COLUMNS}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in table CSV: {sorted(missing)}")

        for row in reader:
            scenario = row["scenario"]
            y = _to_float(row[TARGET_COLUMN])
            if not np.isnan(y):
                y_by_scenario.setdefault(scenario, []).append(y)
                y_vals.append(y)
                scenarios.append(scenario)
                x_rows.append([_to_float(row[col]) for col in FEATURE_COLUMNS])

    if not y_vals:
        raise ValueError("No valid target values in table CSV.")

    X = np.asarray(x_rows, dtype=np.float64)
    y = np.asarray(y_vals, dtype=np.float64)
    return X, y, y_by_scenario


def load_preds(preds_path: str | Path):
    y_true: list[float] = []
    y_pred: list[float] = []
    with Path(preds_path).open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"y_true", "y_pred"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in preds CSV: {sorted(missing)}")
        for row in reader:
            yt = _to_float(row["y_true"])
            yp = _to_float(row["y_pred"])
            if np.isnan(yt) or np.isnan(yp):
                continue
            y_true.append(yt)
            y_pred.append(yp)
    if not y_true:
        raise ValueError("No valid rows in preds CSV.")
    return np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)


def plot_scatter_true_vs_pred(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_true, y_pred, s=20, alpha=0.7)
    min_v = float(min(np.min(y_true), np.min(y_pred)))
    max_v = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", linewidth=1.5)
    ax.set_xlabel("True reliability score")
    ax.set_ylabel("Predicted reliability score")
    ax.set_title("True vs Predicted Reliability Score")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_hist_score_by_scenario(y_by_scenario: dict[str, list[float]], out_path: Path):
    # Use horizontal boxplots for readability with many scenarios.
    ordered = sorted(y_by_scenario.items(), key=lambda kv: np.median(kv[1]))
    labels = [k for k, _ in ordered]
    data = [v for _, v in ordered]

    height = max(6, 0.32 * len(labels))
    fig, ax = plt.subplots(figsize=(12, height))
    ax.boxplot(data, vert=False, labels=labels, showfliers=False)
    ax.set_xlabel("Reliability score")
    ax.set_ylabel("Scenario")
    ax.set_title("Reliability Score Distribution by Scenario")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_feature_importance(X: np.ndarray, y: np.ndarray, out_path: Path):
    # Re-train quickly to get importances if model object wasn't persisted.
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(random_state=42)),
        ]
    )
    pipeline.fit(X, y)
    model: GradientBoostingRegressor = pipeline.named_steps["model"]
    importances = model.feature_importances_

    order = np.argsort(importances)[::-1]
    sorted_features = [FEATURE_COLUMNS[i] for i in order]
    sorted_importances = importances[order]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(sorted_features, sorted_importances)
    ax.set_ylabel("Importance")
    ax.set_title("GradientBoosting Feature Importances")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate report-ready plots.")
    parser.add_argument(
        "--table",
        default="outputs/table_all_scenarios.csv",
        help="Path to table CSV (default: outputs/table_all_scenarios.csv).",
    )
    parser.add_argument(
        "--preds",
        default="outputs/preds_gradient_boosting.csv",
        help="Path to predictions CSV (default: outputs/preds_gradient_boosting.csv).",
    )
    parser.add_argument(
        "--out_dir",
        default="reports/figures",
        help="Directory for output figures (default: reports/figures).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, y_by_scenario = load_table(args.table)
    y_true, y_pred = load_preds(args.preds)

    scatter_path = out_dir / "scatter_true_vs_pred.png"
    scenario_path = out_dir / "hist_score_by_scenario.png"
    importance_path = out_dir / "feature_importance.png"

    plot_scatter_true_vs_pred(y_true, y_pred, scatter_path)
    plot_hist_score_by_scenario(y_by_scenario, scenario_path)
    plot_feature_importance(X, y, importance_path)

    print(f"Saved: {scatter_path}")
    print(f"Saved: {scenario_path}")
    print(f"Saved: {importance_path}")


if __name__ == "__main__":
    main()
