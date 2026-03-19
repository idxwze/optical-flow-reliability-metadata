from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
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


def _target_to_filename_token(target_column: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", target_column.strip())
    return token or "target"


def load_dataset(csv_path: str | Path, target_column: str):
    scenarios: list[str] = []
    record_indices: list[int] = []
    x_rows: list[list[float]] = []
    y_vals: list[float] = []

    with Path(csv_path).open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"scenario", "record_index", target_column, *FEATURE_COLUMNS}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

        for row in reader:
            y = _to_float(row[target_column])
            if np.isnan(y):
                # Skip rows without target.
                continue

            features = [_to_float(row[col]) for col in FEATURE_COLUMNS]
            x_rows.append(features)
            y_vals.append(y)
            scenarios.append(row["scenario"])
            try:
                record_indices.append(int(float(row["record_index"])))
            except Exception:
                record_indices.append(-1)

    if not x_rows:
        raise ValueError("No usable rows found in CSV after filtering missing target values.")

    X = np.asarray(x_rows, dtype=np.float64)
    y = np.asarray(y_vals, dtype=np.float64)
    scenario_arr = np.asarray(scenarios, dtype=object)
    record_index_arr = np.asarray(record_indices, dtype=np.int64)
    return X, y, scenario_arr, record_index_arr


def evaluate_and_save(
    model_name: str,
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    scenarios: np.ndarray,
    record_indices: np.ndarray,
    output_dir: Path,
    target_column: str,
):
    # Fit/test pipeline with median imputation.
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ]
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scenario_train,
        scenario_test,
        record_idx_train,
        record_idx_test,
    ) = train_test_split(
        X,
        y,
        scenarios,
        record_indices,
        test_size=0.2,
        random_state=42,
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    spearman = float(spearmanr(y_test, y_pred).correlation)

    # 5-fold CV MAE on full data.
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_mae_scores = -cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=None,
    )
    cv_mae_mean = float(np.mean(cv_mae_scores))
    cv_mae_std = float(np.std(cv_mae_scores))

    metrics = {
        "model": model_name,
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "test_metrics": {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "spearman": spearman,
        },
        "cv_metrics": {
            "n_splits": 5,
            "mae_mean": cv_mae_mean,
            "mae_std": cv_mae_std,
        },
        "features": FEATURE_COLUMNS,
        "target": target_column,
    }

    target_token = _target_to_filename_token(target_column)
    metrics_path = output_dir / f"metrics_{model_name}_{target_token}.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    preds_path = output_dir / f"preds_{model_name}_{target_token}.csv"
    with preds_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "record_index", "y_true", "y_pred"])
        for scenario, ridx, yt, yp in zip(scenario_test, record_idx_test, y_test, y_pred):
            writer.writerow([scenario, int(ridx), float(yt), float(yp)])

    print(
        f"[{model_name}] MAE={mae:.6f} RMSE={rmse:.6f} R2={r2:.6f} "
        f"Spearman={spearman:.6f} | CV MAE={cv_mae_mean:.6f}±{cv_mae_std:.6f}"
    )
    print(f"[{model_name}] Saved {metrics_path} and {preds_path}")


def main():
    parser = argparse.ArgumentParser(description="Train regression baselines.")
    parser.add_argument(
        "--csv",
        default="outputs/table_all_scenarios.csv",
        help="Path to combined CSV (default: outputs/table_all_scenarios.csv).",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Directory to write metrics and predictions (default: outputs).",
    )
    parser.add_argument(
        "--target",
        default=TARGET_COLUMN,
        help=f"Target column name in CSV (default: {TARGET_COLUMN}).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, scenarios, record_indices = load_dataset(args.csv, args.target)
    print(f"Loaded {X.shape[0]} rows from {args.csv} with target '{args.target}'")

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(random_state=42),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
    }

    for name, model in models.items():
        evaluate_and_save(
            model_name=name,
            estimator=model,
            X=X,
            y=y,
            scenarios=scenarios,
            record_indices=record_indices,
            output_dir=output_dir,
            target_column=args.target,
        )


if __name__ == "__main__":
    main()
