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


def _is_numeric_like(value: str | None) -> bool:
    if value is None:
        return True
    stripped = value.strip()
    if stripped == "" or stripped.lower() == "nan":
        return True
    try:
        float(stripped)
        return True
    except Exception:
        return False


def infer_feature_columns(
    rows: list[dict[str, str]],
    fieldnames: list[str] | None,
    target_column: str,
) -> list[str]:
    available = list(fieldnames or [])
    excluded = {"scenario", "record_index", target_column}
    feature_columns: list[str] = []
    for name in available:
        if name in excluded:
            continue
        if all(_is_numeric_like(row.get(name)) for row in rows):
            if all(np.isnan(_to_float(row.get(name))) for row in rows):
                continue
            feature_columns.append(name)
    if not feature_columns:
        raise ValueError(
            "No numeric feature columns found after excluding scenario, record_index, and target."
        )
    return feature_columns


def load_dataset(csv_path: str | Path, target_column: str):
    scenarios: list[str] = []
    record_indices: list[int] = []
    x_rows: list[list[float]] = []
    y_vals: list[float] = []

    with Path(csv_path).open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        feature_columns = infer_feature_columns(rows, reader.fieldnames, target_column)
        required = {"scenario", "record_index", target_column, *feature_columns}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

        for row in rows:
            y = _to_float(row[target_column])
            if np.isnan(y):
                continue

            features = [_to_float(row[col]) for col in feature_columns]
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
    return X, y, scenario_arr, record_index_arr, feature_columns


def make_split(
    X: np.ndarray,
    y: np.ndarray,
    scenarios: np.ndarray,
    record_indices: np.ndarray,
    split_mode: str,
    seed: int,
):
    if split_mode == "random":
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
            random_state=seed,
        )
        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "scenario_train": scenario_train,
            "scenario_test": scenario_test,
            "record_idx_train": record_idx_train,
            "record_idx_test": record_idx_test,
            "held_out_scenarios": [],
        }

    if split_mode != "scenario":
        raise ValueError(f"Unsupported split_mode '{split_mode}'.")

    unique_scenarios = np.unique(scenarios)
    if unique_scenarios.size < 2:
        raise ValueError("Scenario split requires at least 2 unique scenarios.")

    _, test_scenarios = train_test_split(
        unique_scenarios,
        test_size=0.2,
        random_state=seed,
    )
    test_scenario_set = set(test_scenarios.tolist())
    test_mask = np.asarray([scenario in test_scenario_set for scenario in scenarios], dtype=bool)
    train_mask = ~test_mask

    if not np.any(train_mask):
        raise ValueError("Scenario split produced an empty training set.")
    if not np.any(test_mask):
        raise ValueError("Scenario split produced an empty test set.")

    return {
        "X_train": X[train_mask],
        "X_test": X[test_mask],
        "y_train": y[train_mask],
        "y_test": y[test_mask],
        "scenario_train": scenarios[train_mask],
        "scenario_test": scenarios[test_mask],
        "record_idx_train": record_indices[train_mask],
        "record_idx_test": record_indices[test_mask],
        "held_out_scenarios": sorted(test_scenario_set),
    }


def _safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    corr = spearmanr(y_true, y_pred).correlation
    if corr is None or np.isnan(corr):
        return float("nan")
    return float(corr)


def evaluate_model(
    model_name: str,
    estimator,
    split: dict[str, np.ndarray | list[str]],
    target_column: str,
    split_mode: str,
    feature_columns: list[str],
):
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ]
    )

    X_train = np.asarray(split["X_train"], dtype=np.float64)
    X_test = np.asarray(split["X_test"], dtype=np.float64)
    y_train = np.asarray(split["y_train"], dtype=np.float64)
    y_test = np.asarray(split["y_test"], dtype=np.float64)
    scenario_test = np.asarray(split["scenario_test"], dtype=object)
    record_idx_test = np.asarray(split["record_idx_test"], dtype=np.int64)
    held_out_scenarios = list(split["held_out_scenarios"])
    active_feature_columns = list(feature_columns)

    all_nan_mask = np.all(np.isnan(X_train), axis=0)
    dropped_columns = [
        column_name
        for column_name, should_drop in zip(active_feature_columns, all_nan_mask)
        if should_drop
    ]
    if dropped_columns:
        keep_mask = ~all_nan_mask
        X_train = X_train[:, keep_mask]
        X_test = X_test[:, keep_mask]
        active_feature_columns = [
            column_name
            for column_name, should_keep in zip(active_feature_columns, keep_mask)
            if should_keep
        ]
    print(f"[{model_name}] Dropped all-NaN train columns: {dropped_columns}")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    spearman = _safe_spearman(y_test, y_pred)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_mae_scores = -cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=None,
    )
    cv_mae_mean = float(np.mean(cv_mae_scores))
    cv_mae_std = float(np.std(cv_mae_scores))

    metrics = {
        "model": model_name,
        "n_rows": int(X_train.shape[0] + X_test.shape[0]),
        "n_train_rows": int(X_train.shape[0]),
        "n_test_rows": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "split_mode": split_mode,
        "held_out_scenarios": held_out_scenarios,
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
        "features": active_feature_columns,
        "dropped_all_nan_train_features": dropped_columns,
        "target": target_column,
    }

    preds_rows = [
        {
            "scenario": str(scenario),
            "record_index": int(ridx),
            "y_true": float(yt),
            "y_pred": float(yp),
        }
        for scenario, ridx, yt, yp in zip(scenario_test, record_idx_test, y_test, y_pred)
    ]

    return metrics, preds_rows


def save_single_run_outputs(
    model_name: str,
    metrics: dict,
    preds_rows: list[dict[str, object]],
    output_dir: Path,
    target_column: str,
) -> None:
    target_token = _target_to_filename_token(target_column)
    metrics_path = output_dir / f"metrics_{model_name}_{target_token}.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    preds_path = output_dir / f"preds_{model_name}_{target_token}.csv"
    with preds_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "record_index", "y_true", "y_pred"])
        for row in preds_rows:
            writer.writerow(
                [row["scenario"], int(row["record_index"]), float(row["y_true"]), float(row["y_pred"])]
            )

    tm = metrics["test_metrics"]
    cvm = metrics["cv_metrics"]
    print(
        f"[{model_name}] MAE={tm['mae']:.6f} RMSE={tm['rmse']:.6f} R2={tm['r2']:.6f} "
        f"Spearman={tm['spearman']:.6f} | CV(train) MAE={cvm['mae_mean']:.6f}±{cvm['mae_std']:.6f}"
    )
    print(f"[{model_name}] Saved {metrics_path} and {preds_path}")


def summarize_repeated_metrics(run_metrics: list[dict]) -> dict[str, dict[str, float]]:
    metric_names = ["mae", "rmse", "r2", "spearman"]
    summary: dict[str, dict[str, float]] = {}
    for metric_name in metric_names:
        values = np.asarray(
            [run["test_metrics"][metric_name] for run in run_metrics],
            dtype=np.float64,
        )
        summary[metric_name] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
        }
    return summary


def run_single_evaluation(
    *,
    models: dict[str, object],
    split: dict[str, np.ndarray | list[str]],
    output_dir: Path,
    target_column: str,
    split_mode: str,
    feature_columns: list[str],
    save_outputs: bool,
):
    results: dict[str, dict] = {}
    for name, model in models.items():
        metrics, preds_rows = evaluate_model(
            model_name=name,
            estimator=model,
            split=split,
            target_column=target_column,
            split_mode=split_mode,
            feature_columns=feature_columns,
        )
        results[name] = {
            "metrics": metrics,
            "preds_rows": preds_rows,
        }
        if save_outputs:
            save_single_run_outputs(
                model_name=name,
                metrics=metrics,
                preds_rows=preds_rows,
                output_dir=output_dir,
                target_column=target_column,
            )
        else:
            tm = metrics["test_metrics"]
            cvm = metrics["cv_metrics"]
            print(
                f"[{name}] MAE={tm['mae']:.6f} RMSE={tm['rmse']:.6f} R2={tm['r2']:.6f} "
                f"Spearman={tm['spearman']:.6f} | CV(train) MAE={cvm['mae_mean']:.6f}±{cvm['mae_std']:.6f}"
            )
    return results


def save_repeated_summary(
    *,
    output_dir: Path,
    target_column: str,
    split_mode: str,
    repeats: int,
    base_seed: int,
    per_model_runs: dict[str, list[dict]],
) -> None:
    target_token = _target_to_filename_token(target_column)
    summary_path = output_dir / f"metrics_summary_{target_token}_{split_mode}.json"

    summary_payload = {
        "target": target_column,
        "split_mode": split_mode,
        "repeats": repeats,
        "base_seed": base_seed,
        "models": {},
    }

    for model_name, run_metrics in per_model_runs.items():
        summary_payload["models"][model_name] = {
            "per_run": run_metrics,
            "summary": summarize_repeated_metrics(run_metrics),
        }

    with summary_path.open("w") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"Saved repeated scenario summary: {summary_path}")


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
    parser.add_argument(
        "--split_mode",
        default="random",
        choices=["random", "scenario"],
        help="Split strategy: random row split or scenario holdout (default: random).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repeated scenario holdout runs (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for splitting (default: 42).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, scenarios, record_indices, feature_columns = load_dataset(args.csv, args.target)
    print(
        f"Loaded {X.shape[0]} rows from {args.csv} with target '{args.target}' "
        f"using split_mode='{args.split_mode}'"
    )
    print(f"Using feature columns: {feature_columns}")

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(random_state=42),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
    }

    if args.split_mode == "scenario" and args.repeats > 1:
        per_model_runs: dict[str, list[dict]] = {name: [] for name in models}

        for repeat_idx in range(args.repeats):
            run_seed = args.seed + repeat_idx
            print(f"\nScenario holdout repeat {repeat_idx + 1}/{args.repeats} with seed={run_seed}")
            split = make_split(
                X=X,
                y=y,
                scenarios=scenarios,
                record_indices=record_indices,
                split_mode=args.split_mode,
                seed=run_seed,
            )
            print(
                f"Train rows: {np.asarray(split['X_train']).shape[0]} | "
                f"Test rows: {np.asarray(split['X_test']).shape[0]}"
            )
            held_out = list(split["held_out_scenarios"])
            print(f"Held-out scenarios ({len(held_out)}): {held_out}")

            results = run_single_evaluation(
                models=models,
                split=split,
                output_dir=output_dir,
                target_column=args.target,
                split_mode=args.split_mode,
                feature_columns=feature_columns,
                save_outputs=False,
            )
            for model_name, payload in results.items():
                metrics = dict(payload["metrics"])
                metrics["run_index"] = repeat_idx
                metrics["seed"] = run_seed
                per_model_runs[model_name].append(metrics)

        print("")
        for model_name, run_metrics in per_model_runs.items():
            summary = summarize_repeated_metrics(run_metrics)
            print(
                f"[{model_name}] "
                f"MAE={summary['mae']['mean']:.6f}±{summary['mae']['std']:.6f} "
                f"RMSE={summary['rmse']['mean']:.6f}±{summary['rmse']['std']:.6f} "
                f"R2={summary['r2']['mean']:.6f}±{summary['r2']['std']:.6f} "
                f"Spearman={summary['spearman']['mean']:.6f}±{summary['spearman']['std']:.6f}"
            )

        save_repeated_summary(
            output_dir=output_dir,
            target_column=args.target,
            split_mode=args.split_mode,
            repeats=args.repeats,
            base_seed=args.seed,
            per_model_runs=per_model_runs,
        )
        return

    split = make_split(
        X=X,
        y=y,
        scenarios=scenarios,
        record_indices=record_indices,
        split_mode=args.split_mode,
        seed=args.seed,
    )
    print(
        f"Train rows: {np.asarray(split['X_train']).shape[0]} | "
        f"Test rows: {np.asarray(split['X_test']).shape[0]}"
    )
    if args.split_mode == "scenario":
        held_out = list(split["held_out_scenarios"])
        print(f"Held-out scenarios ({len(held_out)}): {held_out}")

    run_single_evaluation(
        models=models,
        split=split,
        output_dir=output_dir,
        target_column=args.target,
        split_mode=args.split_mode,
        feature_columns=feature_columns,
        save_outputs=True,
    )


if __name__ == "__main__":
    main()
