from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from src.build_table import DEFAULT_DATASET_ROOT, DEFAULT_OUTPUT_DIR, _compute_row
from src.tfrecord_reader import (
    build_raw_dataset,
    iter_examples,
    list_scenarios,
    scenario_files_for_splits,
)


CSV_FIELDS = [
    "scenario",
    "record_index",
    "num_instances",
    "camera_translation_speed_mean",
    "camera_rotation_change_mean",
    "instance_speed_mean",
    "reliability_score",
]


def _normalize_splits(raw_splits: list[str]) -> list[str]:
    splits: list[str] = []
    for item in raw_splits:
        parts = [p.strip() for p in item.split(",") if p.strip()]
        splits.extend(parts)
    # Keep order deterministic while removing duplicates.
    unique: list[str] = []
    seen: set[str] = set()
    for split in splits:
        if split not in seen:
            seen.add(split)
            unique.append(split)
    return unique or ["train"]


def _safe_row(scenario: str, record_index: int) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "record_index": record_index,
        "num_instances": math.nan,
        "camera_translation_speed_mean": math.nan,
        "camera_rotation_change_mean": math.nan,
        "instance_speed_mean": math.nan,
        "reliability_score": math.nan,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build one combined per-example table across all scenarios."
    )
    parser.add_argument(
        "--dataset_root",
        default=str(DEFAULT_DATASET_ROOT),
        help=f"Dataset root containing scenario folders (default: {DEFAULT_DATASET_ROOT}).",
    )
    parser.add_argument(
        "--max_records",
        type=int,
        default=5000,
        help="Maximum records to process per scenario (default: 5000).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        help="Split(s) to include: e.g. --splits train or --splits train validation test.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    splits = _normalize_splits(args.splits)
    scenarios = list_scenarios(args.dataset_root)
    if not scenarios:
        raise FileNotFoundError(f"No scenario folders found under {args.dataset_root}.")

    print(f"Found {len(scenarios)} scenario folder(s).")
    print(f"Including splits: {splits}")

    rows: list[dict[str, Any]] = []
    rows_per_scenario: dict[str, int] = {}
    skipped_scenarios: list[str] = []
    found_splits: set[str] = set()

    for scenario in scenarios:
        all_files, files_by_split = scenario_files_for_splits(args.dataset_root, scenario, splits)
        for split_name, split_files in files_by_split.items():
            if split_files:
                found_splits.add(split_name)
        if not all_files:
            skipped_scenarios.append(scenario)
            print(f"Skipping scenario '{scenario}': 0 records (no matching shards).")
            continue

        print(f"Scenario '{scenario}': {len(all_files)} shard(s).")
        dataset = build_raw_dataset(all_files)

        scenario_rows = 0
        for example in iter_examples(dataset):
            if scenario_rows >= args.max_records:
                break
            try:
                row = _compute_row(example=example, scenario=scenario, index=scenario_rows)
            except Exception:
                row = _safe_row(scenario, scenario_rows)
            rows.append(row)
            scenario_rows += 1
            if scenario_rows == 1 or scenario_rows % 100 == 0 or scenario_rows == args.max_records:
                print(f"  processed {scenario_rows} record(s) for '{scenario}'...")

        rows_per_scenario[scenario] = scenario_rows
        if scenario_rows == 0:
            skipped_scenarios.append(scenario)
            print(f"Scenario '{scenario}': 0 decoded records, skipped.")
        else:
            print(f"Scenario '{scenario}' done: {scenario_rows} row(s).")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "table_all_scenarios.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset_root": str(args.dataset_root),
        "included_splits_requested": splits,
        "included_splits_found": sorted(found_splits),
        "total_rows": len(rows),
        "rows_per_scenario": rows_per_scenario,
        "skipped_scenarios": skipped_scenarios,
    }
    summary_path = output_dir / "table_all_scenarios_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved combined CSV: {csv_path}")
    print(f"Saved summary JSON: {summary_path}")
    print(f"Total rows written: {len(rows)}")


if __name__ == "__main__":
    main()
