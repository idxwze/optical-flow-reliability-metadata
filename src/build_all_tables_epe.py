from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from src.epe import decode_ground_truth_forward_flow, mean_epe
from src.example_features import (
    FEATURE_COLUMNS,
    compute_metadata_features,
    get_feature_array,
    get_scalar_int,
)
from src.flow_estimator import decode_video_frames, estimate_farneback_flow
from src.tfrecord_reader import (
    build_raw_dataset,
    iter_examples,
    list_scenarios,
    scenario_files_for_splits,
)


DEFAULT_DATASET_ROOT = Path("/Users/seifeddinereguige/Documents/tfds_Dataset")
DEFAULT_OUTPUT_DIR = Path("outputs")

CSV_FIELDS = [
    "scenario",
    "record_index",
    *FEATURE_COLUMNS,
    "epe_mean",
]


def _normalize_splits(raw_splits: list[str]) -> list[str]:
    splits: list[str] = []
    for item in raw_splits:
        parts = [p.strip() for p in item.split(",") if p.strip()]
        splits.extend(parts)
    unique: list[str] = []
    seen: set[str] = set()
    for split in splits:
        if split not in seen:
            seen.add(split)
            unique.append(split)
    return unique or ["train"]


def _compute_row(example, scenario: str, index: int) -> dict[str, Any]:
    num_frames = get_scalar_int(example, "metadata/num_frames")
    height = get_scalar_int(example, "metadata/height")
    width = get_scalar_int(example, "metadata/width")

    if num_frames is None or height is None or width is None:
        raise ValueError("Missing one of metadata/num_frames,height,width.")

    metadata_features = compute_metadata_features(example)

    video = get_feature_array(example, "video")
    if video.size == 0:
        raise ValueError("Missing video feature.")
    frame_bytes = [bytes(x) for x in video.reshape(-1)]
    frames = decode_video_frames(
        video_bytes=frame_bytes,
        height=height,
        width=width,
        num_frames=num_frames,
    )
    pred = estimate_farneback_flow(frames)

    forward_flow = get_feature_array(example, "forward_flow")
    flow_range = get_feature_array(example, "metadata/forward_flow_range")
    if forward_flow.size == 0 or flow_range.size < 2:
        raise ValueError("Missing forward_flow or metadata/forward_flow_range.")
    gt = decode_ground_truth_forward_flow(
        raw_flow=forward_flow,
        flow_range=flow_range,
        num_frames=num_frames,
        height=height,
        width=width,
    )
    epe_value = mean_epe(pred_flow_tminus1=pred, gt_flow_t=gt)

    return {
        "scenario": scenario,
        "record_index": index,
        **metadata_features,
        "epe_mean": epe_value,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build combined per-example table across scenarios with Farneback EPE target."
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
        "--max_scenarios",
        type=int,
        default=None,
        help="Optional cap on number of scenarios to process.",
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
    if args.max_scenarios is not None:
        scenarios = scenarios[: int(args.max_scenarios)]

    print(f"Found {len(scenarios)} scenario folder(s).")
    print(f"Including splits: {splits}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "table_all_scenarios_epe.csv"

    rows_per_scenario: dict[str, int] = {}
    skipped_scenarios: list[str] = []
    found_splits: set[str] = set()
    decode_failures = 0
    total_rows = 0

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

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
            scenario_seen = 0
            for example in iter_examples(dataset):
                if scenario_seen >= args.max_records:
                    break
                try:
                    row = _compute_row(example=example, scenario=scenario, index=scenario_seen)
                except Exception as exc:
                    decode_failures += 1
                    print(
                        f"  skip scenario='{scenario}' record_index={scenario_seen}: {exc}"
                    )
                    scenario_seen += 1
                    continue

                writer.writerow(row)
                scenario_seen += 1
                scenario_rows += 1
                total_rows += 1
                if scenario_seen == 1 or scenario_seen % 50 == 0 or scenario_seen == args.max_records:
                    print(
                        f"  scanned {scenario_seen} sample(s), kept {scenario_rows} valid for '{scenario}'..."
                    )

            rows_per_scenario[scenario] = scenario_rows
            if scenario_rows == 0:
                skipped_scenarios.append(scenario)
                print(f"Scenario '{scenario}': 0 valid records, skipped.")
            else:
                print(f"Scenario '{scenario}' done: {scenario_rows} row(s).")

    summary = {
        "dataset_root": str(args.dataset_root),
        "included_splits_requested": splits,
        "included_splits_found": sorted(found_splits),
        "max_records_per_scenario": args.max_records,
        "max_scenarios": args.max_scenarios,
        "total_rows": total_rows,
        "rows_per_scenario": rows_per_scenario,
        "skipped_scenarios": skipped_scenarios,
        "decode_or_compute_failures": decode_failures,
    }
    summary_path = output_dir / "table_all_scenarios_epe_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved combined CSV: {csv_path}")
    print(f"Saved summary JSON: {summary_path}")
    print(f"Total rows written: {total_rows}")
    print(f"Total skipped decode/compute samples: {decode_failures}")


if __name__ == "__main__":
    main()
