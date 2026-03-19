from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from src.build_table import DEFAULT_DATASET_ROOT, DEFAULT_OUTPUT_DIR
from src.epe_metrics import compute_raft_epe
from src.example_features import compute_metadata_features, get_feature_array, get_scalar_int
from src.raft_infer import load_raft_runner
from src.tfrecord_reader import (
    build_raw_dataset,
    iter_examples,
    list_scenarios,
    scenario_files_for_splits,
)
from src.video_decode import decode_video_frames


CSV_FIELDS = [
    "scenario",
    "record_index",
    "num_instances",
    "camera_translation_speed_mean",
    "camera_rotation_change_mean",
    "instance_speed_mean",
    "epe_mean_raft",
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


def _cache_path(cache_dir: Path, scenario: str, record_index: int) -> Path:
    safe_scenario = scenario.replace("/", "_")
    return cache_dir / f"{safe_scenario}_{record_index}.json"


def _load_cached_epe(cache_file: Path) -> float | None:
    if not cache_file.exists():
        return None
    try:
        with cache_file.open() as f:
            payload = json.load(f)
        value = float(payload["epe_mean_raft"])
    except Exception:
        return None
    return value


def _save_cached_epe(cache_file: Path, epe_mean_raft: float) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with cache_file.open("w") as f:
        json.dump({"epe_mean_raft": float(epe_mean_raft)}, f)


def _compute_epe_for_example(example, raft_runner, max_pairs: int | None) -> float:
    num_frames = get_scalar_int(example, "metadata/num_frames")
    height = get_scalar_int(example, "metadata/height")
    width = get_scalar_int(example, "metadata/width")
    if num_frames is None or height is None or width is None:
        raise ValueError("Missing one of metadata/num_frames,height,width.")

    video = get_feature_array(example, "video")
    forward_flow = get_feature_array(example, "forward_flow")
    flow_range = get_feature_array(example, "metadata/forward_flow_range")
    if video.size == 0:
        raise ValueError("Missing video feature.")
    if forward_flow.size == 0 or flow_range.size < 2:
        raise ValueError("Missing forward_flow or metadata/forward_flow_range.")

    frame_bytes = [bytes(x) for x in video.reshape(-1)]
    frames = decode_video_frames(
        video_bytes=frame_bytes,
        height=height,
        width=width,
        num_frames=num_frames,
    )

    return compute_raft_epe(
        video_frames=frames,
        raw_flow=forward_flow,
        flow_range=flow_range,
        num_frames=num_frames,
        height=height,
        width=width,
        raft_runner=raft_runner,
        max_pairs=max_pairs,
    )


def _build_row(
    example,
    scenario: str,
    record_index: int,
    raft_runner,
    cache_dir: Path,
    max_pairs: int | None,
) -> dict[str, Any]:
    row = {
        "scenario": scenario,
        "record_index": record_index,
        **compute_metadata_features(example),
    }

    cache_file = _cache_path(cache_dir=cache_dir, scenario=scenario, record_index=record_index)
    epe_mean_raft = _load_cached_epe(cache_file)
    if epe_mean_raft is None:
        epe_mean_raft = _compute_epe_for_example(
            example=example,
            raft_runner=raft_runner,
            max_pairs=max_pairs,
        )
        _save_cached_epe(cache_file=cache_file, epe_mean_raft=epe_mean_raft)

    row["epe_mean_raft"] = epe_mean_raft
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Build combined per-example table across scenarios with torchvision RAFT EPE target."
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
        "--max_pairs",
        type=int,
        default=None,
        help="Optional cap on the number of frame pairs per video.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        help="Split(s) to include: e.g. --splits train or --splits train validation test.",
    )
    parser.add_argument(
        "--raft_model",
        default="small",
        choices=["small", "large"],
        help="RAFT model variant to use (default: small).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override, e.g. cpu or mps.",
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

    raft_runner = load_raft_runner(
        raft_model=args.raft_model,
        device=args.device,
        progress=True,
    )

    print(f"Found {len(scenarios)} scenario folder(s).")
    print(f"Including splits: {splits}")
    print(f"Using RAFT model '{raft_runner.raft_model}' on device '{raft_runner.device}'.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache_raft"
    csv_path = output_dir / "table_all_scenarios_raft_epe.csv"

    rows_per_scenario: dict[str, int] = {}
    skipped_scenarios: list[str] = []
    found_splits: set[str] = set()
    skipped_samples = 0
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
                    row = _build_row(
                        example=example,
                        scenario=scenario,
                        record_index=scenario_seen,
                        raft_runner=raft_runner,
                        cache_dir=cache_dir,
                        max_pairs=args.max_pairs,
                    )
                except Exception as exc:
                    skipped_samples += 1
                    print(f"  skip scenario='{scenario}' record_index={scenario_seen}: {exc}")
                    scenario_seen += 1
                    continue

                writer.writerow(row)
                scenario_seen += 1
                scenario_rows += 1
                total_rows += 1
                if scenario_seen == 1 or scenario_seen % 10 == 0 or scenario_seen == args.max_records:
                    print(
                        f"  processed {scenario_seen} record(s) for '{scenario}' "
                        f"(written={scenario_rows}, skipped={skipped_samples})..."
                    )

            rows_per_scenario[scenario] = scenario_rows
            if scenario_rows == 0:
                skipped_scenarios.append(scenario)
                print(f"Scenario '{scenario}': 0 decoded records written.")
            else:
                print(f"Scenario '{scenario}' done: {scenario_rows} row(s).")

    summary = {
        "dataset_root": str(args.dataset_root),
        "included_splits_requested": splits,
        "included_splits_found": sorted(found_splits),
        "total_rows": total_rows,
        "skipped_samples": skipped_samples,
        "rows_per_scenario": rows_per_scenario,
        "skipped_scenarios": skipped_scenarios,
        "raft_model": raft_runner.raft_model,
        "device": raft_runner.device,
        "max_pairs": args.max_pairs,
    }
    summary_path = output_dir / "table_all_scenarios_raft_epe_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved combined CSV: {csv_path}")
    print(f"Saved summary JSON: {summary_path}")
    print(f"Total rows written: {total_rows}")


if __name__ == "__main__":
    main()
