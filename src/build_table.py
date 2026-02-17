from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.decode_flow import decode_forward_flow
from src.score import reliability_score_from_flow
from src.tfrecord_reader import build_raw_dataset, iter_examples, scenario_train_files


DEFAULT_DATASET_ROOT = Path("/Users/seifeddinereguige/Documents/tfds_Dataset")
DEFAULT_OUTPUT_DIR = Path("outputs")


def _feature_to_numpy_list(feature: Any) -> np.ndarray:
    if feature is None:
        return np.array([])
    if feature.int64_list.value:
        return np.asarray(feature.int64_list.value)
    if feature.float_list.value:
        return np.asarray(feature.float_list.value)
    if feature.bytes_list.value:
        return np.asarray(feature.bytes_list.value)
    return np.array([])


def _get_feature_array(example, key: str) -> np.ndarray:
    feats = example.features.feature
    if key not in feats:
        return np.array([])
    return _feature_to_numpy_list(feats[key])


def _get_scalar_int(example, key: str) -> int | None:
    arr = _get_feature_array(example, key)
    if arr.size == 0:
        return None
    try:
        return int(arr.reshape(-1)[0])
    except Exception:
        return None


def _camera_translation_speed_mean(positions: np.ndarray, num_frames: int | None) -> float:
    if positions.size == 0:
        return math.nan
    arr = np.asarray(positions, dtype=np.float32).reshape(-1)
    if num_frames is not None and num_frames > 0 and arr.size == num_frames * 3:
        pos = arr.reshape(num_frames, 3)
    elif arr.size % 3 == 0:
        pos = arr.reshape(-1, 3)
    else:
        return math.nan
    if pos.shape[0] < 2:
        return 0.0
    deltas = pos[1:] - pos[:-1]
    speeds = np.linalg.norm(deltas, axis=-1)
    return float(np.mean(speeds))


def _camera_rotation_change_mean(quaternions: np.ndarray, num_frames: int | None) -> float:
    if quaternions.size == 0:
        return math.nan
    arr = np.asarray(quaternions, dtype=np.float64).reshape(-1)
    if num_frames is not None and num_frames > 0 and arr.size == num_frames * 4:
        q = arr.reshape(num_frames, 4)
    elif arr.size % 4 == 0:
        q = arr.reshape(-1, 4)
    else:
        return math.nan
    if q.shape[0] < 2:
        return 0.0
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    q = q / norms
    # Quaternion delta approximation per consecutive frame: 2*acos(|dot(q_t, q_t+1)|)
    dots = np.abs(np.sum(q[:-1] * q[1:], axis=-1))
    dots = np.clip(dots, -1.0, 1.0)
    angles = 2.0 * np.arccos(dots)
    return float(np.mean(angles))


def _instance_speed_mean(
    velocities: np.ndarray,
    num_instances: int | None,
    num_frames: int | None,
) -> float:
    if velocities.size == 0:
        return math.nan
    arr = np.asarray(velocities, dtype=np.float32).reshape(-1)
    if num_instances and num_frames and arr.size == num_instances * num_frames * 3:
        vel = arr.reshape(num_instances, num_frames, 3)
        mag = np.linalg.norm(vel, axis=-1)
        return float(np.mean(mag))
    if arr.size % 3 != 0:
        return math.nan
    vel = arr.reshape(-1, 3)
    mag = np.linalg.norm(vel, axis=-1)
    return float(np.mean(mag))


def _compute_row(example, scenario: str, index: int) -> dict[str, Any]:
    num_frames = _get_scalar_int(example, "metadata/num_frames")
    height = _get_scalar_int(example, "metadata/height")
    width = _get_scalar_int(example, "metadata/width")
    num_instances = _get_scalar_int(example, "metadata/num_instances")

    positions = _get_feature_array(example, "camera/positions")
    quaternions = _get_feature_array(example, "camera/quaternions")
    velocities = _get_feature_array(example, "instances/velocities")

    cam_speed_mean = _camera_translation_speed_mean(positions, num_frames)
    cam_rot_change_mean = _camera_rotation_change_mean(quaternions, num_frames)
    inst_speed_mean = _instance_speed_mean(velocities, num_instances, num_frames)

    score = math.nan
    forward_flow = _get_feature_array(example, "forward_flow")
    flow_range = _get_feature_array(example, "metadata/forward_flow_range")
    if (
        forward_flow.size > 0
        and flow_range.size >= 2
        and num_frames is not None
        and height is not None
        and width is not None
    ):
        try:
            decoded = decode_forward_flow(
                raw_flow=forward_flow,
                flow_range=flow_range,
                num_frames=num_frames,
                height=height,
                width=width,
            )
            score = reliability_score_from_flow(decoded)
        except Exception:
            score = math.nan

    return {
        "scenario": scenario,
        "record_index": index,
        "num_instances": num_instances if num_instances is not None else math.nan,
        "camera_translation_speed_mean": cam_speed_mean,
        "camera_rotation_change_mean": cam_rot_change_mean,
        "instance_speed_mean": inst_speed_mean,
        "reliability_score": score,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build per-example feature + reliability score table from TFRecords."
    )
    parser.add_argument("--scenario", required=True, help="Scenario folder name.")
    parser.add_argument(
        "--max_records",
        type=int,
        default=200,
        help="Maximum number of records to process (default: 200).",
    )
    parser.add_argument(
        "--dataset_root",
        default=str(DEFAULT_DATASET_ROOT),
        help=f"Dataset root (default: {DEFAULT_DATASET_ROOT}).",
    )
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    files = scenario_train_files(args.dataset_root, args.scenario)
    if not files:
        raise FileNotFoundError(
            f"No train TFRecords found for scenario '{args.scenario}' under {args.dataset_root}."
        )

    print(f"Found {len(files)} train shard(s) for scenario '{args.scenario}'.")
    dataset = build_raw_dataset(files)

    rows: list[dict[str, Any]] = []
    for idx, example in enumerate(iter_examples(dataset)):
        if idx >= args.max_records:
            break
        row = _compute_row(example, args.scenario, idx)
        rows.append(row)
        if (idx + 1) == 1 or (idx + 1) % 10 == 0 or (idx + 1) == args.max_records:
            print(f"Processed {idx + 1} records...")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"table_{args.scenario}.csv"

    fieldnames = [
        "scenario",
        "record_index",
        "num_instances",
        "camera_translation_speed_mean",
        "camera_rotation_change_mean",
        "instance_speed_mean",
        "reliability_score",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {output_path}")
    print(f"Final row count: {len(rows)}")


if __name__ == "__main__":
    main()
