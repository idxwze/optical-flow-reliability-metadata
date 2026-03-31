from __future__ import annotations

import math
from typing import Any

import numpy as np


FEATURE_COLUMNS = [
    "num_instances",
    "camera_translation_speed_mean",
    "camera_rotation_change_mean",
    "instance_speed_mean",
    "visibility_mean",
]


def feature_to_numpy_list(feature: Any) -> np.ndarray:
    if feature is None:
        return np.array([])
    if feature.int64_list.value:
        return np.asarray(feature.int64_list.value)
    if feature.float_list.value:
        return np.asarray(feature.float_list.value)
    if feature.bytes_list.value:
        return np.asarray(feature.bytes_list.value, dtype=object)
    return np.array([])


def get_feature_array(example, key: str) -> np.ndarray:
    feats = example.features.feature
    if key not in feats:
        return np.array([])
    return feature_to_numpy_list(feats[key])


def get_first_available_feature_array(example, keys: list[str]) -> np.ndarray:
    for key in keys:
        arr = get_feature_array(example, key)
        if arr.size > 0:
            return arr
    return np.array([])


def get_scalar_int(example, key: str) -> int | None:
    arr = get_feature_array(example, key)
    if arr.size == 0:
        return None
    try:
        return int(arr.reshape(-1)[0])
    except Exception:
        return None


def camera_translation_speed_mean(positions: np.ndarray, num_frames: int | None) -> float:
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


def camera_rotation_change_mean(quaternions: np.ndarray, num_frames: int | None) -> float:
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
    dots = np.abs(np.sum(q[:-1] * q[1:], axis=-1))
    dots = np.clip(dots, -1.0, 1.0)
    angles = 2.0 * np.arccos(dots)
    return float(np.mean(angles))


def instance_speed_mean(
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


def _debug_log(debug: bool, message: str) -> None:
    if debug:
        print(f"[example_features] {message}")


def _safe_optional_feature(name: str, fn, debug: bool) -> float:
    try:
        value = fn()
        if value is None:
            return math.nan
        return float(value)
    except Exception as exc:
        _debug_log(debug, f"{name} failed: {exc}")
        return math.nan


def _reshape_visibility(
    visibility: np.ndarray,
    num_instances: int | None,
    num_frames: int | None,
) -> np.ndarray | None:
    if visibility.size == 0:
        return None

    arr = np.asarray(visibility, dtype=np.float32)
    if arr.ndim == 2:
        return arr

    flat = arr.reshape(-1)

    if num_frames and num_instances and flat.size == num_frames * num_instances:
        return flat.reshape(num_frames, num_instances)

    if num_frames and flat.size % num_frames == 0:
        return flat.reshape(num_frames, -1)

    if num_instances and flat.size % num_instances == 0:
        return flat.reshape(-1, num_instances)

    return flat.reshape(1, -1)


def visibility_mean(
    visibility: np.ndarray,
    num_instances: int | None,
    num_frames: int | None,
) -> float:
    vis = _reshape_visibility(visibility, num_instances, num_frames)
    if vis is None:
        return math.nan
    return float(np.mean(vis))


def compute_metadata_features(example, debug: bool = False) -> dict[str, float | int]:
    num_frames = get_scalar_int(example, "metadata/num_frames")
    num_instances = get_scalar_int(example, "metadata/num_instances")

    positions = get_feature_array(example, "camera/positions")
    quaternions = get_feature_array(example, "camera/quaternions")
    velocities = get_feature_array(example, "instances/velocities")

    visibility = get_first_available_feature_array(
        example,
        ["instances/visibility", "visibility"],
    )

    return {
        "num_instances": num_instances if num_instances is not None else math.nan,
        "camera_translation_speed_mean": camera_translation_speed_mean(positions, num_frames),
        "camera_rotation_change_mean": camera_rotation_change_mean(quaternions, num_frames),
        "instance_speed_mean": instance_speed_mean(velocities, num_instances, num_frames),
        "visibility_mean": _safe_optional_feature(
            "visibility_mean",
            lambda: visibility_mean(visibility, num_instances, num_frames),
            debug,
        ),
    }
