from __future__ import annotations

import numpy as np


def reshape_forward_flow(
    raw_flow: np.ndarray | list[int],
    num_frames: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Reshape flat forward flow into [T, H, W, 2]."""
    raw = np.asarray(raw_flow, dtype=np.float32)
    expected = int(num_frames) * int(height) * int(width) * 2
    if raw.size != expected:
        raise ValueError(
            f"forward_flow size mismatch: got {raw.size}, expected {expected} "
            f"for shape ({num_frames}, {height}, {width}, 2)."
        )
    return raw.reshape((int(num_frames), int(height), int(width), 2))


def decode_forward_flow(
    raw_flow: np.ndarray | list[int],
    flow_range: np.ndarray | list[float],
    num_frames: int,
    height: int,
    width: int,
) -> np.ndarray:
    """
    Decode quantized forward flow to float32:
    flow = flow_min + (raw/65535.0) * (flow_max - flow_min)
    """
    flow = reshape_forward_flow(raw_flow, num_frames, height, width)
    range_arr = np.asarray(flow_range, dtype=np.float32).reshape(-1)
    if range_arr.size < 2:
        raise ValueError("metadata/forward_flow_range must have 2 values [min, max].")
    flow_min, flow_max = float(range_arr[0]), float(range_arr[1])
    return flow_min + (flow / 65535.0) * (flow_max - flow_min)
