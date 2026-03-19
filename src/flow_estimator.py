from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


def _decode_frame(frame_bytes: bytes, height: int | None, width: int | None) -> np.ndarray:
    arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is not None:
        return bgr

    if height and width and arr.size == int(height) * int(width) * 3:
        return arr.reshape(int(height), int(width), 3)

    raise ValueError("Failed to decode video frame bytes with OpenCV.")


def decode_video_frames(
    video_bytes: Iterable[bytes],
    height: int | None = None,
    width: int | None = None,
    num_frames: int | None = None,
) -> np.ndarray:
    """
    Decode per-frame bytes into [T, H, W, 3] uint8.
    """
    frame_list = list(video_bytes)
    if num_frames is not None:
        frame_list = frame_list[: int(num_frames)]
    if not frame_list:
        raise ValueError("Video feature is empty.")

    frames: list[np.ndarray] = []
    for frame in frame_list:
        frames.append(_decode_frame(frame, height, width))

    first_shape = frames[0].shape
    for idx, frame in enumerate(frames):
        if frame.shape != first_shape:
            raise ValueError(
                f"Inconsistent frame shape at index {idx}: {frame.shape} vs {first_shape}."
            )

    return np.stack(frames, axis=0).astype(np.uint8, copy=False)


def estimate_farneback_flow(video_frames: np.ndarray) -> np.ndarray:
    """
    Estimate dense forward optical flow between t and t+1.
    Input: [T, H, W, 3] uint8
    Output: [T-1, H, W, 2] float32
    """
    frames = np.asarray(video_frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"video_frames must have shape [T,H,W,3], got {frames.shape}.")
    if frames.shape[0] < 2:
        raise ValueError("Need at least 2 frames to estimate flow.")

    gray = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    flows: list[np.ndarray] = []
    for t in range(len(gray) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            gray[t],
            gray[t + 1],
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        flows.append(flow.astype(np.float32, copy=False))
    return np.stack(flows, axis=0)

