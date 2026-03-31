from __future__ import annotations

from typing import Iterable

import numpy as np
import tensorflow as tf


def _decode_one_frame(
    frame_bytes: bytes,
    height: int | None,
    width: int | None,
    channels: int | None = 3,
) -> np.ndarray:
    try:
        frame = tf.io.decode_image(frame_bytes, channels=channels, expand_animations=False)
        decoded = frame.numpy()
    except Exception as exc:
        arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        if channels is None:
            expected_sizes: list[tuple[int, tuple[int, ...]]] = []
        else:
            expected_sizes = [
                (int(height) * int(width) * int(channels), (int(height), int(width), int(channels)))
            ]
            if channels == 1:
                expected_sizes.append((int(height) * int(width), (int(height), int(width))))
        if height and width and any(arr.size == expected for expected, _ in expected_sizes):
            for expected, shape in expected_sizes:
                if arr.size == expected:
                    decoded = arr.reshape(shape)
                    break
        else:
            raise ValueError(f"TensorFlow failed to decode frame bytes: {exc}") from exc

    if channels == 3:
        if decoded.ndim != 3 or decoded.shape[-1] != 3:
            raise ValueError(f"Decoded frame must have shape [H,W,3], got {decoded.shape}.")
    elif channels == 1:
        if decoded.ndim == 3 and decoded.shape[-1] == 1:
            decoded = decoded[..., 0]
        elif decoded.ndim != 2:
            raise ValueError(f"Decoded single-channel frame must have shape [H,W], got {decoded.shape}.")

    return np.asarray(decoded, dtype=np.uint8)


def decode_video_frames(
    video_bytes: Iterable[bytes],
    height: int | None = None,
    width: int | None = None,
    num_frames: int | None = None,
) -> np.ndarray:
    """
    Decode TFRecord video bytes into [T, H, W, 3] uint8 RGB frames.
    """
    frame_list = list(video_bytes)
    if num_frames is not None:
        frame_list = frame_list[: int(num_frames)]
    if not frame_list:
        raise ValueError("Video feature is empty.")

    frames = [_decode_one_frame(frame, height=height, width=width) for frame in frame_list]
    first_shape = frames[0].shape
    for idx, frame in enumerate(frames):
        if frame.shape != first_shape:
            raise ValueError(
                f"Inconsistent frame shape at index {idx}: {frame.shape} vs {first_shape}."
            )

    return np.stack(frames, axis=0).astype(np.uint8, copy=False)


def decode_dense_frames(
    frame_bytes: Iterable[bytes],
    height: int | None = None,
    width: int | None = None,
    num_frames: int | None = None,
    channels: int | None = 1,
) -> np.ndarray:
    """
    Decode per-frame bytes into [T, H, W] or [T, H, W, C] uint8 frames.
    Useful for segmentation and depth frames stored as encoded images.
    """
    frame_list = list(frame_bytes)
    if num_frames is not None:
        frame_list = frame_list[: int(num_frames)]
    if not frame_list:
        raise ValueError("Dense frame feature is empty.")

    frames = [
        _decode_one_frame(frame, height=height, width=width, channels=channels)
        for frame in frame_list
    ]
    first_shape = frames[0].shape
    for idx, frame in enumerate(frames):
        if frame.shape != first_shape:
            raise ValueError(
                f"Inconsistent frame shape at index {idx}: {frame.shape} vs {first_shape}."
            )

    return np.stack(frames, axis=0).astype(np.uint8, copy=False)
