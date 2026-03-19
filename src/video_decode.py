from __future__ import annotations

from typing import Iterable

import numpy as np
import tensorflow as tf


def _decode_one_frame(frame_bytes: bytes, height: int | None, width: int | None) -> np.ndarray:
    try:
        frame = tf.io.decode_image(frame_bytes, channels=3, expand_animations=False)
        decoded = frame.numpy()
    except Exception as exc:
        arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        if height and width and arr.size == int(height) * int(width) * 3:
            decoded = arr.reshape(int(height), int(width), 3)
        else:
            raise ValueError(f"TensorFlow failed to decode frame bytes: {exc}") from exc

    if decoded.ndim != 3 or decoded.shape[-1] != 3:
        raise ValueError(f"Decoded frame must have shape [H,W,3], got {decoded.shape}.")

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
