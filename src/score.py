from __future__ import annotations

import numpy as np


def reliability_score_from_flow(decoded_flow: np.ndarray) -> float:
    """
    Continuous risk/difficulty score:
    mean over all frames/pixels of flow magnitude sqrt(dx^2 + dy^2).
    """
    flow = np.asarray(decoded_flow, dtype=np.float32)
    if flow.ndim != 4 or flow.shape[-1] != 2:
        raise ValueError(f"decoded_flow must have shape [T,H,W,2], got {flow.shape}.")
    mag = np.sqrt(np.sum(np.square(flow), axis=-1))
    return float(np.mean(mag))
