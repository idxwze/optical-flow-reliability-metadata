from __future__ import annotations

import numpy as np

from src.decode_flow import decode_forward_flow


def decode_ground_truth_forward_flow(
    raw_flow: np.ndarray | list[int],
    flow_range: np.ndarray | list[float],
    num_frames: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Decode forward_flow into [T, H, W, 2] float32."""
    gt = decode_forward_flow(
        raw_flow=raw_flow,
        flow_range=flow_range,
        num_frames=num_frames,
        height=height,
        width=width,
    )
    return np.asarray(gt, dtype=np.float32)


def mean_epe(pred_flow_tminus1: np.ndarray, gt_flow_t: np.ndarray) -> float:
    """
    pred_flow_tminus1: [T-1, H, W, 2]
    gt_flow_t: [T, H, W, 2] (uses gt[t] for t->t+1 alignment)
    """
    pred = np.asarray(pred_flow_tminus1, dtype=np.float32)
    gt = np.asarray(gt_flow_t, dtype=np.float32)

    if pred.ndim != 4 or pred.shape[-1] != 2:
        raise ValueError(f"pred flow must be [T-1,H,W,2], got {pred.shape}.")
    if gt.ndim != 4 or gt.shape[-1] != 2:
        raise ValueError(f"gt flow must be [T,H,W,2], got {gt.shape}.")
    if gt.shape[0] < 2:
        raise ValueError("gt flow must have at least 2 frames.")

    gt_aligned = gt[:-1]
    if pred.shape != gt_aligned.shape:
        raise ValueError(
            f"shape mismatch after alignment: pred {pred.shape} vs gt {gt_aligned.shape}."
        )

    diff = pred - gt_aligned
    epe = np.sqrt(np.sum(np.square(diff), axis=-1))
    return float(np.mean(epe))

