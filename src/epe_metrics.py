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
    gt = decode_forward_flow(
        raw_flow=raw_flow,
        flow_range=flow_range,
        num_frames=num_frames,
        height=height,
        width=width,
    )
    return np.asarray(gt, dtype=np.float32)


def compute_epe_map(pred_flow: np.ndarray, gt_flow: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred_flow, dtype=np.float32)
    gt = np.asarray(gt_flow, dtype=np.float32)
    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt flow shapes must match, got {pred.shape} vs {gt.shape}.")
    if pred.ndim != 3 or pred.shape[-1] != 2:
        raise ValueError(f"Flow must have shape [H,W,2], got {pred.shape}.")
    diff = pred - gt
    return np.sqrt(np.sum(np.square(diff), axis=-1))


def compute_mean_epe(pred_flow: np.ndarray, gt_flow: np.ndarray) -> float:
    return float(np.mean(compute_epe_map(pred_flow=pred_flow, gt_flow=gt_flow)))


def compute_raft_epe(
    video_frames: np.ndarray,
    raw_flow: np.ndarray | list[int],
    flow_range: np.ndarray | list[float],
    num_frames: int,
    height: int,
    width: int,
    raft_runner,
    max_pairs: int | None = None,
) -> float:
    frames = np.asarray(video_frames, dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"video_frames must have shape [T,H,W,3], got {frames.shape}.")
    if frames.shape[0] < 2:
        raise ValueError("Need at least 2 frames to compute RAFT EPE.")

    gt_flow = decode_ground_truth_forward_flow(
        raw_flow=raw_flow,
        flow_range=flow_range,
        num_frames=num_frames,
        height=height,
        width=width,
    )

    available_pairs = min(frames.shape[0] - 1, gt_flow.shape[0] - 1)
    if max_pairs is not None:
        available_pairs = min(available_pairs, int(max_pairs))
    if available_pairs <= 0:
        raise ValueError("No valid frame pairs available for RAFT EPE.")

    epe_values: list[float] = []
    for t in range(available_pairs):
        pred_flow = raft_runner.predict_pair(frames[t], frames[t + 1])
        gt_pair = gt_flow[t]
        epe_values.append(compute_mean_epe(pred_flow=pred_flow, gt_flow=gt_pair))

    return float(np.mean(epe_values))
