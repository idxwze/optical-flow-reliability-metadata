from __future__ import annotations

import numpy as np
from matplotlib import cm
from matplotlib.colors import hsv_to_rgb


def flow_to_rgb_hsv(flow: np.ndarray) -> np.ndarray:
    """
    Convert dense flow [H, W, 2] to an RGB visualization using HSV encoding.
    Hue encodes direction, value encodes normalized magnitude, and saturation is 1.
    """
    flow_arr = np.asarray(flow, dtype=np.float32)
    if flow_arr.ndim != 3 or flow_arr.shape[-1] != 2:
        raise ValueError(f"flow must have shape [H,W,2], got {flow_arr.shape}.")

    u = flow_arr[..., 0]
    v = flow_arr[..., 1]
    angle = np.arctan2(v, u)
    magnitude = np.sqrt(np.square(u) + np.square(v))

    robust_max = float(np.percentile(magnitude, 99))
    if not np.isfinite(robust_max) or robust_max <= 0.0:
        robust_max = float(np.max(magnitude)) if magnitude.size else 1.0
    robust_max = max(robust_max, 1e-6)

    hue = (angle + np.pi) / (2.0 * np.pi)
    saturation = np.ones_like(hue, dtype=np.float32)
    value = np.clip(magnitude / robust_max, 0.0, 1.0)

    hsv = np.stack([hue, saturation, value], axis=-1)
    rgb = hsv_to_rgb(hsv)
    return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)


def epe_to_heatmap(epe: np.ndarray) -> np.ndarray:
    """
    Convert an EPE map [H, W] to an RGB heatmap using a robust percentile scale.
    """
    epe_arr = np.asarray(epe, dtype=np.float32)
    if epe_arr.ndim != 2:
        raise ValueError(f"epe must have shape [H,W], got {epe_arr.shape}.")

    robust_max = float(np.percentile(epe_arr, 99))
    if not np.isfinite(robust_max) or robust_max <= 0.0:
        robust_max = float(np.max(epe_arr)) if epe_arr.size else 1.0
    robust_max = max(robust_max, 1e-6)

    normalized = np.clip(epe_arr / robust_max, 0.0, 1.0)
    heatmap = cm.get_cmap("inferno")(normalized)[..., :3]
    return np.clip(heatmap * 255.0, 0.0, 255.0).astype(np.uint8)
