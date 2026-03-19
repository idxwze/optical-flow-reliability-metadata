from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


def _lazy_import_torch():
    try:
        import torch
        import torch.nn.functional as torch_f
        from torchvision.models.optical_flow import (
            Raft_Large_Weights,
            Raft_Small_Weights,
            raft_large,
            raft_small,
        )
    except ImportError as exc:
        raise ImportError(
            "RAFT inference requires torch and torchvision. "
            "Install them with `pip install -r requirements.txt`."
        ) from exc

    return torch, torch_f, raft_small, raft_large, Raft_Small_Weights, Raft_Large_Weights


def resolve_device(device: str | None = None) -> str:
    torch, _, _, _, _, _ = _lazy_import_torch()
    if device:
        return device
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_model(raft_model: str, progress: bool):
    torch, _, raft_small, raft_large, Raft_Small_Weights, Raft_Large_Weights = _lazy_import_torch()

    raft_model = raft_model.lower()
    if raft_model == "small":
        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights, progress=progress)
    elif raft_model == "large":
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights, progress=progress)
    else:
        raise ValueError(f"Unsupported raft_model '{raft_model}'. Use 'small' or 'large'.")

    return model.eval(), weights, torch


def _pad_to_multiple_of_8(img1, img2, torch_f):
    _, _, height, width = img1.shape
    pad_h = (8 - (height % 8)) % 8
    pad_w = (8 - (width % 8)) % 8
    if pad_h == 0 and pad_w == 0:
        return img1, img2, 0, 0
    pad = (0, pad_w, 0, pad_h)
    return (
        torch_f.pad(img1, pad=pad, mode="replicate"),
        torch_f.pad(img2, pad=pad, mode="replicate"),
        pad_h,
        pad_w,
    )


def _prepare_pair(frame1: np.ndarray, frame2: np.ndarray, device: str, weights):
    torch, torch_f, _, _, _, _ = _lazy_import_torch()

    first = np.asarray(frame1, dtype=np.uint8)
    second = np.asarray(frame2, dtype=np.uint8)
    if first.shape != second.shape:
        raise ValueError(f"Frame shapes must match, got {first.shape} vs {second.shape}.")
    if first.ndim != 3 or first.shape[-1] != 3:
        raise ValueError(f"Frames must have shape [H,W,3], got {first.shape}.")

    img1 = torch.from_numpy(first).permute(2, 0, 1).unsqueeze(0)
    img2 = torch.from_numpy(second).permute(2, 0, 1).unsqueeze(0)

    transforms = weights.transforms()
    img1, img2 = transforms(img1, img2)
    img1, img2, pad_h, pad_w = _pad_to_multiple_of_8(img1, img2, torch_f)
    return img1.to(device), img2.to(device), pad_h, pad_w, torch


@dataclass
class RaftInferenceRunner:
    model: object
    weights: object
    device: str
    raft_model: str

    def predict_pair(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        img1, img2, pad_h, pad_w, torch = _prepare_pair(
            frame1=frame1,
            frame2=frame2,
            device=self.device,
            weights=self.weights,
        )

        with torch.inference_mode():
            predicted_flows = self.model(img1, img2)

        flow = predicted_flows[-1][0]
        if pad_h:
            flow = flow[:, :-pad_h, :]
        if pad_w:
            flow = flow[:, :, :-pad_w]

        return flow.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32, copy=False)


def load_raft_runner(
    raft_model: str = "small",
    device: str | None = None,
    progress: bool = True,
) -> RaftInferenceRunner:
    model, weights, torch = _build_model(raft_model=raft_model, progress=progress)
    resolved_device = resolve_device(device)
    model = model.to(resolved_device)
    if resolved_device != "cpu" and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    return RaftInferenceRunner(
        model=model,
        weights=weights,
        device=resolved_device,
        raft_model=raft_model.lower(),
    )


def predict_raft_flow_pair(
    frame1: np.ndarray,
    frame2: np.ndarray,
    raft_model: str = "small",
    device: str | None = None,
) -> np.ndarray:
    runner = load_raft_runner(raft_model=raft_model, device=device)
    return runner.predict_pair(frame1=frame1, frame2=frame2)


def main():
    parser = argparse.ArgumentParser(description="Run torchvision RAFT on two image files.")
    parser.add_argument("frame1", help="Path to first image.")
    parser.add_argument("frame2", help="Path to second image.")
    parser.add_argument(
        "--raft_model",
        default="small",
        choices=["small", "large"],
        help="RAFT model variant (default: small).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override, e.g. cpu or mps.",
    )
    args = parser.parse_args()

    from PIL import Image

    frame1 = np.asarray(Image.open(args.frame1).convert("RGB"))
    frame2 = np.asarray(Image.open(args.frame2).convert("RGB"))
    flow = predict_raft_flow_pair(
        frame1=frame1,
        frame2=frame2,
        raft_model=args.raft_model,
        device=args.device,
    )
    print(flow.shape)


if __name__ == "__main__":
    main()
