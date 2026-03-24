from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from src.epe_metrics import compute_epe_map, decode_ground_truth_forward_flow
from src.example_features import get_feature_array, get_scalar_int
from src.raft_infer import load_raft_runner
from src.tfrecord_reader import build_raw_dataset, iter_examples, scenario_files_for_splits
from src.video_decode import decode_video_frames
from src.viz_flow import epe_to_heatmap, flow_to_rgb_hsv


DEFAULT_DATASET_ROOT = Path("/Users/seifeddinereguige/Documents/tfds_Dataset")


def _export_raft_visuals(
    *,
    frames: np.ndarray,
    target_example,
    num_frames: int,
    height: int,
    width: int,
    pair_index: int,
    raft_model: str,
    output_root: Path,
) -> None:
    if pair_index < 0:
        raise ValueError(f"pair_index must be non-negative, got {pair_index}.")
    if pair_index >= frames.shape[0] - 1:
        raise IndexError(
            f"pair_index {pair_index} is out of range for {frames.shape[0]} frame(s)."
        )

    forward_flow = get_feature_array(target_example, "forward_flow")
    flow_range = get_feature_array(target_example, "metadata/forward_flow_range")
    if forward_flow.size == 0 or flow_range.size < 2:
        raise ValueError("Missing forward_flow or metadata/forward_flow_range.")

    gt_flow = decode_ground_truth_forward_flow(
        raw_flow=forward_flow,
        flow_range=flow_range,
        num_frames=num_frames,
        height=height,
        width=width,
    )
    if pair_index >= gt_flow.shape[0] - 1:
        raise IndexError(
            f"pair_index {pair_index} is out of range for decoded ground-truth flow with "
            f"{gt_flow.shape[0]} frame(s)."
        )

    raft_runner = load_raft_runner(raft_model=raft_model, device=None, progress=True)
    pred_flow = raft_runner.predict_pair(frames[pair_index], frames[pair_index + 1])
    gt_pair = gt_flow[pair_index]
    epe_map = compute_epe_map(pred_flow=pred_flow, gt_flow=gt_pair)

    flow_vis = flow_to_rgb_hsv(pred_flow)
    epe_vis = epe_to_heatmap(epe_map)

    flow_path = output_root / "flow_raft.png"
    epe_path = output_root / "epe_raft.png"
    imageio.imwrite(flow_path, flow_vis)
    imageio.imwrite(epe_path, epe_vis)

    print(
        f"Saved RAFT flow visualization for pair {pair_index}->{pair_index + 1}: {flow_path}"
    )
    print(
        f"Saved RAFT EPE heatmap for pair {pair_index}->{pair_index + 1}: {epe_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Export decoded frames, a GIF preview, and optional RAFT visualizations for one TFRecord example."
    )
    parser.add_argument("--scenario", required=True, help="Scenario folder name.")
    parser.add_argument(
        "--record_index",
        type=int,
        default=0,
        help="Zero-based record index within the selected split(s) (default: 0).",
    )
    parser.add_argument(
        "--dataset_root",
        default=str(DEFAULT_DATASET_ROOT),
        help=f"Dataset root containing scenario folders (default: {DEFAULT_DATASET_ROOT}).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        help="Split(s) to search, in order (default: train).",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/sample_media",
        help="Directory for exported media (default: outputs/sample_media).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=8.0,
        help="GIF frame rate in frames per second (default: 8).",
    )
    parser.add_argument(
        "--raft_model",
        default="small",
        choices=["small", "large"],
        help="RAFT model variant for visualization export (default: small).",
    )
    parser.add_argument(
        "--pair_index",
        type=int,
        default=0,
        help="Frame pair index t for visualizing t -> t+1 (default: 0).",
    )
    args = parser.parse_args()

    files, _ = scenario_files_for_splits(args.dataset_root, args.scenario, args.splits)
    if not files:
        raise FileNotFoundError(
            f"No TFRecord shards found for scenario '{args.scenario}' in splits {args.splits}."
        )

    dataset = build_raw_dataset(files)
    target_example = None
    for idx, example in enumerate(iter_examples(dataset)):
        if idx == args.record_index:
            target_example = example
            break
    if target_example is None:
        raise IndexError(f"Record index {args.record_index} is out of range.")

    num_frames = get_scalar_int(target_example, "metadata/num_frames")
    height = get_scalar_int(target_example, "metadata/height")
    width = get_scalar_int(target_example, "metadata/width")
    if num_frames is None or height is None or width is None:
        raise ValueError("Missing one of metadata/num_frames,height,width.")

    video = get_feature_array(target_example, "video")
    if video.size == 0:
        raise ValueError("Missing video feature.")

    frames = decode_video_frames(
        video_bytes=[bytes(x) for x in video.reshape(-1)],
        height=height,
        width=width,
        num_frames=num_frames,
    )

    output_root = Path(args.output_dir) / args.scenario / f"record_{args.record_index:05d}"
    frames_dir = output_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for idx, frame in enumerate(frames):
        imageio.imwrite(frames_dir / f"frame_{idx:03d}.png", frame)

    gif_path = output_root / "preview.gif"
    imageio.mimsave(gif_path, list(frames), duration=1.0 / args.fps)

    print(f"Saved {len(frames)} frame(s) to {frames_dir}")
    print(f"Saved GIF: {gif_path}")

    try:
        _export_raft_visuals(
            frames=frames,
            target_example=target_example,
            num_frames=num_frames,
            height=height,
            width=width,
            pair_index=args.pair_index,
            raft_model=args.raft_model,
            output_root=output_root,
        )
    except Exception as exc:
        print(f"Warning: failed to export RAFT flow/EPE visualizations: {exc}")


if __name__ == "__main__":
    main()
