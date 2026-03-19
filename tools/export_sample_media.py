from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio

from src.example_features import get_feature_array, get_scalar_int
from src.tfrecord_reader import build_raw_dataset, iter_examples, scenario_files_for_splits
from src.video_decode import decode_video_frames


DEFAULT_DATASET_ROOT = Path("/Users/seifeddinereguige/Documents/tfds_Dataset")


def main():
    parser = argparse.ArgumentParser(
        description="Export decoded frames and an animated GIF for one TFRecord example."
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


if __name__ == "__main__":
    main()
