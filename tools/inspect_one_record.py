from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.decode_flow import decode_forward_flow
from src.example_features import compute_metadata_features, get_feature_array, get_scalar_int
from src.tfrecord_reader import build_raw_dataset, iter_examples, scenario_files_for_splits


DEFAULT_DATASET_ROOT = Path("/Users/seifeddinereguige/Documents/tfds_Dataset")


def _summarize_array(name: str, arr: np.ndarray) -> dict[str, object]:
    array = np.asarray(arr)
    summary: dict[str, object] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }
    if array.size > 0 and np.issubdtype(array.dtype, np.number):
        summary["min"] = float(np.min(array))
        summary["max"] = float(np.max(array))
    return {name: summary}


def main():
    parser = argparse.ArgumentParser(
        description="Inspect one TFRecord example and print metadata, feature summaries, and flow shape."
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
    video = get_feature_array(target_example, "video")
    forward_flow = get_feature_array(target_example, "forward_flow")
    flow_range = get_feature_array(target_example, "metadata/forward_flow_range")

    payload: dict[str, object] = {
        "scenario": args.scenario,
        "record_index": args.record_index,
        "splits": args.splits,
        "metadata": {
            "num_frames": num_frames,
            "height": height,
            "width": width,
            **compute_metadata_features(target_example),
        },
        "features_present": sorted(target_example.features.feature.keys()),
        **_summarize_array("video", video),
        **_summarize_array("forward_flow_raw", forward_flow),
        **_summarize_array("forward_flow_range", flow_range),
    }

    if (
        num_frames is not None
        and height is not None
        and width is not None
        and forward_flow.size > 0
        and flow_range.size >= 2
    ):
        decoded_flow = decode_forward_flow(
            raw_flow=forward_flow,
            flow_range=flow_range,
            num_frames=num_frames,
            height=height,
            width=width,
        )
        payload.update(_summarize_array("forward_flow_decoded", decoded_flow))

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
