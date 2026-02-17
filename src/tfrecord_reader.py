from __future__ import annotations

from pathlib import Path
from typing import Iterable

import tensorflow as tf


def list_scenarios(dataset_root: str | Path) -> list[str]:
    """List scenario names as immediate subfolders under dataset_root."""
    root = Path(dataset_root)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def scenario_split_files(
    dataset_root: str | Path,
    scenario: str,
    split: str,
) -> list[str]:
    """Return sorted TFRecord shards for one scenario and split."""
    root = Path(dataset_root)
    pattern = root / scenario / "*" / f"flow_dataset-{split}.tfrecord-*"
    return sorted(str(p) for p in root.glob(str(pattern.relative_to(root))))


def scenario_files_for_splits(
    dataset_root: str | Path,
    scenario: str,
    splits: Iterable[str],
) -> tuple[list[str], dict[str, list[str]]]:
    """
    Return all shard files across requested splits and split->files mapping.
    """
    mapping: dict[str, list[str]] = {}
    all_files: list[str] = []
    for split in splits:
        files = scenario_split_files(dataset_root, scenario, split)
        mapping[split] = files
        all_files.extend(files)
    return all_files, mapping


def scenario_train_files(dataset_root: str | Path, scenario: str) -> list[str]:
    """Return sorted train TFRecord shards for one scenario."""
    return scenario_split_files(dataset_root, scenario, "train")


def build_raw_dataset(
    tfrecord_files: Iterable[str],
    deterministic: bool = True,
) -> tf.data.Dataset:
    """Build a simple TFRecord dataset pipeline of serialized examples."""
    files = list(tfrecord_files)
    if not files:
        raise FileNotFoundError("No TFRecord files provided.")
    dataset = tf.data.TFRecordDataset(
        files,
        num_parallel_reads=tf.data.AUTOTUNE,
    )
    options = tf.data.Options()
    options.deterministic = deterministic
    return dataset.with_options(options)


def iter_examples(dataset: tf.data.Dataset):
    """Yield tf.train.Example objects from a serialized TFRecord dataset."""
    for raw in dataset.as_numpy_iterator():
        example = tf.train.Example()
        example.ParseFromString(raw)
        yield example
