# Predicting Optical Flow Reliability from Motion Scenario Metadata

This repository contains a compact, reproducible pipeline for predicting optical-flow reliability from scenario metadata extracted from TFRecord video examples.

The repo supports three targets:
- `reliability_score`: proxy difficulty label from ground-truth flow magnitude
- `epe_mean`: Farneback optical-flow error against ground truth
- `epe_mean_raft`: RAFT optical-flow error against ground truth

## Repository Layout
- `src/`: library code and CLI entrypoints
- `tools/`: lightweight inspection and sample-export utilities
- `reports/figures/`: tracked report figures
- `outputs/`: generated CSVs, metrics, predictions, and caches (git-ignored)

## Dataset
- Expected TFRecord root: `/Users/seifeddinereguige/Documents/tfds_Dataset`
- TFRecords are not committed
- Override the dataset path with `--dataset_root`

Each example contains:
- `video`
- `forward_flow`
- `metadata/forward_flow_range`
- `metadata/height`, `metadata/width`, `metadata/num_frames`
- `metadata/num_instances`
- `camera/positions`, `camera/quaternions`
- `instances/velocities`

## Environment Setup
```bash
cd /Users/seifeddinereguige/PycharmProjects/optical-flow-reliability-metadata
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Quickstart
### Farneback EPE build
```bash
python -m src.build_all_tables_epe \
  --dataset_root "/Users/seifeddinereguige/Documents/tfds_Dataset" \
  --max_records 5000 \
  --splits train
```

### Farneback EPE train
```bash
python -m src.train_regressor \
  --csv outputs/table_all_scenarios_epe.csv \
  --target epe_mean
```

### RAFT EPE build
```bash
python -m src.build_all_tables_raft_epe \
  --dataset_root "/Users/seifeddinereguige/Documents/tfds_Dataset" \
  --max_records 10 \
  --max_pairs 3 \
  --splits train \
  --raft_model small
```

### RAFT EPE train
```bash
python -m src.train_regressor \
  --csv outputs/table_all_scenarios_raft_epe.csv \
  --target epe_mean_raft
```

### Plots generation
```bash
python -m src.plots \
  --table outputs/table_all_scenarios.csv \
  --preds outputs/preds_gradient_boosting_reliability_score.csv \
  --target reliability_score \
  --out_dir reports/figures
```

### Make targets
```bash
make build_epe
make train_epe
make build_raft_epe
make train_raft_epe
make plots
```

## Outputs
- `outputs/table_all_scenarios.csv`
- `outputs/table_all_scenarios_epe.csv`
- `outputs/table_all_scenarios_raft_epe.csv`
- `outputs/metrics_<model>_<target>.json`
- `outputs/preds_<model>_<target>.csv`
- `outputs/cache_raft/*.json`
- `reports/figures/*.png`

## Utility Tools
Inspect one sample:
```bash
python tools/inspect_one_record.py --scenario linear_movement_rotate_bar --record_index 0
```

Export sample frames and GIF:
```bash
python tools/export_sample_media.py --scenario linear_movement_slide --record_index 0
```

## Notes
- TensorFlow is kept because TFRecord reading and frame decoding use it directly.
- `src.plots` supports configurable targets, but the tracked report figures currently reflect the proxy-label workflow.
- No Streamlit app is present in this repo at the moment.
