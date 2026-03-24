# Predicting Optical Flow Reliability from Motion Scenario Metadata

This CSI4900 project studies whether we can predict optical-flow reliability from lightweight scenario metadata instead of running a full optical-flow model at deployment time.

## Project At A Glance
- Predicts optical-flow reliability from scene metadata such as object count, camera motion, and instance velocity statistics.
- Uses three reliability definitions:
  - `reliability_score`: proxy target from ground-truth flow magnitude
  - `epe_mean`: classical reliability target from Farneback EPE vs ground truth
  - `epe_mean_raft`: modern reliability target from RAFT EPE vs ground truth
- Implements:
  - TFRecord ingestion and decoding
  - metadata feature extraction
  - CSV generation for all three targets
  - regression baselines
  - saved metrics and prediction CSVs
  - plots and media exports
  - a Streamlit demo

## Repository Layout
- `src/`: library code and CLI entrypoints
- `tools/`: inspection and sample-export utilities
- `reports/figures/`: tracked figures for the report
- `outputs/`: generated CSVs, metrics, predictions, and sample media
- `streamlit_app.py`: interactive demo

## Dataset
- Expected local dataset root: `/Users/seifeddinereguige/Documents/tfds_Dataset`
- Raw TFRecords are not committed to the repo
- Generated outputs are written to `outputs/`
- Both TFRecords and outputs are excluded from git via `.gitignore`

Each TFRecord example includes:
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

### Farneback EPE workflow
```bash
python -m src.build_all_tables_epe \
  --dataset_root "/Users/seifeddinereguige/Documents/tfds_Dataset" \
  --max_records 5000 \
  --splits train

python -m src.train_regressor \
  --csv outputs/table_all_scenarios_epe.csv \
  --target epe_mean

python -m src.plots \
  --table outputs/table_all_scenarios_epe.csv \
  --preds outputs/preds_gradient_boosting_epe_mean.csv \
  --target epe_mean \
  --out_dir reports/figures
```

### RAFT EPE workflow + demo
```bash
python -m src.build_all_tables_raft_epe \
  --dataset_root "/Users/seifeddinereguige/Documents/tfds_Dataset" \
  --max_records 10 \
  --max_pairs 3 \
  --splits train \
  --raft_model small

python -m src.train_regressor \
  --csv outputs/table_all_scenarios_raft_epe.csv \
  --target epe_mean_raft

python -m tools.export_sample_media \
  --dataset_root "/Users/seifeddinereguige/Documents/tfds_Dataset" \
  --scenario "fixed_random_rotate" \
  --record_index 0 \
  --fps 8 \
  --raft_model small \
  --pair_index 0

streamlit run streamlit_app.py
```

### Optional make targets
```bash
make build_epe
make train_epe
make build_raft_epe
make train_raft_epe
make plots
```

## Visual Docs
- [Workflow](WORKFLOW.md)
- [Roadmap](ROADMAP.md)
- [Results](RESULTS.md)

## Screenshots
- `reports/figures/streamlit_screenshot.png` is not in the repo yet.
- To add it, run the demo, capture one clean screenshot, and save it under `reports/figures/streamlit_screenshot.png`.
- Short capture instructions are available in [scripts/capture_demo_instructions.md](scripts/capture_demo_instructions.md).

## Outputs
- `outputs/table_all_scenarios.csv`
- `outputs/table_all_scenarios_epe.csv`
- `outputs/table_all_scenarios_raft_epe.csv`
- `outputs/metrics_<model>_<target>.json`
- `outputs/preds_<model>_<target>.csv`
- `outputs/cache_raft/*.json`
- `reports/figures/*.png`

## Utility Commands
Inspect one TFRecord sample:
```bash
python tools/inspect_one_record.py --scenario linear_movement_rotate_bar --record_index 0
```

Export frames, GIF, RAFT flow, and EPE heatmap:
```bash
python -m tools.export_sample_media --scenario fixed_random_rotate --record_index 0
```

## Notes
- TensorFlow is kept because TFRecord reading and frame decoding use it directly.
- `src.plots` supports configurable targets.
- The current saved experiment set is still relatively small, so the metrics should be read as promising intermediate results rather than final conclusions.
