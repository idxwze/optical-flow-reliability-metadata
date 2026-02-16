# Predicting Optical Flow Reliability from Motion Scenario Metadata

This project predicts the reliability of optical flow estimation using motion scenario metadata.
Goal: flag when optical flow is likely to be inaccurate, and analyze which scenarios cause failures.

## Problem
Optical flow can fail under certain conditions (fast motion, blur, occlusions, low texture, lighting changes).
We aim to predict reliability using scenario metadata.

## Approach (initial)
- Inputs: motion scenario metadata (and optionally cheap image statistics)
- Output: reliability score or class (reliable/unreliable)
- Models: baseline (logistic regression / random forest), then stronger models (e.g., XGBoost)
- Evaluation: AUC/F1 (classification) or MAE/Spearman (regression), plus per-scenario breakdown

## Repo Structure
- `src/` training + evaluation code
- `notebooks/` exploration
- `data/` placeholders (raw data not committed)
- `reports/` figures + write-up assets
- `meeting_notes/` supervisor meeting notes
- `metadata_schema.md` definition of metadata fields

## How to Run (placeholder)
Coming soon.
