# Project Report: Predicting the Reliability of Optical Flow Estimation using Motion Scenario Metadata

## 1. Problem Statement and Motivation
Optical flow quality is highly scenario-dependent. Camera motion, object count, and object dynamics can make flow estimation easy in some scenes and difficult in others.  

This project asks: **Can we predict optical-flow reliability from lightweight scenario metadata, without running a heavy neural model at inference time?**

Why this matters:
- It enables early reliability estimation before expensive downstream processing.
- It can support adaptive pipelines (for example: choose a stronger estimator only when expected error is high).
- It provides interpretable links between scene dynamics and expected estimation quality.

## 2. Dataset Description
### 2.1 Source and storage
- Dataset format: TFRecords (local only, not committed to git).
- Dataset root: `/Users/seifeddinereguige/Documents/tfds_Dataset`
- Current observed state:
  - 47 TFRecord shards
  - train split only (no validation/test split currently extracted)
  - 178 extracted train examples

Important context from advising discussion:
- Expected dataset size may be **88** (core) or **264** (full with lens effects).
- Current local dataset may therefore be outdated/incomplete.

### 2.2 TFRecord schema summary used in this project
- Video and flow fields:
  - `video` (24 frame blobs per sample)
  - `forward_flow` (flattened quantized flow)
  - `metadata/forward_flow_range` (min/max for decoding)
  - `metadata/num_frames`, `metadata/height`, `metadata/width`
- Metadata for model features:
  - `metadata/num_instances`
  - `camera/positions`
  - `camera/quaternions`
  - `instances/velocities`

## 3. Methodology
## 3.1 Feature extraction
For each example, we build a compact feature vector:
- `num_instances`
- `camera_translation_speed_mean`
- `camera_rotation_change_mean`
- `instance_speed_mean`

These are scenario-level motion descriptors intended to capture expected flow difficulty.

### 3.2 Labeling targets
The project includes two target definitions.

#### Phase 1 target: `reliability_score` (proxy label)
1. Decode quantized `forward_flow` using `metadata/forward_flow_range`.
2. Compute per-pixel flow magnitude: `sqrt(dx^2 + dy^2)`.
3. Average across pixels and frames.

Interpretation: larger motion magnitude implies likely higher challenge.

#### Phase 2 target: `epe_mean` (true estimation error)
1. Decode `video` frame bytes to images.
2. Estimate flow between frame `t` and `t+1` with OpenCV Farneback.
3. Decode ground-truth `forward_flow`.
4. Align temporal dimensions (`gt[t]` for transition `t -> t+1`).
5. Compute End-Point Error (EPE):
   - `EPE = sqrt((dx_pred - dx_gt)^2 + (dy_pred - dy_gt)^2)`
6. Average over all pixels and frame pairs to get `epe_mean`.

Interpretation: this is a direct error signal for the chosen estimator (Farneback).

### 3.3 Regression models and evaluation
Models:
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Evaluation protocol:
- 80/20 train-test split
- Metrics on test set: MAE, RMSE, R², Spearman correlation
- 5-fold cross-validation MAE on full dataset

## 4. Experiments and Results
All numbers below come from 178 rows.

### 4.1 Phase 1 (proxy target: `reliability_score`)
- `linear_regression`:  
  MAE=0.992388, RMSE=1.198795, R2=0.847000, Spearman=0.863406, CV MAE=1.007531±0.133525
- `random_forest`:  
  MAE=0.427841, RMSE=0.929308, R2=0.908056, Spearman=0.954308, CV MAE=0.688899±0.143145
- `gradient_boosting`:  
  MAE=0.421908, RMSE=0.893670, R2=0.914973, Spearman=0.957148, CV MAE=0.709794±0.183668

### 4.2 Phase 2 (true target: `epe_mean`)
- `linear_regression`:  
  MAE=1.096928, RMSE=1.397007, R2=0.837229, Spearman=0.872189, CV MAE=1.271339±0.187349
- `random_forest`:  
  MAE=0.740377, RMSE=1.307970, R2=0.857316, Spearman=0.905399, CV MAE=1.010093±0.163411
- `gradient_boosting`:  
  MAE=0.751511, RMSE=1.261456, R2=0.867284, Spearman=0.912346, CV MAE=1.082110±0.289058

### 4.3 Interpretation
- Nonlinear models outperform linear regression in both phases.
- Performance drops from proxy target to true EPE target, which is expected:
  - Proxy label is smoother and easier to model from metadata.
  - True EPE captures estimator-specific errors and is harder to predict.
- Even for true EPE, metadata remains predictive (R² around 0.86 for tree models), which is promising.

## 5. Figures
Current figures in `reports/figures/`:
- `scatter_true_vs_pred.png`
- `hist_score_by_scenario.png`
- `feature_importance.png`

These plots currently correspond to the Phase 1 reliability target workflow.

## 6. Discussion, Limitations, and Risks
### What the current results imply
- Metadata-only prediction is feasible and useful as a reliability prior.
- Tree-based regressors capture nonlinear interactions among scenario features.

### Main limitations
- Very small sample size (178 examples).
- Only train data is currently available in extracted workflow.
- Potential dataset mismatch with expected 88/264 counts suggests stale or incomplete local data.
- Phase 1 target is a proxy and may not reflect estimator-specific failure modes.
- Phase 2 target is estimator-specific to Farneback, not universally representative of all optical flow methods.

## 7. Next Steps
1. **Dataset synchronization**
   - Confirm and sync latest dataset version (core vs full with lens effects).
   - Re-run extraction and training with the complete set.
2. **Scenario-holdout evaluation**
   - Split by scenario (not random rows) to test generalization to unseen motion regimes.
3. **Richer metadata features**
   - Add feature engineering for acceleration, depth variation proxies, and temporal statistics.
4. **Stronger flow estimator baseline**
   - Replace/add Farneback with RAFT-based EPE labels for a modern reference.
5. **Phase 2 visualization support**
   - Extend plotting pipeline to support configurable targets (including `epe_mean`).

## 8. Reproducibility Notes
- TFRecords are intentionally not tracked in git.
- `outputs/` is git-ignored for generated artifacts.
- Code is designed to run from CLI with explicit dataset path and target selection.
