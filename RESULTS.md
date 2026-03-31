# Results

This page summarizes the saved experiment outputs currently present in `outputs/metrics_*.json` and the latest repeated scenario-holdout runs captured from the terminal. The tables below use the exact values reported by the repository.

Current saved runs are based on `n_rows = 178`.

## Target: `reliability_score`

| model | MAE_test | RMSE_test | R2_test | Spearman_test | CV_MAE_mean | CV_MAE_std |
|---|---:|---:|---:|---:|---:|---:|
| gradient_boosting | 0.4219 | 0.8937 | 0.9150 | 0.9571 | 0.7098 | 0.1837 |
| linear_regression | 0.9924 | 1.1988 | 0.8470 | 0.8634 | 1.0075 | 0.1335 |
| random_forest | 0.4278 | 0.9293 | 0.9081 | 0.9543 | 0.6889 | 0.1431 |

## Target: `epe_mean`

| model | MAE_test | RMSE_test | R2_test | Spearman_test | CV_MAE_mean | CV_MAE_std |
|---|---:|---:|---:|---:|---:|---:|
| gradient_boosting | 0.7515 | 1.2615 | 0.8673 | 0.9123 | 1.0821 | 0.2891 |
| linear_regression | 1.0969 | 1.3970 | 0.8372 | 0.8722 | 1.2713 | 0.1873 |
| random_forest | 0.7404 | 1.3080 | 0.8573 | 0.9054 | 1.0101 | 0.1634 |

## Target: `epe_mean_raft`

| model | MAE_test | RMSE_test | R2_test | Spearman_test | CV_MAE_mean | CV_MAE_std |
|---|---:|---:|---:|---:|---:|---:|
| gradient_boosting | 0.8233 | 1.7038 | 0.7042 | 0.9514 | 1.2131 | 0.3247 |
| linear_regression | 1.4959 | 1.9012 | 0.6317 | 0.7839 | 1.5696 | 0.1978 |
| random_forest | 0.7602 | 1.4810 | 0.7765 | 0.9515 | 1.0666 | 0.1586 |

## Latest Repeated Scenario-Holdout Results

### Experimental Setup

The latest evaluation used repeated scenario-holdout splits rather than a random row split. For each target, we trained on 80% of unique scenarios and tested on the remaining 20%, repeated 10 times with seeds `42` through `51`. Both runs used `178` rows. The Farneback experiment used `outputs/table_all_scenarios_epe.csv` with target `epe_mean` and features `num_instances`, `camera_translation_speed_mean`, `camera_rotation_change_mean`, and `instance_speed_mean`. The RAFT experiment used `outputs/table_all_scenarios_raft_epe.csv` with target `epe_mean_raft` and features `num_instances`, `camera_translation_speed_mean`, `camera_rotation_change_mean`, `instance_speed_mean`, and `visibility_mean`. The repeated summaries were saved to `outputs/metrics_summary_epe_mean_scenario.json` and `outputs/metrics_summary_epe_mean_raft_scenario.json`.

### Target: `epe_mean` from `outputs/table_all_scenarios_epe.csv`

| model | MAE_test | RMSE_test | R2_test | Spearman_test |
|---|---:|---:|---:|---:|
| linear_regression | 1.2041 ± 0.2815 | 1.6619 ± 0.4938 | 0.7605 ± 0.1088 | 0.8316 ± 0.0623 |
| random_forest | 0.9517 ± 0.3170 | 1.5117 ± 0.5459 | 0.7987 ± 0.1147 | 0.8466 ± 0.0604 |
| gradient_boosting | 1.0259 ± 0.3248 | 1.7910 ± 0.5024 | 0.7251 ± 0.1163 | 0.8419 ± 0.0513 |

### Target: `epe_mean_raft` from `outputs/table_all_scenarios_raft_epe.csv`

| model | MAE_test | RMSE_test | R2_test | Spearman_test |
|---|---:|---:|---:|---:|
| linear_regression | 1.4870 ± 0.4315 | 2.1255 ± 0.7189 | 0.6036 ± 0.1676 | 0.7673 ± 0.0725 |
| random_forest | 1.0979 ± 0.3953 | 1.9095 ± 0.6451 | 0.6935 ± 0.1057 | 0.8505 ± 0.0599 |
| gradient_boosting | 1.2435 ± 0.3568 | 2.1913 ± 0.5615 | 0.5790 ± 0.1354 | 0.7824 ± 0.1126 |

## Key Takeaways
- The metadata features are strongly predictive for the proxy target `reliability_score`, with tree models reaching about `R² ≈ 0.91`.
- Predicting true estimator error is harder than predicting the proxy target, which is expected because EPE depends on estimator-specific failures, not just motion magnitude.
- Farneback EPE remains quite learnable from metadata, while RAFT EPE is noticeably harder in terms of `R²`, even though the ranking correlation stays strong for the tree models.
- Linear regression is a useful baseline, but the nonlinear models are consistently stronger across all three targets.
- The main caution is dataset size: the current saved runs use 178 rows, so conclusions should be treated as promising but still preliminary.
- A dataset refresh is expected and should improve confidence in final comparisons.

In the repeated scenario-holdout setting, Random Forest is the strongest baseline for both `epe_mean` and `epe_mean_raft`. Farneback error generalizes better than RAFT error under unseen-scenario evaluation: the Farneback target reaches about `R² = 0.7987 ± 0.1147`, while the RAFT target drops to about `R² = 0.6935 ± 0.1057`. That matches the intuition that RAFT EPE is a harder target to predict from metadata alone, even though it is still meaningfully structured.

## Figures & Media Gallery

### Figures
- [Scatter: true vs predicted](reports/figures/scatter_true_vs_pred.png)
- [Distribution by scenario](reports/figures/hist_score_by_scenario.png)
- [Feature importance](reports/figures/feature_importance.png)

### Sample Media
- [Sample media folder: fixed_random_rotate / record_00000](outputs/sample_media/fixed_random_rotate/record_00000)
- [Preview GIF](outputs/sample_media/fixed_random_rotate/record_00000/preview.gif)
- [RAFT flow visualization](outputs/sample_media/fixed_random_rotate/record_00000/flow_raft.png)
- [RAFT EPE heatmap](outputs/sample_media/fixed_random_rotate/record_00000/epe_raft.png)

Note: media under `outputs/` is generated locally and git-ignored, so these links are mainly for local browsing inside the project workspace.

## Demo

Run the demo with:

```bash
streamlit run streamlit_app.py
```

The demo lets you:
- choose a dataset root and scenario
- select a record index based on the saved RAFT table
- generate or reuse sample media for a record
- inspect the preview GIF, RAFT flow visualization, and RAFT EPE heatmap
- view the metadata features and saved `epe_mean_raft` value for that sample
