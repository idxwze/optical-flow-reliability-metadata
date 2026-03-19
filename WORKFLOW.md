# Workflow

This page is the quick project map for the CSI4900 optical-flow reliability repo. The goal is to predict how hard optical flow estimation will be from lightweight motion metadata, without running a heavy estimator at deployment time.

## End-to-End Pipeline
```mermaid
flowchart TD
    A["TFRecord dataset<br/>video + forward_flow + metadata"] --> B["Decode video frames"]
    A --> C["Decode ground-truth flow<br/>using metadata/forward_flow_range"]
    A --> D["Extract metadata features<br/>num_instances<br/>camera translation speed<br/>camera rotation change<br/>instance speed"]

    C --> E["Phase 1 target<br/>reliability_score<br/>mean GT flow magnitude"]
    B --> F["Farneback flow estimate"]
    B --> G["RAFT flow estimate"]

    F --> H["Phase 2 target<br/>epe_mean"]
    C --> H
    G --> I["Phase 3 target<br/>epe_mean_raft"]
    C --> I

    D --> J["Train regressors<br/>Linear Regression<br/>Random Forest<br/>Gradient Boosting"]
    E --> J
    H --> J
    I --> J

    J --> K["Evaluation metrics<br/>MAE RMSE R² Spearman<br/>5-fold CV MAE"]
    J --> L["Prediction CSVs"]
    K --> M["Plots in reports/figures"]

    B --> N["Sample media exporter<br/>preview.gif"]
    G --> N
    I --> N
    N --> O["Streamlit demo"]
```

This diagram shows the full repository flow. We start from TFRecords, decode both the video frames and the quantized ground-truth flow, extract compact metadata features, define three reliability targets, train regression baselines, then surface the results through figures, exported media, and the Streamlit demo.

## EPE Computation Focus
```mermaid
sequenceDiagram
    participant TF as TFRecord sample
    participant VD as Video decoder
    participant FD as Flow decoder
    participant EST as Flow estimator
    participant CMP as EPE comparison
    participant AGG as Aggregation

    TF->>VD: Decode frames t and t+1
    TF->>FD: Decode GT forward_flow[t]
    VD->>EST: Frame pair
    EST->>CMP: Predicted flow (u_pred, v_pred)
    FD->>CMP: Ground-truth flow (u_gt, v_gt)
    CMP->>CMP: EPE = sqrt((u_pred-u_gt)^2 + (v_pred-v_gt)^2)
    CMP->>AGG: Per-pixel EPE map
    AGG->>AGG: Mean over pixels and pairs
    AGG-->>TF: epe_mean or epe_mean_raft
```

This second diagram zooms in on how error labels are created. For each frame pair, we estimate flow, compare it to the decoded ground-truth flow, compute the End-Point Error at each pixel, and average the result to produce the label used for model training.

## Where Things Live
- `src/build_all_tables.py`: Phase 1 proxy target builder
- `src/build_all_tables_epe.py`: Farneback EPE table builder
- `src/build_all_tables_raft_epe.py`: RAFT EPE table builder
- `src/train_regressor.py`: baseline training and evaluation
- `src/plots.py`: report figure generation
- `tools/export_sample_media.py`: GIF + RAFT visual export for one sample
- `streamlit_app.py`: interactive demo layer

## Dataset Note
- Local dataset root used in this repo: `/Users/seifeddinereguige/Documents/tfds_Dataset`
- The raw dataset is not tracked in git
- Generated outputs are written to `outputs/`, which is git-ignored
