# Roadmap

This roadmap summarizes the current state of the project and what remains before final submission polish. The intention is realistic student project planning, not a polished product roadmap.

## Status

### Done
- [x] TFRecord ingestion and scenario enumeration
- [x] Ground-truth flow decoding from quantized `forward_flow`
- [x] Metadata feature extraction
- [x] Phase 1 proxy target: `reliability_score`
- [x] Phase 2 classical target: `epe_mean` using Farneback
- [x] Phase 3 modern target: `epe_mean_raft` using RAFT
- [x] Baseline regression training and saved metrics
- [x] Core report plots
- [x] Sample media exporter with RAFT flow and EPE heatmap
- [x] Streamlit demo app

### In Progress
- [ ] Repo presentation and top-level documentation polish
- [ ] Final comparison framing across proxy, Farneback, and RAFT targets
- [ ] Demo packaging for presentation day

### Next
- [ ] Scenario-holdout evaluation instead of only random row split
- [ ] Richer metadata features such as visibility, occlusion, or temporal variation
- [ ] Direct comparison of Farneback vs RAFT target difficulty
- [ ] Dataset sync with expected updated extraction
- [ ] Final report and presentation slides

## Suggested Timeline
```mermaid
gantt
    title CSI4900 Optical Flow Reliability Roadmap
    dateFormat  YYYY-MM-DD
    axisFormat  %b %d

    section Completed
    TFRecord ingestion + decoding           :done, a1, 2026-02-01, 7d
    Metadata feature extraction             :done, a2, 2026-02-08, 5d
    Proxy target + CSV pipeline             :done, a3, 2026-02-13, 5d
    Farneback EPE target                    :done, a4, 2026-02-18, 6d
    RAFT EPE target                         :done, a5, 2026-02-24, 7d
    Baselines + plots                       :done, a6, 2026-03-02, 6d
    Exporter + Streamlit demo               :done, a7, 2026-03-08, 6d

    section Current
    Documentation pack                      :active, b1, 2026-03-14, 5d
    Results framing and demo polish         :active, b2, 2026-03-17, 5d

    section Next
    Scenario-holdout evaluation             :b3, 2026-03-22, 7d
    Richer metadata features                :b4, 2026-03-26, 8d
    Farneback vs RAFT comparison            :b5, 2026-03-30, 5d
    Dataset update sync                     :b6, 2026-04-03, 5d
    Final report + presentation             :b7, 2026-04-08, 7d
```

## Why These Next Steps Matter
- Scenario-holdout evaluation is the best next validity check because it tests whether the model generalizes to unseen motion regimes, not just unseen rows.
- Richer metadata could better explain estimator failures caused by visibility, crowding, or occlusion.
- A cleaner Farneback vs RAFT comparison would make the project narrative stronger by showing how the target definition changes the learning problem.
- Dataset synchronization matters because the current local extraction is still relatively small, which limits confidence in the conclusions.
