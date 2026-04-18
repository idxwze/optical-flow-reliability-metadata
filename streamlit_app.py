from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def _st_cache_data(show_spinner: bool = True):
    """Streamlit >= 1.18: st.cache_data; older releases (e.g. 1.12 on Python 3.9.7): experimental_memo."""
    if hasattr(st, "cache_data"):
        return st.cache_data(show_spinner=show_spinner)
    return st.experimental_memo


DEFAULT_DATASET_ROOT = "/Users/nassym/tfds_Dataset"
DEFAULT_OUTPUT_DIR = Path("outputs")
RAFT_TABLE_PATH = DEFAULT_OUTPUT_DIR / "table_all_scenarios_raft_epe.csv"
RAFT_TARGET_TOKEN = "epe_mean_raft"


@_st_cache_data(show_spinner=False)
def list_scenarios(dataset_root: str) -> list[str]:
    root = Path(dataset_root).expanduser()
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


@_st_cache_data(show_spinner=False)
def load_raft_rows(csv_path: str) -> list[dict[str, str]]:
    path = Path(csv_path)
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


@_st_cache_data(show_spinner=False)
def load_json_metrics(path_str: str) -> dict | None:
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return None


@_st_cache_data(show_spinner=False)
def load_preds_numeric(preds_path_str: str) -> list[dict[str, Any]]:
    path = Path(preds_path_str)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                rec: dict[str, Any] = {
                    "scenario": str(row["scenario"]),
                    "record_index": int(float(row["record_index"])),
                    "y_true": float(row["y_true"]),
                    "y_pred": float(row["y_pred"]),
                }
                split_val = (row.get("split") or "").strip().lower()
                if split_val in ("train", "test"):
                    rec["split"] = split_val
                rows.append(rec)
            except (KeyError, ValueError, TypeError):
                continue
    return rows


def discover_raft_pred_metric_pairs(output_dir: Path) -> list[tuple[str, Path, Path]]:
    """Pairs of (model_name, preds_csv, metrics_json) for RAFT mean EPE target."""
    pairs: list[tuple[str, Path, Path]] = []
    suffix = f"_{RAFT_TARGET_TOKEN}.csv"
    for preds_path in sorted(output_dir.glob(f"preds_*{suffix}")):
        name = preds_path.name
        if not name.startswith("preds_") or not name.endswith(suffix):
            continue
        model = name[len("preds_") : -len(suffix)]
        if not model:
            continue
        metrics_path = output_dir / f"metrics_{model}_{RAFT_TARGET_TOKEN}.json"
        pairs.append((model, preds_path, metrics_path))
    return pairs


def lookup_pred_row(
    preds_rows: list[dict[str, Any]], scenario: str, record_index: int
) -> dict[str, Any] | None:
    for row in preds_rows:
        if row["scenario"] == scenario and row["record_index"] == record_index:
            return row
    return None


def plot_raft_true_vs_pred(
    preds_rows: list[dict[str, Any]],
    *,
    highlight: dict[str, Any] | None,
    model_label: str,
) -> plt.Figure:
    y_t = np.asarray([r["y_true"] for r in preds_rows], dtype=np.float64)
    y_p = np.asarray([r["y_pred"] for r in preds_rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(y_t, y_p, s=28, alpha=0.45, label="Test rows")
    lo = float(min(np.min(y_t), np.min(y_p)))
    hi = float(max(np.max(y_t), np.max(y_p)))
    pad = 0.05 * (hi - lo + 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1.2, label="y = x")
    if highlight is not None:
        ax.scatter(
            [highlight["y_true"]],
            [highlight["y_pred"]],
            s=140,
            color="crimson",
            edgecolors="black",
            linewidths=1.2,
            zorder=5,
            label="Current record",
        )
    ax.set_xlabel("True RAFT mean EPE")
    ax.set_ylabel("Predicted RAFT mean EPE")
    ax.set_title(f"Evaluation: held-out test ({model_label})")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_raft_demo_all_splits(
    demo_rows: list[dict[str, Any]],
    *,
    highlight: dict[str, Any] | None,
    model_label: str,
) -> plt.Figure:
    """Train (in-sample) vs test (same model); for UI exploration only."""
    train_t, train_p = [], []
    test_t, test_p = [], []
    for r in demo_rows:
        sp = r.get("split", "test")
        if sp == "train":
            train_t.append(float(r["y_true"]))
            train_p.append(float(r["y_pred"]))
        else:
            test_t.append(float(r["y_true"]))
            test_p.append(float(r["y_pred"]))
    all_t = np.asarray(train_t + test_t, dtype=np.float64)
    all_p = np.asarray(train_p + test_p, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    if train_t:
        ax.scatter(train_t, train_p, s=22, alpha=0.35, c="#888888", label="Train (demo / in-sample)")
    if test_t:
        ax.scatter(test_t, test_p, s=30, alpha=0.55, c="#1f77b4", label="Test (held-out)")
    lo = float(min(np.min(all_t), np.min(all_p)))
    hi = float(max(np.max(all_t), np.max(all_p)))
    pad = 0.05 * (hi - lo + 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1.2, label="y = x")
    if highlight is not None:
        ax.scatter(
            [float(highlight["y_true"])],
            [float(highlight["y_pred"])],
            s=140,
            color="crimson",
            edgecolors="black",
            linewidths=1.2,
            zorder=5,
            label="Current record",
        )
    ax.set_xlabel("True RAFT mean EPE")
    ax.set_ylabel("Predicted RAFT mean EPE")
    ax.set_title(f"Demonstration: all rows ({model_label})")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_raft_record_bars(y_true: float, y_pred: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.bar(["True EPE", "Predicted EPE"], [y_true, y_pred], color=["#2ecc71", "#3498db"], width=0.55)
    ax.set_ylabel("Mean EPE (RAFT)")
    ax.set_title("Current record")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def scenario_record_indices(rows: list[dict[str, str]], scenario: str) -> list[int]:
    indices: set[int] = set()
    for row in rows:
        if row.get("scenario") != scenario:
            continue
        try:
            indices.add(int(float(row.get("record_index", "0"))))
        except Exception:
            continue
    return sorted(indices)


def lookup_row(rows: list[dict[str, str]], scenario: str, record_index: int) -> dict[str, str] | None:
    for row in rows:
        if row.get("scenario") != scenario:
            continue
        try:
            row_index = int(float(row.get("record_index", "0")))
        except Exception:
            continue
        if row_index == record_index:
            return row
    return None


def media_paths(output_dir: Path, scenario: str, record_index: int) -> dict[str, Path]:
    record_dir = output_dir / "sample_media" / scenario / f"record_{record_index:05d}"
    return {
        "record_dir": record_dir,
        "gif": record_dir / "preview.gif",
        "flow": record_dir / "flow_raft.png",
        "epe": record_dir / "epe_raft.png",
    }


def run_exporter(dataset_root: str, scenario: str, record_index: int) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-m",
        "tools.export_sample_media",
        "--dataset_root",
        dataset_root,
        "--scenario",
        scenario,
        "--record_index",
        str(record_index),
        "--fps",
        "8",
        "--raft_model",
        "small",
        "--pair_index",
        "0",
    ]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).resolve().parent)


def metadata_table(row: dict[str, str] | None) -> list[dict[str, str]]:
    if row is None:
        return []
    keys = [
        "num_instances",
        "camera_translation_speed_mean",
        "camera_rotation_change_mean",
        "instance_speed_mean",
        "epe_mean_raft",
    ]
    return [{"metric": key, "value": row.get(key, "")} for key in keys]


def _st_image_responsive(image_path: str, *, caption: str) -> None:
    """Widen image when supported; some Streamlit 1.39.x builds omit use_container_width on st.image."""
    try:
        st.image(image_path, caption=caption, use_container_width=True)
    except TypeError:
        try:
            st.image(image_path, caption=caption, use_column_width=True)
        except TypeError:
            st.image(image_path, caption=caption)


def show_media(path: Path, label: str, is_gif: bool = False) -> None:
    if path.exists():
        if is_gif:
            st.image(str(path), caption=label)
        else:
            _st_image_responsive(str(path), caption=label)
    else:
        st.info(f"{label} is not available yet.")


def main():
    st.set_page_config(page_title="Optical Flow Reliability Demo", layout="wide")
    st.title("Optical Flow Reliability Demo")
    st.caption("Preview TFRecord samples and RAFT-based visualizations for one scenario record.")

    dataset_root = st.text_input("Dataset root", value=DEFAULT_DATASET_ROOT)
    scenarios = list_scenarios(dataset_root)
    raft_rows = load_raft_rows(str(RAFT_TABLE_PATH))

    if not scenarios:
        st.warning("No scenario folders were found under the selected dataset root.")
        st.stop()

    selected_scenario = st.selectbox("Scenario", options=scenarios)

    available_indices = scenario_record_indices(raft_rows, selected_scenario)
    if available_indices:
        selected_record_index = st.select_slider(
            "Record index",
            options=available_indices,
            value=available_indices[0],
        )
        st.caption(
            f"Using indices discovered in {RAFT_TABLE_PATH}. "
            f"Found {len(available_indices)} row(s) for this scenario."
        )
    else:
        selected_record_index = st.select_slider("Record index", options=[0], value=0)
        st.caption(
            "No matching rows were found in outputs/table_all_scenarios_raft_epe.csv "
            "for this scenario, so record index 0 is used."
        )

    paths = media_paths(DEFAULT_OUTPUT_DIR, selected_scenario, selected_record_index)
    row = lookup_row(raft_rows, selected_scenario, selected_record_index)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("Generate media", type="primary", use_container_width=True):
            with st.spinner("Generating sample media..."):
                result = run_exporter(
                    dataset_root=dataset_root,
                    scenario=selected_scenario,
                    record_index=selected_record_index,
                )
            if result.returncode == 0:
                st.success("Media generation finished.")
            else:
                st.error("Media generation failed.")
            if result.stdout.strip():
                st.code(result.stdout.strip(), language="text")
            if result.stderr.strip():
                st.code(result.stderr.strip(), language="text")

    with col_b:
        if paths["record_dir"].exists():
            st.success(f"Using media from {paths['record_dir']}")
        else:
            st.info("No generated media found yet. Click Generate media to create it.")

    st.subheader("Metadata")
    if row is None:
        st.info("No metadata row was found in outputs/table_all_scenarios_raft_epe.csv for this sample.")
    else:
        st.table(metadata_table(row))

    st.subheader("RAFT EPE: true vs predicted")
    pairs = discover_raft_pred_metric_pairs(DEFAULT_OUTPUT_DIR)
    if not pairs:
        st.info(
            f"No `outputs/preds_*_{RAFT_TARGET_TOKEN}.csv` files found. Train a regressor on the RAFT "
            "table so predictions are saved, for example: "
            f"`python3 -m src.train_regressor --csv outputs/table_all_scenarios_raft_epe.csv "
            f"--target {RAFT_TARGET_TOKEN} --split_mode scenario --repeats 1` "
            "(writes held-out test preds plus `*_demo.csv` with predictions on every row for this UI)."
        )
    else:
        model_options = [m.replace("_", " ").title() for m, _, _ in pairs]
        choice = st.selectbox(
            "Model",
            options=list(range(len(pairs))),
            format_func=lambda i: model_options[i],
            key="raft_preds_model",
        )
        _model_name, preds_path, metrics_path = pairs[int(choice)]
        preds_rows_eval = load_preds_numeric(str(preds_path))
        demo_path = preds_path.parent / f"{preds_path.stem}_demo.csv"
        preds_rows_demo = (
            load_preds_numeric(str(demo_path)) if demo_path.exists() else []
        )
        metrics_payload = load_json_metrics(str(metrics_path))

        st.caption(
            "Metrics use **held-out test** rows only. Per-record numbers and the bar chart use "
            "**demo** predictions on **all** rows (same fitted model) when `*_demo.csv` exists — "
            "train points are in-sample and optimistic."
        )

        if metrics_payload:
            tm = metrics_payload.get("test_metrics") or {}
            c_mae, c_rmse, c_r2, c_sp = st.columns(4)
            c_mae.metric("Test MAE", f"{float(tm.get('mae', float('nan'))):.4f}")
            c_rmse.metric("Test RMSE", f"{float(tm.get('rmse', float('nan'))):.4f}")
            c_r2.metric("Test R²", f"{float(tm.get('r2', float('nan'))):.4f}")
            c_sp.metric("Test Spearman", f"{float(tm.get('spearman', float('nan'))):.4f}")
            with st.expander("Full metrics JSON path and target"):
                st.caption(f"{metrics_path}")
                st.caption(f"Target column: {metrics_payload.get('target', RAFT_TARGET_TOKEN)}")
        else:
            st.warning(f"No metrics file found at `{metrics_path}` (preds-only view).")

        pred_row = lookup_pred_row(preds_rows_demo, selected_scenario, selected_record_index)
        pred_source = "demo"
        if pred_row is None:
            pred_row = lookup_pred_row(preds_rows_eval, selected_scenario, selected_record_index)
            pred_source = "test"
        if pred_row is None:
            st.warning(
                "This scenario / record index was not found in demo or test prediction CSVs. "
                "Regenerate training outputs after building `table_all_scenarios_raft_epe.csv`, "
                "or pick a scenario/record that exists in that table."
            )
        else:
            yt = float(pred_row["y_true"])
            yp = float(pred_row["y_pred"])
            err = yp - yt
            abs_err = abs(err)
            st.markdown("**Current record (numerical)**")
            n1, n2, n3, n4 = st.columns(4)
            n1.metric("True RAFT mean EPE", f"{yt:.4f}")
            n2.metric("Predicted RAFT mean EPE", f"{yp:.4f}")
            n3.metric("Prediction error (pred − true)", f"{err:+.4f}")
            n4.metric("Absolute error", f"{abs_err:.4f}")
            if pred_source == "demo" and pred_row.get("split"):
                st.caption(f"Source: **{demo_path.name}** (`split={pred_row.get('split')}`).")

        if preds_rows_eval:
            fig_bar = (
                plot_raft_record_bars(float(pred_row["y_true"]), float(pred_row["y_pred"]))
                if pred_row is not None
                else None
            )
            fig_eval = plot_raft_true_vs_pred(
                preds_rows_eval,
                highlight=pred_row,
                model_label=model_options[int(choice)],
            )
            if preds_rows_demo:
                fig_demo = plot_raft_demo_all_splits(
                    preds_rows_demo,
                    highlight=pred_row,
                    model_label=model_options[int(choice)],
                )
            else:
                fig_demo = None

            bcol, ecol, dcol = st.columns([1, 1.15, 1.15])
            with bcol:
                if pred_row is not None and fig_bar is not None:
                    try:
                        st.pyplot(fig_bar, use_container_width=True)
                    except TypeError:
                        st.pyplot(fig_bar)
                    plt.close(fig_bar)
                else:
                    st.caption("Bar chart needs a matching row in demo or test preds.")
            with ecol:
                try:
                    st.pyplot(fig_eval, use_container_width=True)
                except TypeError:
                    st.pyplot(fig_eval)
                plt.close(fig_eval)
            with dcol:
                if fig_demo is not None:
                    try:
                        st.pyplot(fig_demo, use_container_width=True)
                    except TypeError:
                        st.pyplot(fig_demo)
                    plt.close(fig_demo)
                else:
                    st.info(
                        f"No `{demo_path.name}` yet. Re-run `src.train_regressor` with `--repeats 1` "
                        "(single run, `save_outputs=True`) to emit demo predictions for every row."
                    )
            st.caption(
                f"Evaluation scatter: {len(preds_rows_eval)} held-out test rows in `{preds_path.name}`."
                + (
                    f" Demo scatter: {len(preds_rows_demo)} rows in `{demo_path.name}`."
                    if preds_rows_demo
                    else ""
                )
            )
        else:
            st.warning(f"`{preds_path.name}` is empty or unreadable.")

    st.subheader("Media")
    media_col_1, media_col_2 = st.columns(2)
    with media_col_1:
        show_media(paths["gif"], "Preview GIF", is_gif=True)
        show_media(paths["flow"], "RAFT Flow")
    with media_col_2:
        show_media(paths["epe"], "RAFT EPE Heatmap")


if __name__ == "__main__":
    main()
