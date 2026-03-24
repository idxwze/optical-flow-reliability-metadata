from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import streamlit as st


DEFAULT_DATASET_ROOT = "/Users/seifeddinereguige/Documents/tfds_Dataset"
DEFAULT_OUTPUT_DIR = Path("outputs")
RAFT_TABLE_PATH = DEFAULT_OUTPUT_DIR / "table_all_scenarios_raft_epe.csv"


@st.cache_data(show_spinner=False)
def list_scenarios(dataset_root: str) -> list[str]:
    root = Path(dataset_root).expanduser()
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


@st.cache_data(show_spinner=False)
def load_raft_rows(csv_path: str) -> list[dict[str, str]]:
    path = Path(csv_path)
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


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


def show_media(path: Path, label: str, is_gif: bool = False) -> None:
    if path.exists():
        if is_gif:
            st.image(str(path), caption=label)
        else:
            st.image(str(path), caption=label, use_container_width=True)
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

    st.subheader("Media")
    media_col_1, media_col_2 = st.columns(2)
    with media_col_1:
        show_media(paths["gif"], "Preview GIF", is_gif=True)
        show_media(paths["flow"], "RAFT Flow")
    with media_col_2:
        show_media(paths["epe"], "RAFT EPE Heatmap")


if __name__ == "__main__":
    main()
