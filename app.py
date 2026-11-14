"""
MarkerPhish — Main Application Entry Point
===========================================
Streamlit dashboard for automated evaluation of large language models in
phishing and smishing detection with concurrent hardware resource profiling.

Thesis: Platform for evaluating language models in phishing and smishing
detection with a focus on resource efficiency.
"""

from __future__ import annotations

import streamlit as st

from backend.data_loader import load_dataset_with_report
from backend.sampling import balanced_sample
from backend.hardware import scan_hardware
from backend.utils import (
    list_saved_result_files,
    load_result_by_filename,
)
from frontend import (
    render_data_overview,
    render_evaluation_runner,
    render_header,
    render_multi_comparison_view,
    render_sidebar,
    render_single_run_view,
)
from frontend.sidebar import render_sidebar_dataset_summary
from config import PREPROCESSING_VERSION
from ui.theme import inject_css

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="markerphish",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Style Injection (fonts, theme, hides Streamlit chrome)
# ──────────────────────────────────────────────
inject_css()


# ──────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────
def init_session() -> None:
    """Initialize application session state with defaults and hardware scan."""
    defaults = {
        "evaluation_run": None,
        "report": None,
        "dataset_df": None,
        "is_running": False,
        "hardware_profile": None,
        "selected_result_file": None,
        "loaded_filename": None,
        "dataset_report": None,
        "sample_info": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    if st.session_state["hardware_profile"] is None:
        st.session_state["hardware_profile"] = scan_hardware()

    # Automatically load the latest benchmark report on startup if available
    if st.session_state["selected_result_file"] is None:
        available_files = list_saved_result_files()
        if available_files:
            latest = available_files[0]
            st.session_state["selected_result_file"] = latest
            st.session_state["loaded_filename"] = latest
            st.session_state["report"] = load_result_by_filename(latest)


init_session()


@st.cache_data(show_spinner=False, max_entries=8)
def load_dataset_cached(dataset_choice: str, preprocessing_version: str):
    """
    Load the unified dataset and its preprocessing report once per
    (dataset choice, preprocessing version).

    preprocessing_version is unused in the body; it is part of the cache key, so
    a change to the preprocessing pipeline never serves a stale dataset.
    """
    return load_dataset_with_report(dataset_choice)


@st.cache_data(show_spinner=False, max_entries=16)
def sample_cached(dataset_choice: str, preprocessing_version: str, sample_size: int):
    """The balanced sample of a dataset choice (deterministic, so safe to cache)."""
    df, _ = load_dataset_cached(dataset_choice, preprocessing_version)
    result = balanced_sample(df, sample_size)
    return result.df, result.info


# ──────────────────────────────────────────────
# Main Application Flow
# ──────────────────────────────────────────────
def main() -> None:
    """Main application routine orchestrating sidebar, header, and analytical views."""
    # Render sidebar controls
    selected_model, dataset_choice, sample_size, num_gpu, mode_display_name, prompt_variant = render_sidebar()

    # Render main page branding header
    render_header()

    if not selected_model:
        st.info("Select a model and configure parameters in the sidebar to initialize the evaluation session.")
        return

    # Primary Analytical Tabs (Strictly no emojis)
    tab_data, tab_eval, tab_single, tab_multi = st.tabs(
        [
            "Dataset",
            "Run",
            "Results",
            "Compare",
        ]
    )

    # ────────────── TAB 1: DATASET OVERVIEW ──────────────
    with tab_data:
        df_sample = None
        sample_info = None
        with st.spinner("Compiling dataset subset..."):
            _, dataset_report = load_dataset_cached(dataset_choice, PREPROCESSING_VERSION)
            try:
                df_sample, sample_info = sample_cached(dataset_choice, PREPROCESSING_VERSION, sample_size)
            except ValueError as exc:
                st.error(f"Cannot draw a balanced sample: {exc}")
            st.session_state["dataset_df"] = df_sample
            st.session_state["dataset_report"] = dataset_report
            st.session_state["sample_info"] = sample_info
            render_sidebar_dataset_summary(df_sample)

        render_data_overview(df_sample, dataset_report=dataset_report, sample_info=sample_info)

    # ────────────── TAB 2: RUN EVALUATION ──────────────
    with tab_eval:
        df_eval = st.session_state.get("dataset_df")
        if df_eval is None or df_eval.empty:
            st.warning("Dataset not loaded. Return to the Dataset tab to compile records.")
        else:
            render_evaluation_runner(
                selected_model=selected_model,
                df_eval=df_eval,
                num_gpu=num_gpu,
                mode_display_name=mode_display_name,
                prompt_variant=prompt_variant,
            )

    # ────────────── TAB 3: SINGLE-RUN RESULTS & ANALYTICS ──────────────
    with tab_single:
        render_single_run_view()

    # ────────────── TAB 4: MULTI-MODEL COMPARISON ──────────────
    with tab_multi:
        render_multi_comparison_view()


if __name__ == "__main__":
    main()
