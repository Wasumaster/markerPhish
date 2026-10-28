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

from backend.data_loader import load_unified_dataset
from backend.hardware import scan_hardware
from backend.utils import (
    list_saved_result_files,
    load_result_by_filename,
)
from frontend import (
    inject_styles,
    render_data_overview,
    render_evaluation_runner,
    render_header,
    render_multi_comparison_view,
    render_sidebar,
    render_single_run_view,
)

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="MarkerPhish — Threat Detection & Resource Evaluation",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Style Injection (Hides Deploy button & Hamburger menu)
# ──────────────────────────────────────────────
inject_styles()


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


# ──────────────────────────────────────────────
# Main Application Flow
# ──────────────────────────────────────────────
def main() -> None:
    """Main application routine orchestrating sidebar, header, and analytical views."""
    # Render sidebar controls
    selected_model, dataset_choice, sample_size, num_gpu, mode_display_name = render_sidebar()

    # Render main page branding header
    render_header()

    if not selected_model:
        st.info("Select a model and configure parameters in the sidebar to initialize the evaluation session.")
        return

    # Primary Analytical Tabs (Strictly no emojis)
    tab_data, tab_eval, tab_single, tab_multi = st.tabs(
        [
            "Dataset Overview",
            "Run Evaluation",
            "Results & Analytics",
            "Multi-Model Comparison",
        ]
    )

    # ────────────── TAB 1: DATASET OVERVIEW ──────────────
    with tab_data:
        with st.spinner("Compiling dataset subset..."):
            df = load_unified_dataset(dataset_choice)
            if sample_size is not None and sample_size < len(df):
                df_sample = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
                df_sample["id"] = range(1, len(df_sample) + 1)
                st.caption(
                    f"Subsample active: Evaluated sample contains {sample_size:,} records "
                    f"selected via seeded pseudorandom sampling from {len(df):,} total messages."
                )
            else:
                df_sample = df

            st.session_state["dataset_df"] = df_sample

        render_data_overview(df_sample)

    # ────────────── TAB 2: RUN EVALUATION ──────────────
    with tab_eval:
        df_eval = st.session_state.get("dataset_df")
        if df_eval is None or df_eval.empty:
            st.warning("Dataset not loaded. Return to the Dataset Overview tab to compile records.")
        else:
            render_evaluation_runner(
                selected_model=selected_model,
                df_eval=df_eval,
                num_gpu=num_gpu,
                mode_display_name=mode_display_name,
            )

    # ────────────── TAB 3: SINGLE-RUN RESULTS & ANALYTICS ──────────────
    with tab_single:
        render_single_run_view()

    # ────────────── TAB 4: MULTI-MODEL COMPARISON ──────────────
    with tab_multi:
        render_multi_comparison_view()


if __name__ == "__main__":
    main()
