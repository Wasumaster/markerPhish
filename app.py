"""
MarkerPhish — Streamlit Dashboard
===================================
Analytics and evaluation platform for language models in phishing and smishing
threat detection, with concurrent measurement of computational overhead.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    BINARY_LABELS,
    SAMPLE_SIZES,
    is_binary_match,
    to_binary_label,
)
from data_loader import DATASET_CHOICES, load_unified_dataset
from evaluator import (
    EvaluationRun,
    check_ollama_connection,
    list_available_models,
    run_evaluation,
)
from hardware import resolve_num_gpu, scan_hardware
from metrics import (
    build_full_report,
    compute_classification_metrics,
    compute_resource_metrics,
)
from utils import (
    format_result_label,
    list_saved_result_files,
    load_latest_results,
    load_result_by_filename,
    save_results,
)

# ──────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="MarkerPhish — Threat Detection & Resource Evaluation",
    page_icon=":material/shield:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Design System & Professional Styling
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }

    /* Header styling */
    .platform-header {
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #1e293b;
    }

    .platform-title {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #f8fafc;
        margin: 0 0 0.25rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .platform-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        margin: 0;
        font-weight: 400;
        line-height: 1.4;
    }

    .academic-tag {
        display: inline-block;
        padding: 2px 8px;
        background: #1e293b;
        color: #cbd5e1;
        font-size: 0.75rem;
        font-weight: 500;
        border-radius: 4px;
        border: 1px solid #334155;
        margin-top: 0.4rem;
    }

    /* Metric card overrides */
    div[data-testid="stMetric"] {
        background: #131d2e;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 14px 18px;
        transition: border-color 0.2s ease;
    }

    div[data-testid="stMetric"]:hover {
        border-color: #3b82f6;
    }

    div[data-testid="stMetric"] label {
        font-weight: 600;
        color: #94a3b8;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-weight: 700;
        font-size: 1.65rem;
        color: #f8fafc;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0d131f;
        border-right: 1px solid #1e293b;
    }

    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #f1f5f9;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    /* Status badges */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    .badge-online {
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .badge-offline {
        background: rgba(244, 63, 94, 0.12);
        color: #f87171;
        border: 1px solid rgba(244, 63, 94, 0.3);
    }

    .badge-info {
        background: rgba(59, 130, 246, 0.12);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }

    .badge-warning {
        background: rgba(245, 158, 11, 0.12);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    /* Container cards */
    .card-panel {
        background: #131d2e;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }

    .card-panel-header {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 8px;
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #f8fafc;
        padding-bottom: 6px;
        border-bottom: 1px solid #1e293b;
        margin: 24px 0 16px 0;
    }

    /* Table styling */
    .dataframe {
        border-radius: 6px;
        overflow: hidden;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #1e293b;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 0.9rem;
        font-weight: 500;
        color: #94a3b8;
        padding: 8px 16px;
        border-radius: 6px 6px 0 0;
    }

    .stTabs [aria-selected="true"] {
        color: #3b82f6 !important;
        font-weight: 600;
        border-bottom: 2px solid #3b82f6 !important;
    }

    /* Primary button */
    .stButton > button {
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s ease;
    }

    /* Callout notice */
    .notice-box {
        background: #0f172a;
        border-left: 3px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 0 6px 6px 0;
        font-size: 0.85rem;
        color: #cbd5e1;
        margin: 12px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# Session state initialization
# ──────────────────────────────────────────────
def init_session():
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
# Helper: Backward Compatibility for Saved Reports
# ──────────────────────────────────────────────
def ensure_report_compatibility(report: dict) -> dict:
    """
    Ensure saved results from older schema versions have binary metrics populated.
    """
    cls = report.get("classification_metrics", {})
    if "binary_confusion_matrix" not in cls and "raw_predictions" in report:
        # Re-derive binary metrics from raw predictions
        raw = report["raw_predictions"]
        y_true = [r.get("true", r.get("true_channel", "valid")) for r in raw]
        y_pred = [r.get("predicted", r.get("predicted_channel", "unknown")) for r in raw]
        recomputed = compute_classification_metrics(y_true, y_pred)
        report["classification_metrics"] = recomputed

        # Also add binary distributions if missing
        dist = report.get("data_distribution", {})
        if "true_binary" not in dist:
            dist["true_binary"] = {
                "malicious": sum(1 for t in y_true if to_binary_label(t) == "malicious"),
                "safe": sum(1 for t in y_true if to_binary_label(t) == "safe"),
            }
            dist["predicted_binary"] = {
                "malicious": sum(1 for p in y_pred if to_binary_label(p) == "malicious"),
                "safe": sum(1 for p in y_pred if to_binary_label(p) == "safe"),
            }
            report["data_distribution"] = dist

    return report


# ──────────────────────────────────────────────
# Sidebar: System Status & Controls
# ──────────────────────────────────────────────
def render_sidebar():
    hw = st.session_state["hardware_profile"]

    with st.sidebar:
        st.markdown(
            """
            <div style="padding-bottom: 8px;">
                <span style="font-size: 1.25rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.01em;">MarkerPhish</span><br>
                <span style="font-size: 0.78rem; color: #94a3b8;">Evaluation & Analytics Platform</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        # ── 1. MODEL SELECTION (Very Top) ──
        st.markdown("### Model Selection")

        # Inference Server Connection Status
        connected = check_ollama_connection()
        if connected:
            st.markdown(
                '<div class="badge badge-online">Ollama Service Online</div>',
                unsafe_allow_html=True,
            )
            st.caption("Endpoint: `http://localhost:11434`")
        else:
            st.markdown(
                '<div class="badge badge-offline">Ollama Service Offline</div>',
                unsafe_allow_html=True,
            )
            st.caption("Start Ollama service: `ollama serve`")
            st.error("Server connection required to evaluate models.")
            return None, None, None, None, None

        # Installed Models Dropdown
        models = list_available_models()
        if not models:
            st.warning("No locally installed models detected. Execute `ollama pull <model>` in terminal.")
            return None, None, None, None, None

        model_names = [m["name"] for m in models]
        model_sizes = {m["name"]: m["size"] for m in models}

        selected_model = st.selectbox(
            "Installed LLM",
            options=model_names,
            index=0,
            help="Select a model installed in local Ollama repository",
        )

        if selected_model:
            size_mb = model_sizes.get(selected_model, 0) / (1024 * 1024)
            st.caption(f"Storage footprint: **{size_mb:.1f} MB** ({size_mb / 1024:.2f} GB)")

        st.divider()

        # ── 2. DATASET & SCOPE (Second) ──
        st.markdown("### Dataset & Scope")
        dataset_label = st.selectbox(
            "Evaluation Dataset",
            options=list(DATASET_CHOICES.keys()),
            index=0,
            help="Select dataset combination for evaluation",
        )
        dataset_choice = DATASET_CHOICES[dataset_label]

        sample_label = st.selectbox(
            "Evaluation Sample Size",
            options=list(SAMPLE_SIZES.keys()),
            index=1,  # Small (100) default
            help="Select subsample size for testing and latency measurement",
        )
        sample_size = SAMPLE_SIZES[sample_label]

        st.divider()

        # ── 3. EXECUTION ENVIRONMENT (Very Bottom) ──
        st.markdown("### Execution Environment")

        # Hardware scan telemetry card
        gpu_label = hw.gpu_name if hw.has_gpu else "No dedicated GPU detected"
        gpu_sub = (
            f"VRAM: {hw.gpu_vram_mb} MB | Driver: {hw.gpu_driver} | Compute: {hw.compute_cap}"
            if hw.has_gpu
            else "Standard multi-threaded CPU execution"
        )

        st.markdown(
            f"""
            <div class="card-panel" style="padding: 10px 14px; margin-bottom: 10px;">
                <div class="card-panel-header">Host Hardware Profile</div>
                <div style="font-size: 0.8rem; color: #f1f5f9; font-weight: 500;">
                    CPU: {hw.cpu_name}
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 2px;">
                    Cores: {hw.cpu_cores_physical} physical, {hw.cpu_cores_logical} logical | RAM: {hw.ram_total_gb:.1f} GB ({hw.ram_available_gb:.1f} GB free)
                </div>
                <div style="font-size: 0.8rem; color: #f1f5f9; font-weight: 500; margin-top: 6px;">
                    GPU: {gpu_label}
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 2px;">
                    {gpu_sub}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Execution Mode Selection
        device_options = ["CPU Safe Mode (Recommended)", "GPU Hardware Acceleration"]
        default_index = 0 if hw.recommended_mode == "cpu" else 1

        selected_mode_str = st.radio(
            "Inference Device Selection",
            options=device_options,
            index=default_index,
            help=(
                "CPU Safe Mode executes inference on host processor (num_gpu=0), "
                "guaranteeing stability on Pascal (sm_61) GPUs and avoiding CUDA PTX buffer overruns. "
                "GPU Acceleration enables CUDA device offloading."
            ),
        )

        num_gpu, mode_display_name = resolve_num_gpu(selected_mode_str)

        # Factual feedback on selected mode
        if num_gpu == 0:
            st.markdown(
                '<div class="notice-box">'
                "<strong>CPU Safe Mode active:</strong> Ollama `num_gpu=0` set. "
                "Inference runs strictly on CPU threads, ensuring reproducibility and zero CUDA runtime crashes."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="notice-box">'
                "<strong>GPU Acceleration active:</strong> Ollama CUDA offloading enabled. "
                "If CUDA toolchain or PTX errors occur, switch back to CPU Safe Mode."
                "</div>",
                unsafe_allow_html=True,
            )

        return selected_model, dataset_choice, sample_size, num_gpu, mode_display_name


# ──────────────────────────────────────────────
# Tab 1: Dataset Overview
# ──────────────────────────────────────────────
def render_data_overview(df: pd.DataFrame):
    st.markdown('<div class="section-title">Dataset Characteristics & Composition</div>', unsafe_allow_html=True)

    # Top KPI summary: 3 columns
    c1, c2, c3 = st.columns([1, 1, 1.2])
    with c1:
        st.metric("Total Evaluated Records", f"{len(df):,}")
    with c2:
        malicious_count = len(df[df["true_label"].isin(["phishing", "smishing"])])
        malicious_pct = (malicious_count / len(df) * 100) if len(df) > 0 else 0
        st.metric("Malicious Threat Messages", f"{malicious_count:,}", f"{malicious_pct:.1f}% of total")
    with c3:
        valid_count = len(df[df["true_label"] == "valid"])
        valid_pct = (valid_count / len(df) * 100) if len(df) > 0 else 0
        st.metric("Safe Legitimate Messages", f"{valid_count:,}", f"{valid_pct:.1f}% of total")

    # Asymmetric layout: Visualization (58%) & Provenance Table (42%)
    col_chart, col_meta = st.columns([1.4, 1.0])

    with col_chart:
        # Threat vs Safe distribution bar chart
        df_binary = df.copy()
        df_binary["binary_class"] = df_binary["true_label"].apply(to_binary_label)
        binary_counts = df_binary["binary_class"].value_counts().reset_index()
        binary_counts.columns = ["Classification", "Count"]

        color_map = {
            "malicious": "#ef4444",
            "safe": "#10b981",
        }

        fig = px.bar(
            binary_counts,
            x="Classification",
            y="Count",
            color="Classification",
            color_discrete_map=color_map,
            title="Binary Threat Distribution (Safe vs. Malicious)",
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#94a3b8"),
            showlegend=False,
            height=300,
            title_font_size=14,
            margin=dict(t=40, b=20, l=20, r=20),
            xaxis=dict(gridcolor="#1e293b", title=""),
            yaxis=dict(gridcolor="#1e293b", title="Record Count"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_meta:
        st.markdown(
            """
            <div class="card-panel-header">Distribution Channel Breakdown</div>
            """,
            unsafe_allow_html=True,
        )
        source_counts = df.groupby(["source", "true_label"]).size().unstack(fill_value=0)
        st.dataframe(source_counts, use_container_width=True)

        st.caption(
            "Evaluation mapping: Emails (PhishFuzzer) and SMS messages (Mendeley) "
            "are unified into binary classes (Malicious threats vs. Safe communications)."
        )

    # Sample Inspector
    with st.expander("Inspect Message Samples by Class", expanded=False):
        for label, class_desc in [
            ("phishing", "Phishing Email (Malicious)"),
            ("smishing", "Smishing SMS (Malicious)"),
            ("valid", "Valid Communication (Safe)"),
        ]:
            subset = df[df["true_label"] == label]
            if not subset.empty:
                st.markdown(f"**{class_desc}**")
                sample_item = subset.iloc[0]
                st.code(sample_item["text"][:350], language="text")


# ──────────────────────────────────────────────
# Tab 2: Run Evaluation (Live Monitoring)
# ──────────────────────────────────────────────
def render_evaluation_runner(
    selected_model: str,
    df_eval: pd.DataFrame,
    num_gpu: int | None,
    mode_display_name: str,
):
    st.markdown('<div class="section-title">Automated Model Evaluation</div>', unsafe_allow_html=True)

    # Manifest panel
    st.markdown(
        f"""
        <div class="card-panel">
            <div class="card-panel-header">Evaluation Configuration Manifest</div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; font-size: 0.85rem;">
                <div>
                    <span style="color: #94a3b8;">Target Model:</span><br>
                    <strong style="color: #f8fafc; font-family: monospace;">{selected_model}</strong>
                </div>
                <div>
                    <span style="color: #94a3b8;">Execution Environment:</span><br>
                    <strong style="color: #38bdf8;">{mode_display_name}</strong>
                </div>
                <div>
                    <span style="color: #94a3b8;">Dataset Volume:</span><br>
                    <strong style="color: #f8fafc;">{len(df_eval):,} messages</strong>
                </div>
                <div>
                    <span style="color: #94a3b8;">Evaluation Standard:</span><br>
                    <strong style="color: #10b981;">Binary Threat Classification</strong>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_btn, _ = st.columns([1, 2])
    with col_btn:
        start_btn = st.button(
            "Execute Evaluation Run",
            icon=":material/play_arrow:",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.get("is_running", False),
        )

    if start_btn:
        st.session_state["is_running"] = True

        progress_bar = st.progress(0, text="Initializing evaluation session...")
        telemetry_area = st.empty()
        live_log_area = st.empty()

        correct_binary_count = 0
        total_latency = 0.0

        def progress_callback(current, total, result):
            nonlocal correct_binary_count, total_latency
            pct = current / total
            is_correct_bin = result.is_binary_correct
            status_text = "CORRECT" if is_correct_bin else "MISMATCH"

            progress_bar.progress(
                pct,
                text=(
                    f"Processing record {current}/{total} | "
                    f"Prediction: {result.predicted_label} [{status_text}] | "
                    f"Latency: {result.latency_seconds:.2f}s"
                ),
            )

            if is_correct_bin:
                correct_binary_count += 1
            total_latency += result.latency_seconds

            # Refresh telemetry metrics every 2 records or on completion
            if current % 2 == 0 or current == total:
                with telemetry_area.container():
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Processed Records", f"{current}/{total}")
                    with m2:
                        running_acc = (correct_binary_count / current) if current > 0 else 0
                        st.metric("Running Binary Accuracy", f"{running_acc:.1%}")
                    with m3:
                        avg_lat = (total_latency / current) if current > 0 else 0
                        st.metric("Mean Latency", f"{avg_lat:.2f}s")
                    with m4:
                        eta_seconds = avg_lat * (total - current) if avg_lat > 0 else 0
                        mins, secs = divmod(int(eta_seconds), 60)
                        st.metric("Estimated Remaining", f"{mins}m {secs}s")

            # Refresh live activity log card
            with live_log_area.container():
                outcome_color = "#10b981" if is_correct_bin else "#f43f5e"
                st.markdown(
                    f"""
                    <div class="card-panel" style="margin-top: 10px; padding: 12px 16px;">
                        <div class="card-panel-header">Recent Inference Telemetry (Record #{result.record_id})</div>
                        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.85rem;">
                            <span>True Class: <strong>{result.true_label}</strong> ({result.true_binary})</span>
                            <span>Model Prediction: <strong>{result.predicted_label}</strong> ({result.predicted_binary})</span>
                            <span style="color: {outcome_color}; font-weight: 600;">{status_text}</span>
                            <span>Latency: <strong>{result.latency_seconds:.3f}s</strong></span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        eval_run = run_evaluation(
            model_name=selected_model,
            df=df_eval,
            num_gpu=num_gpu,
            progress_callback=progress_callback,
        )

        st.session_state["evaluation_run"] = eval_run
        hw_dict = st.session_state["hardware_profile"].to_dict()
        report = build_full_report(eval_run, hardware_profile=hw_dict, execution_mode=mode_display_name)
        st.session_state["report"] = report

        filepath = save_results(report, selected_model)
        new_filename = os.path.basename(filepath)
        st.session_state["selected_result_file"] = new_filename
        st.session_state["loaded_filename"] = new_filename

        progress_bar.progress(1.0, text="Evaluation completed successfully.")
        st.session_state["is_running"] = False
        st.success(f"Evaluation session completed. Persistent report archived to `{filepath}`.")


# ──────────────────────────────────────────────
# Tab 3: Results & Analytics
# ──────────────────────────────────────────────
def render_results(report: dict):
    report = ensure_report_compatibility(report)
    cls = report["classification_metrics"]
    res = report["resource_metrics"]
    dist = report.get("data_distribution", {})

    # Warning alert if 100% predictions failed
    if cls.get("unknown_predictions") == cls.get("total_evaluated") and cls.get("total_evaluated", 0) > 0:
        st.error(
            "All model inferences returned 'unknown' or timed out. "
            "If using Pascal architecture (GTX 1060), ensure 'CPU Safe Mode' is enabled in the sidebar."
        )

    # ── Level 1: Primary Binary Threat Detection Performance (Task 1) ──
    st.markdown('<div class="section-title">Binary Threat Detection Performance</div>', unsafe_allow_html=True)

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.metric(
            "Binary Accuracy",
            f"{cls['accuracy']:.2%}",
            help="Overall proportion of correct classifications (Threats vs. Safe messages)",
        )
    with b2:
        st.metric(
            "Threat Precision",
            f"{cls['precision']:.2%}",
            help="Of messages flagged as threats, percentage that were genuinely malicious (TP / (TP + FP))",
        )
    with b3:
        st.metric(
            "Threat Recall (Detection Rate)",
            f"{cls['recall']:.2%}",
            help="Of all actual threats, percentage successfully intercepted (TP / (TP + FN))",
        )
    with b4:
        st.metric(
            "Threat F1-Score",
            f"{cls['f1']:.2%}",
            help="Harmonic mean of threat precision and threat recall",
        )

    # Secondary diagnostic metrics
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric(
            "Safe Specificity",
            f"{cls.get('specificity', 0.0):.2%}",
            help="True Negative Rate: proportion of legitimate safe messages correctly passed (TN / (TN + FP))",
        )
    with s2:
        st.metric(
            "Threat Miss Rate (FNR)",
            f"{cls.get('miss_rate', 0.0):.2%}",
            help="False Negative Rate: proportion of threats erroneously allowed through as safe (FN / (TP + FN))",
        )
    with s3:
        st.metric(
            "False Alarm Rate (FPR)",
            f"{cls.get('false_alarm_rate', 0.0):.2%}",
            help="False Positive Rate: proportion of safe messages erroneously flagged as threats (FP / (TN + FP))",
        )
    with s4:
        st.metric(
            "Failed / Unknown Inferences",
            f"{cls['unknown_predictions']}",
            help="Number of requests that timed out or generated unparsable output",
        )

    # ── Level 2: Visual Diagnostics (Confusion Matrix & Distributions) ──
    col_cm, col_dist = st.columns([1.2, 1.0])

    with col_cm:
        st.markdown("#### Binary Confusion Matrix")
        binary_cm = cls.get("binary_confusion_matrix")
        if binary_cm:
            # Labels: Malicious, Safe
            cm_matrix = np.array(binary_cm)
            labels_disp = ["Malicious (Threat)", "Safe (Legitimate)"]

            fig_cm = px.imshow(
                cm_matrix,
                text_auto=True,
                x=labels_disp,
                y=labels_disp,
                color_continuous_scale=[
                    [0.0, "#0f172a"],
                    [0.5, "#1e3a8a"],
                    [1.0, "#2563eb"],
                ],
                labels=dict(x="Predicted Class", y="Ground Truth Class", color="Count"),
            )
            fig_cm.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#f1f5f9", size=12),
                height=340,
                margin=dict(t=30, b=40, l=40, r=20),
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            st.caption(
                f"Evaluation Breakdown: "
                f"True Positives: {cls.get('tp', 0)} | "
                f"True Negatives: {cls.get('tn', 0)} | "
                f"False Positives: {cls.get('fp', 0)} | "
                f"False Negatives: {cls.get('fn', 0)}"
            )

    with col_dist:
        st.markdown("#### Threat vs. Safe Distribution Comparison")
        t_dist = dist.get("true_binary", {})
        p_dist = dist.get("predicted_binary", {})

        if t_dist and p_dist:
            dist_df = pd.DataFrame(
                [
                    {"Subset": "Ground Truth", "Class": "Malicious", "Count": t_dist.get("malicious", 0)},
                    {"Subset": "Ground Truth", "Class": "Safe", "Count": t_dist.get("safe", 0)},
                    {"Subset": "Model Predicted", "Class": "Malicious", "Count": p_dist.get("malicious", 0)},
                    {"Subset": "Model Predicted", "Class": "Safe", "Count": p_dist.get("safe", 0)},
                ]
            )

            fig_dist = px.bar(
                dist_df,
                x="Subset",
                y="Count",
                color="Class",
                barmode="group",
                color_discrete_map={"Malicious": "#ef4444", "Safe": "#10b981"},
            )
            fig_dist.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#94a3b8"),
                height=340,
                margin=dict(t=30, b=40, l=20, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(gridcolor="#1e293b", title=""),
                yaxis=dict(gridcolor="#1e293b", title="Count"),
            )
            st.plotly_chart(fig_dist, use_container_width=True)

    # ── Level 3: Distribution-Channel Breakdown (Task 3: No detail lost) ──
    with st.expander("Channel-Level Breakdown (Phishing vs. Smishing vs. Valid)", expanded=False):
        ch_metrics = cls.get("channel_metrics", {})
        if ch_metrics and "per_class" in ch_metrics:
            ch_df = pd.DataFrame(ch_metrics["per_class"]).T.reset_index()
            ch_df.columns = ["Channel", "Precision", "Recall", "F1-Score", "Support"]
            st.dataframe(
                ch_df.style.format(
                    {
                        "Precision": "{:.4f}",
                        "Recall": "{:.4f}",
                        "F1-Score": "{:.4f}",
                        "Support": "{:.0f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        if "confusion_matrix" in ch_metrics:
            st.markdown("**Channel Multi-Class Confusion Matrix**")
            ch_cm = np.array(ch_metrics["confusion_matrix"])
            ch_labels = ch_metrics.get("labels", ["phishing", "smishing", "valid"])
            fig_ch_cm = px.imshow(
                ch_cm,
                text_auto=True,
                x=ch_labels,
                y=ch_labels,
                color_continuous_scale="Blues",
                labels=dict(x="Predicted Channel", y="True Channel", color="Count"),
            )
            fig_ch_cm.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#f1f5f9"),
                height=300,
                margin=dict(t=20, b=30, l=40, r=20),
            )
            st.plotly_chart(fig_ch_cm, use_container_width=True)

    # ── Level 4: Computational Overhead & Resource Profiling (Thesis Focus) ──
    st.markdown('<div class="section-title">Computational Overhead & Resource Profiling</div>', unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.metric("Mean Latency", f"{res['avg_latency_s']:.3f}s")
    with r2:
        st.metric("Median Latency", f"{res['median_latency_s']:.3f}s")
    with r3:
        st.metric("95th Percentile (P95)", f"{res['p95_latency_s']:.3f}s")
    with r4:
        st.metric("Throughput", f"{res['throughput_msgs_per_min']:.1f} msg/min")

    r5, r6, r7, r8 = st.columns(4)
    with r5:
        st.metric("Model Storage Size", f"{res['model_size_mb']:.1f} MB")
    with r6:
        st.metric("Total Execution Time", f"{res['total_time_s']:.1f}s")
    with r7:
        st.metric("Minimum Latency", f"{res['min_latency_s']:.3f}s")
    with r8:
        st.metric("Maximum Latency", f"{res['max_latency_s']:.3f}s")

    # Latency histogram
    raw_preds = report.get("raw_predictions", [])
    if raw_preds:
        latencies = [p["latency_s"] for p in raw_preds if "latency_s" in p]
        fig_lat = px.histogram(
            x=latencies,
            nbins=25,
            title="Inference Latency Distribution (seconds)",
            labels={"x": "Latency (seconds)", "y": "Frequency"},
            color_discrete_sequence=["#3b82f6"],
        )
        fig_lat.add_vline(
            x=res["median_latency_s"],
            line_dash="dash",
            line_color="#10b981",
            annotation_text="Median",
            annotation_position="top left",
        )
        fig_lat.add_vline(
            x=res["p95_latency_s"],
            line_dash="dot",
            line_color="#f59e0b",
            annotation_text="P95",
            annotation_position="top right",
        )
        fig_lat.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#94a3b8"),
            height=280,
            margin=dict(t=40, b=30, l=40, r=20),
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b"),
        )
        st.plotly_chart(fig_lat, use_container_width=True)

    # ── Level 5: Audit Trail & Export ──
    with st.expander("Detailed Inference Records (Audit Log)", expanded=False):
        if raw_preds:
            pred_table = pd.DataFrame(raw_preds)
            # Standardize columns
            if "is_correct_binary" not in pred_table.columns and "true" in pred_table.columns:
                pred_table["is_correct_binary"] = pred_table.apply(
                    lambda r: is_binary_match(r.get("true", ""), r.get("predicted", "")),
                    axis=1,
                )
            st.dataframe(pred_table, use_container_width=True, height=350)

    with st.expander("Full Scikit-Learn Report Output", expanded=False):
        st.code(cls.get("classification_report_text", "N/A"), language="text")


# ──────────────────────────────────────────────
# Main Application Flow
# ──────────────────────────────────────────────
def main():
    selected_model, dataset_choice, sample_size, num_gpu, mode_display_name = render_sidebar()

    # Main Header
    st.markdown(
        """
        <div class="platform-header">
            <h1 class="platform-title">MarkerPhish</h1>
            <p class="platform-subtitle">
                Automated evaluation platform for large language models in text-based social engineering attack detection
                (phishing & smishing) with concurrent hardware resource profiling.
            </p>
            <span class="academic-tag">Engineering Thesis Research System | Academic & Business Benchmark</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not selected_model:
        st.info("Select a model and configure parameters in the sidebar to initialize the evaluation session.")
        return

    tab_data, tab_eval, tab_results = st.tabs(
        [
            "Dataset Overview",
            "Run Evaluation",
            "Results & Analytics",
        ]
    )

    # ────────────── TAB 1: DATA OVERVIEW ──────────────
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

    # ────────────── TAB 3: RESULTS & ANALYTICS ──────────────
    with tab_results:
        available_files = list_saved_result_files()

        if not available_files:
            st.info(
                "No saved evaluation reports found in the results/ directory. "
                "Execute an evaluation run in the Run Evaluation tab to generate benchmark data."
            )
        else:
            # Maintain active selection in session state
            current_selected = st.session_state.get("selected_result_file")
            if current_selected not in available_files:
                current_selected = available_files[0]
                st.session_state["selected_result_file"] = current_selected

            default_idx = available_files.index(current_selected)

            col_select, col_refresh = st.columns([4, 1])
            with col_select:
                selected_file = st.selectbox(
                    "Select Evaluation Report",
                    options=available_files,
                    index=default_idx,
                    format_func=format_result_label,
                    help="Select a benchmark report from the results/ directory to inspect.",
                )

            with col_refresh:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                if st.button("Refresh Archive", icon=":material/refresh:", use_container_width=True):
                    st.rerun()

            # Dynamically load newly selected report if changed or not yet loaded
            if (
                selected_file != st.session_state.get("loaded_filename")
                or st.session_state.get("report") is None
            ):
                loaded = load_result_by_filename(selected_file)
                if loaded:
                    st.session_state["report"] = loaded
                    st.session_state["selected_result_file"] = selected_file
                    st.session_state["loaded_filename"] = selected_file

            report = st.session_state.get("report")
            if report:
                env_str = report.get("execution_mode", "CPU Safe Mode")
                target_model = report.get("model_name", "Unknown Model")

                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; margin: 12px 0 20px 0; padding: 10px 16px; background: #131d2e; border: 1px solid #1e293b; border-radius: 6px;">
                        <div>
                            <span style="font-size: 0.95rem; font-weight: 600; color: #f8fafc;">Active Report: <code>{selected_file}</code></span>
                        </div>
                        <div style="font-size: 0.85rem; color: #94a3b8;">
                            Model: <strong style="color: #f1f5f9;">{target_model}</strong> | Environment: <strong style="color: #38bdf8;">{env_str}</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                render_results(report)

                st.divider()
                clean_name = target_model.replace(":", "_").replace("/", "_")
                st.download_button(
                    "Export Active Report (JSON)",
                    icon=":material/download:",
                    data=json.dumps(report, indent=2, default=str),
                    file_name=f"markerphish_report_{clean_name}.json",
                    mime="application/json",
                )


if __name__ == "__main__":
    main()
