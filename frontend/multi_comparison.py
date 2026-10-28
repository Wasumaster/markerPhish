"""
MarkerPhish — Multi-Model Comparison Component
===============================================
Enables concurrent benchmark comparison across multiple evaluation runs (supporting 6+
models simultaneously). Features a dense summary metrics matrix, interactive Plotly
trade-off visualisations, and per-class channel breakdowns.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backend.utils import (
    ensure_report_compatibility,
    format_result_label,
    list_saved_result_files,
    load_result_by_filename,
)


def _extract_report_summary(filename: str, report: dict) -> dict:
    """Extract standard summary metrics from an evaluation report."""
    report = ensure_report_compatibility(report)
    cls = report.get("classification_metrics", {})
    res = report.get("resource_metrics", {})
    hw = report.get("hardware_profile", {})

    model_name = report.get("model_name", "Unknown Model")
    exec_mode = report.get("execution_mode", "CPU Safe Mode")

    # Channel metrics
    ch_metrics = cls.get("channel_metrics", {}).get("per_class", {})
    phishing_stats = ch_metrics.get("phishing", {})
    smishing_stats = ch_metrics.get("smishing", {})
    valid_stats = ch_metrics.get("valid", {})

    return {
        "filename": filename,
        "model_name": model_name,
        "execution_mode": exec_mode,
        "model_size_mb": float(res.get("model_size_mb", 0.0)),
        # Primary Binary Detection Metrics
        "accuracy": float(cls.get("accuracy", 0.0)),
        "precision": float(cls.get("precision", 0.0)),
        "recall": float(cls.get("recall", 0.0)),
        "f1": float(cls.get("f1", 0.0)),
        "specificity": float(cls.get("specificity", 0.0)),
        "miss_rate": float(cls.get("miss_rate", 0.0)),
        "false_alarm_rate": float(cls.get("false_alarm_rate", 0.0)),
        # Resource & Overhead Metrics
        "avg_latency_s": float(res.get("avg_latency_s", 0.0)),
        "median_latency_s": float(res.get("median_latency_s", 0.0)),
        "p95_latency_s": float(res.get("p95_latency_s", 0.0)),
        "throughput_msgs_per_min": float(res.get("throughput_msgs_per_min", 0.0)),
        "total_time_s": float(res.get("total_time_s", 0.0)),
        "total_evaluated": int(cls.get("total_evaluated", 0)),
        "unknown_predictions": int(cls.get("unknown_predictions", 0)),
        # Channel-Level Breakdown
        "phishing_precision": float(phishing_stats.get("precision", 0.0)),
        "phishing_recall": float(phishing_stats.get("recall", 0.0)),
        "phishing_f1": float(phishing_stats.get("f1-score", 0.0)),
        "smishing_precision": float(smishing_stats.get("precision", 0.0)),
        "smishing_recall": float(smishing_stats.get("recall", 0.0)),
        "smishing_f1": float(smishing_stats.get("f1-score", 0.0)),
        "valid_precision": float(valid_stats.get("precision", 0.0)),
        "valid_recall": float(valid_stats.get("recall", 0.0)),
        "valid_f1": float(valid_stats.get("f1-score", 0.0)),
    }


def render_multi_comparison_view() -> None:
    """
    Render the comprehensive multi-model benchmark comparison dashboard.
    Supports loading, matrix tabling, trade-off plotting, and channel breakdowns
    for 6 or more benchmark runs simultaneously.
    """
    st.markdown('<div class="section-title">Multi-Model Comparative Benchmarking</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="comparison-subtitle">'
        "Cross-model performance analysis evaluating detection effectiveness against computational overhead."
        "</div>",
        unsafe_allow_html=True,
    )

    available_files = list_saved_result_files()
    if not available_files:
        st.info("No saved evaluation reports found in the results/ directory to compare.")
        return

    # ──────────────────────────────────────────────
    # 1. Multi-Select Run Selection (Supports 6+ concurrently)
    # ──────────────────────────────────────────────
    ctrl_col1, ctrl_col2 = st.columns([4, 1])
    with ctrl_col1:
        # Default to all available benchmark files (or the latest 6+ if many exist)
        default_selection = available_files[: max(6, len(available_files))]
        selected_files = st.multiselect(
            "Select Evaluation Runs for Comparative Benchmark",
            options=available_files,
            default=default_selection,
            format_func=format_result_label,
            help="Select 2 or more evaluation reports to generate comparative analytics. Supports 6+ runs concurrently.",
        )

    with ctrl_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("Select All Runs", width="stretch"):
            st.session_state["multi_selected_files"] = available_files
            st.rerun()

    if not selected_files or len(selected_files) < 1:
        st.warning("Select at least one benchmark report to inspect metrics.")
        return

    # Load and compile all selected reports
    compiled_data = []
    for f in selected_files:
        rep = load_result_by_filename(f)
        if rep:
            summary = _extract_report_summary(f, rep)
            compiled_data.append(summary)

    if not compiled_data:
        st.error("Failed to load data for selected evaluation files.")
        return

    comp_df = pd.DataFrame(compiled_data)

    # Short label for cleaner plots: Model Name + File Suffix
    def make_run_label(row: pd.Series) -> str:
        fn = row["filename"]
        # Extract timestamp part if present
        stem = fn[:-5] if fn.endswith(".json") else fn
        parts = stem.split("_")
        ts = f"{parts[-2]}_{parts[-1]}" if len(parts) >= 3 else stem
        return f"{row['model_name']} ({ts})"

    comp_df["run_label"] = comp_df.apply(make_run_label, axis=1)

    # ──────────────────────────────────────────────
    # 2. Executive KPI Summary Cards
    # ──────────────────────────────────────────────
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    best_f1_row = comp_df.loc[comp_df["f1"].idxmax()]
    best_rec_row = comp_df.loc[comp_df["recall"].idxmax()]
    best_p95_row = comp_df.loc[comp_df["p95_latency_s"].idxmin()]
    best_tput_row = comp_df.loc[comp_df["throughput_msgs_per_min"].idxmax()]

    with kpi1:
        st.metric(
            "Highest Threat F1-Score",
            f"{best_f1_row['f1']:.2%}",
            f"{best_f1_row['model_name']}",
        )
    with kpi2:
        st.metric(
            "Highest Threat Recall",
            f"{best_rec_row['recall']:.2%}",
            f"{best_rec_row['model_name']}",
        )
    with kpi3:
        st.metric(
            "Lowest P95 Latency",
            f"{best_p95_row['p95_latency_s']:.2f}s",
            f"{best_p95_row['model_name']}",
        )
    with kpi4:
        st.metric(
            "Peak Throughput",
            f"{best_tput_row['throughput_msgs_per_min']:.1f} msg/m",
            f"{best_tput_row['model_name']}",
        )

    # ──────────────────────────────────────────────
    # 3. Summary Metrics Matrix (Dense Side-by-Side Table)
    # ──────────────────────────────────────────────
    st.markdown('<div class="section-title">Summary Metrics Matrix</div>', unsafe_allow_html=True)
    st.caption("Dense cross-model comparison of classification effectiveness and computational overhead.")

    matrix_cols = [
        "run_label",
        "execution_mode",
        "model_size_mb",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "p95_latency_s",
        "throughput_msgs_per_min",
    ]

    display_matrix = comp_df[matrix_cols].copy()
    display_matrix.columns = [
        "Model / Run",
        "Hardware Mode",
        "Size (MB)",
        "Accuracy",
        "Threat Precision",
        "Threat Recall",
        "Threat F1",
        "Safe Specificity",
        "P95 Latency (s)",
        "Throughput (msg/min)",
    ]

    st.dataframe(
        display_matrix.style.format(
            {
                "Size (MB)": "{:.1f}",
                "Accuracy": "{:.2%}",
                "Threat Precision": "{:.2%}",
                "Threat Recall": "{:.2%}",
                "Threat F1": "{:.2%}",
                "Safe Specificity": "{:.2%}",
                "P95 Latency (s)": "{:.2f}",
                "Throughput (msg/min)": "{:.1f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    # CSV Download for external thesis analysis
    csv_data = display_matrix.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Export Comparison Matrix (CSV)",
        data=csv_data,
        file_name="markerphish_multi_model_benchmark.csv",
        mime="text/csv",
    )

    # ──────────────────────────────────────────────
    # 4. Interactive Plotly Trade-off Charts
    # ──────────────────────────────────────────────
    st.markdown('<div class="section-title">Resource vs. Accuracy Trade-off Analysis</div>', unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Trade-off 1: Threat F1-Score vs. P95 Latency
        fig_f1_lat = px.scatter(
            comp_df,
            x="p95_latency_s",
            y="f1",
            text="model_name",
            color="model_name",
            size="model_size_mb",
            hover_data={
                "model_name": True,
                "execution_mode": True,
                "f1": ":.2%",
                "p95_latency_s": ":.2f",
                "model_size_mb": ":.1f",
                "throughput_msgs_per_min": ":.1f",
            },
            title="Threat F1-Score vs. P95 Latency (Bubble Size = Model Size)",
            labels={
                "p95_latency_s": "P95 Inference Latency (seconds) — Lower is better",
                "f1": "Threat F1-Score — Higher is better",
                "model_name": "Model",
            },
        )
        fig_f1_lat.update_traces(
            textposition="top center",
            marker=dict(line=dict(width=1, color="#1e293b")),
        )
        fig_f1_lat.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#94a3b8"),
            height=380,
            title_font_size=13,
            margin=dict(t=40, b=30, l=40, r=20),
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b", tickformat=".0%"),
            showlegend=False,
        )
        st.plotly_chart(fig_f1_lat, width="stretch")

    with chart_col2:
        # Trade-off 2: Throughput vs. Model Storage Size
        fig_tput_size = px.scatter(
            comp_df,
            x="model_size_mb",
            y="throughput_msgs_per_min",
            text="model_name",
            color="execution_mode",
            size="accuracy",
            hover_data={
                "model_name": True,
                "model_size_mb": ":.1f",
                "throughput_msgs_per_min": ":.1f",
                "accuracy": ":.2%",
                "execution_mode": True,
            },
            title="Throughput vs. Model Storage Footprint (Bubble Size = Accuracy)",
            labels={
                "model_size_mb": "Model Storage Footprint (MB) — Lower is more efficient",
                "throughput_msgs_per_min": "Throughput (messages / min) — Higher is better",
                "execution_mode": "Execution Environment",
            },
        )
        fig_tput_size.update_traces(
            textposition="top center",
            marker=dict(line=dict(width=1, color="#1e293b")),
        )
        fig_tput_size.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#94a3b8"),
            height=380,
            title_font_size=13,
            margin=dict(t=40, b=30, l=40, r=20),
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_tput_size, width="stretch")

    # ──────────────────────────────────────────────
    # 5. Per-Class Performance Breakdown (Phishing vs. Smishing vs. Valid)
    # ──────────────────────────────────────────────
    st.markdown('<div class="section-title">Per-Class Performance Breakdown</div>', unsafe_allow_html=True)
    st.caption("Distribution-channel diagnostic effectiveness: comparing model capabilities across specific attack vectors.")

    # Prepare melted DataFrame for grouped bar chart
    class_rows = []
    for _, row in comp_df.iterrows():
        class_rows.append({"Model": row["run_label"], "Channel": "Phishing (Email)", "F1-Score": row["phishing_f1"], "Recall": row["phishing_recall"]})
        class_rows.append({"Model": row["run_label"], "Channel": "Smishing (SMS)", "F1-Score": row["smishing_f1"], "Recall": row["smishing_recall"]})
        class_rows.append({"Model": row["run_label"], "Channel": "Valid (Legitimate)", "F1-Score": row["valid_f1"], "Recall": row["valid_recall"]})

    class_df = pd.DataFrame(class_rows)

    fig_class = px.bar(
        class_df,
        x="Model",
        y="F1-Score",
        color="Channel",
        barmode="group",
        title="Per-Class F1-Score Comparison Across Evaluated Models",
        color_discrete_map={
            "Phishing (Email)": "#f43f5e",
            "Smishing (SMS)": "#f59e0b",
            "Valid (Legitimate)": "#10b981",
        },
    )
    fig_class.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#94a3b8"),
        height=360,
        title_font_size=13,
        margin=dict(t=40, b=50, l=40, r=20),
        xaxis=dict(gridcolor="#1e293b", tickangle=-15),
        yaxis=dict(gridcolor="#1e293b", tickformat=".0%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_class, width="stretch")

    # Detailed Per-Class Matrix Table
    with st.expander("Detailed Per-Class Performance Data Table", expanded=False):
        detailed_cols = [
            "run_label",
            "phishing_precision",
            "phishing_recall",
            "phishing_f1",
            "smishing_precision",
            "smishing_recall",
            "smishing_f1",
            "valid_precision",
            "valid_recall",
            "valid_f1",
        ]
        det_df = comp_df[detailed_cols].copy()
        det_df.columns = [
            "Model / Run",
            "Phishing Precision",
            "Phishing Recall",
            "Phishing F1",
            "Smishing Precision",
            "Smishing Recall",
            "Smishing F1",
            "Valid Precision",
            "Valid Recall",
            "Valid F1",
        ]
        format_dict = {c: "{:.2%}" for c in det_df.columns if c != "Model / Run"}
        st.dataframe(det_df.style.format(format_dict), width="stretch", hide_index=True)
