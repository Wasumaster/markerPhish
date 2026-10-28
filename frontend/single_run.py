"""
MarkerPhish — Detailed Single-Run Analysis Component
=====================================================
Comprehensive analysis for an individual benchmark run: primary binary metrics,
visual confusion matrices, channel diagnostics, latency profiling, audit logs,
and hardware execution telemetry.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from backend.utils import (
    ensure_report_compatibility,
    format_result_label,
    list_saved_result_files,
    load_result_by_filename,
)
from config import is_binary_match


def render_results(report: dict) -> None:
    """
    Render all analytical tiers for a single evaluation report.

    Args:
        report: Dictionary containing classification_metrics, resource_metrics,
                data_distribution, and optionally raw_predictions & hardware_profile.
    """
    report = ensure_report_compatibility(report)
    cls = report.get("classification_metrics", {})
    res = report.get("resource_metrics", {})
    dist = report.get("data_distribution", {})

    # Warning alert if all predictions failed
    if cls.get("unknown_predictions") == cls.get("total_evaluated") and cls.get("total_evaluated", 0) > 0:
        st.error(
            "All model inferences returned 'unknown' or timed out. "
            "If using Pascal architecture GPUs, ensure 'CPU Safe Mode' is enabled in the sidebar."
        )

    # ──────────────────────────────────────────────
    # Level 1: Primary Binary Threat Detection Performance
    # ──────────────────────────────────────────────
    st.markdown('<div class="section-title">Binary Threat Detection Performance</div>', unsafe_allow_html=True)

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.metric(
            "Binary Accuracy",
            f"{cls.get('accuracy', 0.0):.2%}",
            help="Overall proportion of correct classifications (Threats vs. Safe messages)",
        )
    with b2:
        st.metric(
            "Threat Precision",
            f"{cls.get('precision', 0.0):.2%}",
            help="Of messages flagged as threats, percentage that were genuinely malicious (TP / (TP + FP))",
        )
    with b3:
        st.metric(
            "Threat Recall (Detection Rate)",
            f"{cls.get('recall', 0.0):.2%}",
            help="Of all actual threats, percentage successfully intercepted (TP / (TP + FN))",
        )
    with b4:
        st.metric(
            "Threat F1-Score",
            f"{cls.get('f1', 0.0):.2%}",
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
            f"{cls.get('unknown_predictions', 0)}",
            help="Number of requests that timed out or generated unparsable output",
        )

    # ──────────────────────────────────────────────
    # Level 2: Visual Diagnostics (Confusion Matrix & Distributions)
    # ──────────────────────────────────────────────
    col_cm, col_dist = st.columns([1.2, 1.0])

    with col_cm:
        st.markdown("#### Binary Confusion Matrix")
        binary_cm = cls.get("binary_confusion_matrix")
        if binary_cm:
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
            st.plotly_chart(fig_cm, width="stretch")
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
            st.plotly_chart(fig_dist, width="stretch")

    # ──────────────────────────────────────────────
    # Level 3: Distribution-Channel Breakdown
    # ──────────────────────────────────────────────
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
                width="stretch",
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
            st.plotly_chart(fig_ch_cm, width="stretch")

    # ──────────────────────────────────────────────
    # Level 4: Computational Overhead & Resource Profiling
    # ──────────────────────────────────────────────
    st.markdown('<div class="section-title">Computational Overhead & Resource Profiling</div>', unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.metric("Mean Latency", f"{res.get('avg_latency_s', 0.0):.3f}s")
    with r2:
        st.metric("Median Latency", f"{res.get('median_latency_s', 0.0):.3f}s")
    with r3:
        st.metric("95th Percentile (P95)", f"{res.get('p95_latency_s', 0.0):.3f}s")
    with r4:
        st.metric("Throughput", f"{res.get('throughput_msgs_per_min', 0.0):.1f} msg/min")

    r5, r6, r7, r8 = st.columns(4)
    with r5:
        st.metric("Model Storage Size", f"{res.get('model_size_mb', 0.0):.1f} MB")
    with r6:
        st.metric("Total Execution Time", f"{res.get('total_time_s', 0.0):.1f}s")
    with r7:
        st.metric("Minimum Latency", f"{res.get('min_latency_s', 0.0):.3f}s")
    with r8:
        st.metric("Maximum Latency", f"{res.get('max_latency_s', 0.0):.3f}s")

    # Latency histogram
    raw_preds = report.get("raw_predictions", [])
    if raw_preds:
        latencies = [p["latency_s"] for p in raw_preds if "latency_s" in p]
        if latencies:
            fig_lat = px.histogram(
                x=latencies,
                nbins=25,
                title="Inference Latency Distribution (seconds)",
                labels={"x": "Latency (seconds)", "y": "Frequency"},
                color_discrete_sequence=["#3b82f6"],
            )
            fig_lat.add_vline(
                x=res.get("median_latency_s", 0.0),
                line_dash="dash",
                line_color="#10b981",
                annotation_text="Median",
                annotation_position="top left",
            )
            fig_lat.add_vline(
                x=res.get("p95_latency_s", 0.0),
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
            st.plotly_chart(fig_lat, width="stretch")

    # ──────────────────────────────────────────────
    # Level 5: Audit Trail, Hardware & Export
    # ──────────────────────────────────────────────
    hw_prof = report.get("hardware_profile")
    if hw_prof:
        with st.expander("Host Hardware Profile During Benchmark", expanded=False):
            hw_col1, hw_col2 = st.columns(2)
            with hw_col1:
                st.markdown(f"**CPU:** {hw_prof.get('cpu_name', 'Unknown')}")
                st.markdown(
                    f"**Cores:** {hw_prof.get('cpu_cores_physical', 'N/A')} physical, "
                    f"{hw_prof.get('cpu_cores_logical', 'N/A')} logical"
                )
                st.markdown(f"**RAM:** {hw_prof.get('ram_total_gb', 0.0):.1f} GB total")
            with hw_col2:
                gpu_name = hw_prof.get('gpu_name') or "None / CPU Mode"
                st.markdown(f"**GPU:** {gpu_name}")
                if hw_prof.get("has_gpu"):
                    st.markdown(f"**VRAM:** {hw_prof.get('gpu_vram_mb', 0)} MB | Compute: {hw_prof.get('compute_cap', 'N/A')}")
                    st.markdown(f"**Driver:** {hw_prof.get('gpu_driver', 'N/A')}")

    with st.expander("Detailed Inference Records (Audit Log)", expanded=False):
        if raw_preds:
            pred_table = pd.DataFrame(raw_preds)
            if "is_correct_binary" not in pred_table.columns and "true" in pred_table.columns:
                pred_table["is_correct_binary"] = pred_table.apply(
                    lambda r: is_binary_match(r.get("true", ""), r.get("predicted", "")),
                    axis=1,
                )
            st.dataframe(pred_table, width="stretch", height=350)

    with st.expander("Full Scikit-Learn Report Output", expanded=False):
        st.code(cls.get("classification_report_text", "N/A"), language="text")


def render_single_run_view() -> None:
    """
    Render the complete single-run analysis tab view including the report selector,
    metrics inspection, and JSON export.
    """
    available_files = list_saved_result_files()

    if not available_files:
        st.info(
            "No saved evaluation reports found in the results/ directory. "
            "Execute an evaluation run in the Run Evaluation tab to generate benchmark data."
        )
        return

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
        if st.button("Refresh Archive", width="stretch"):
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
            data=json.dumps(report, indent=2, default=str),
            file_name=f"markerphish_report_{clean_name}.json",
            mime="application/json",
        )
