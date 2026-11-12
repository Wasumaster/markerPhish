"""
MarkerPhish — Detailed Single-Run Analysis Component
=====================================================
Comprehensive analysis for an individual benchmark run: primary binary metrics,
confusion matrix, channel diagnostics, latency profiling, audit logs,
and hardware execution telemetry.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend.utils import (
    ensure_report_compatibility,
    format_result_label,
    list_saved_result_files,
    load_result_by_filename,
    report_cold_start,
    report_prompt_variant,
)
from config import LEGACY_PROMPT_VARIANT, is_binary_match
from ui.components import (
    DASH,
    cell,
    data_table,
    esc,
    fmt_int,
    fmt_mb,
    fmt_pct,
    fmt_rate,
    fmt_ratio,
    fmt_seconds,
    html_block,
    kv_table,
    legend_row,
    note,
    page_title,
    plot,
    section_label,
    section_title,
)
from ui.theme import FONT_MONO, INK, MALICIOUS, RULE, SAFE, SURFACE


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
    # Level 1: Binary threat detection metrics
    # ──────────────────────────────────────────────
    section_title("Detection", "Binary threat detection: malicious (phishing or smishing) vs. safe.")
    with st.container():
        section_label("metrics")
        data_table(
            ["metric", "value", "definition"],
            [
                ["binary accuracy", fmt_pct(cls.get("accuracy")), "Share of correct decisions (threats and safe messages)"],
                ["threat precision", fmt_pct(cls.get("precision")), "Of messages flagged as threats, share that were malicious: TP / (TP + FP)"],
                ["threat recall", fmt_pct(cls.get("recall")), "Of all actual threats, share intercepted: TP / (TP + FN)"],
                ["threat F1", fmt_pct(cls.get("f1")), "Harmonic mean of threat precision and recall"],
                ["safe specificity", fmt_pct(cls.get("specificity")), "Safe messages correctly passed: TN / (TN + FP)"],
                ["threat miss rate (FNR)", fmt_pct(cls.get("miss_rate")), "Threats let through as safe: FN / (TP + FN)"],
                ["false alarm rate (FPR)", fmt_pct(cls.get("false_alarm_rate")), "Safe messages flagged as threats: FP / (TN + FP)"],
                ["failed / unknown inferences", fmt_int(cls.get("unknown_predictions")), "Requests that timed out or gave unparsable output"],
            ],
            numeric={1},
            sans={2},
        )

    # ──────────────────────────────────────────────
    # Level 2: Confusion matrix & distributions
    # ──────────────────────────────────────────────
    col_cm, col_dist = st.columns([1.0, 1.2], gap="large")

    with col_cm:
        section_label("confusion matrix")
        binary_cm = cls.get("binary_confusion_matrix")
        if binary_cm:
            cm = np.array(binary_cm)
            data_table(
                ["true \\ predicted", "malicious", "safe"],
                [
                    [cell("malicious", color=MALICIOUS, bold=True), str(int(cm[0][0])), str(int(cm[0][1]))],
                    [cell("safe", color=SAFE, bold=True), str(int(cm[1][0])), str(int(cm[1][1]))],
                ],
                numeric={1, 2},
            )
            note(
                f"TP {cls.get('tp', DASH)} · TN {cls.get('tn', DASH)} · "
                f"FP {cls.get('fp', DASH)} · FN {cls.get('fn', DASH)}",
                mono=True,
            )
        else:
            note(DASH, mono=True)

    with col_dist:
        t_dist = dist.get("true_binary", {})
        p_dist = dist.get("predicted_binary", {})
        if t_dist and p_dist:
            groups = ["ground truth", "predicted"]
            fig_dist = go.Figure(
                [
                    go.Bar(
                        name="malicious",
                        x=groups,
                        y=[t_dist.get("malicious", 0), p_dist.get("malicious", 0)],
                        marker=dict(color=MALICIOUS, line=dict(width=0)),
                        text=[t_dist.get("malicious", 0), p_dist.get("malicious", 0)],
                        textposition="outside",
                        cliponaxis=False,
                    ),
                    go.Bar(
                        name="safe",
                        x=groups,
                        y=[t_dist.get("safe", 0), p_dist.get("safe", 0)],
                        marker=dict(color=SAFE, line=dict(width=0)),
                        text=[t_dist.get("safe", 0), p_dist.get("safe", 0)],
                        textposition="outside",
                        cliponaxis=False,
                    ),
                ]
            )
            fig_dist.update_layout(barmode="group", yaxis_title="messages", bargroupgap=0.08)
            plot(fig_dist, caption="ground truth vs predicted", height=300)
            legend_row([("malicious", MALICIOUS), ("safe", SAFE)])

    # ──────────────────────────────────────────────
    # Level 3: Distribution-channel breakdown
    # ──────────────────────────────────────────────
    with st.expander("channel breakdown (phishing / smishing / valid)", expanded=False):
        if report_prompt_variant(report) == LEGACY_PROMPT_VARIANT:
            st.caption("Legacy report: the model chose the channel label (phishing / smishing / valid) itself.")
        else:
            st.caption(
                "The model decides only malicious vs. valid. The channel (email / SMS) is derived from the "
                "dataset source, so a detected threat is labelled phishing for emails and smishing for SMS."
            )
        ch_metrics = cls.get("channel_metrics", {})
        if ch_metrics and "per_class" in ch_metrics:
            section_label("per channel")
            data_table(
                ["channel", "precision", "recall", "F1", "support"],
                [
                    [
                        name,
                        fmt_ratio(stats.get("precision")),
                        fmt_ratio(stats.get("recall")),
                        fmt_ratio(stats.get("f1-score")),
                        fmt_int(stats.get("support")),
                    ]
                    for name, stats in ch_metrics["per_class"].items()
                ],
                numeric={1, 2, 3, 4},
                max_width="640px",
            )

        if "confusion_matrix" in ch_metrics:
            ch_cm = np.array(ch_metrics["confusion_matrix"])
            ch_labels = ch_metrics.get("labels", ["phishing", "smishing", "valid"])
            fig_ch_cm = go.Figure(
                go.Heatmap(
                    z=ch_cm,
                    x=ch_labels,
                    y=ch_labels,
                    colorscale=[[0.0, SURFACE], [1.0, SAFE]],
                    showscale=False,
                    xgap=1,
                    ygap=1,
                    text=ch_cm,
                    texttemplate="%{text}",
                    textfont=dict(family=FONT_MONO, size=13, color=INK),
                    hovertemplate="true %{y} · predicted %{x}: %{z}<extra></extra>",
                )
            )
            fig_ch_cm.update_layout(xaxis_title="predicted channel", yaxis_title="true channel")
            fig_ch_cm.update_yaxes(autorange="reversed")
            plot(fig_ch_cm, caption="channel confusion matrix", height=300)

    # ──────────────────────────────────────────────
    # Level 4: Computational overhead
    # ──────────────────────────────────────────────
    section_title("Cost", "Latency per message and throughput; model loading is reported separately as cold start.")
    with st.container():
        section_label("computational overhead")
        data_table(
            ["metric", "value"],
            [
                ["mean latency", fmt_seconds(res.get("avg_latency_s"), 3)],
                ["median latency", fmt_seconds(res.get("median_latency_s"), 3)],
                ["p95 latency", fmt_seconds(res.get("p95_latency_s"), 3)],
                ["minimum latency", fmt_seconds(res.get("min_latency_s"), 3)],
                ["maximum latency", fmt_seconds(res.get("max_latency_s"), 3)],
                ["throughput", fmt_rate(res.get("throughput_msgs_per_min"))],
                ["total execution time", fmt_seconds(res.get("total_time_s"), 1)],
                ["cold start (model load)", fmt_seconds(report_cold_start(report))],
                ["model storage size", fmt_mb(res.get("model_size_mb"))],
            ],
            numeric={1},
            max_width="640px",
        )

    # Latency histogram
    raw_preds = report.get("raw_predictions", [])
    if raw_preds:
        latencies = [p["latency_s"] for p in raw_preds if "latency_s" in p]
        if latencies:
            fig_lat = go.Figure(
                go.Histogram(
                    x=latencies,
                    nbinsx=25,
                    marker=dict(color=INK, line=dict(color=SURFACE, width=1)),
                    hovertemplate="%{x} s: %{y}<extra></extra>",
                )
            )
            for value, label, position in (
                (res.get("median_latency_s"), "median", "top left"),
                (res.get("p95_latency_s"), "p95", "top right"),
            ):
                if value is not None:
                    fig_lat.add_vline(
                        x=value,
                        line_dash="dash",
                        line_color=RULE,
                        line_width=1,
                        annotation_text=f"{label} {value:.2f} s",
                        annotation_position=position,
                        annotation_font=dict(family=FONT_MONO, size=11, color=INK),
                    )
            fig_lat.update_layout(xaxis_title="latency (s)", yaxis_title="messages")
            plot(fig_lat, caption="inference latency distribution", height=280)

    # ──────────────────────────────────────────────
    # Level 5: Audit Trail, Hardware & Export
    # ──────────────────────────────────────────────
    hw_prof = report.get("hardware_profile")
    if hw_prof:
        with st.expander("host hardware profile during benchmark", expanded=False):
            pairs = [
                ("cpu", hw_prof.get("cpu_name", "Unknown")),
                (
                    "cores",
                    f"{hw_prof.get('cpu_cores_physical', DASH)} physical, "
                    f"{hw_prof.get('cpu_cores_logical', DASH)} logical",
                ),
                ("ram", f"{hw_prof.get('ram_total_gb', 0.0):.1f} GB total"),
                ("gpu", hw_prof.get("gpu_name") or "None / CPU Mode"),
            ]
            if hw_prof.get("has_gpu"):
                pairs += [
                    ("vram", f"{hw_prof.get('gpu_vram_mb', 0)} MB"),
                    ("compute", hw_prof.get("compute_cap", DASH)),
                    ("driver", hw_prof.get("gpu_driver", DASH)),
                ]
            kv_table(pairs, headers=("component", "value"))

    with st.expander("detailed inference records (audit log)", expanded=False):
        if raw_preds:
            pred_table = pd.DataFrame(raw_preds)
            if "is_correct_binary" not in pred_table.columns and "true" in pred_table.columns:
                pred_table["is_correct_binary"] = pred_table.apply(
                    lambda r: is_binary_match(r.get("true", ""), r.get("predicted", "")),
                    axis=1,
                )
            st.dataframe(pred_table, width="stretch", height=350)

    with st.expander("full scikit-learn report output", expanded=False):
        st.code(cls.get("classification_report_text", "N/A"), language="text")


def render_single_run_view() -> None:
    """
    Render the complete single-run analysis tab view including the report selector,
    metrics inspection, and JSON export.
    """
    page_title("Results", "Detection quality and computational cost of one saved evaluation run.")

    available_files = list_saved_result_files()

    if not available_files:
        st.info(
            "No saved evaluation reports found in the results/ directory. "
            "Execute an evaluation run in the Run tab to generate benchmark data."
        )
        return

    # Maintain active selection in session state
    current_selected = st.session_state.get("selected_result_file")
    if current_selected not in available_files:
        current_selected = available_files[0]
        st.session_state["selected_result_file"] = current_selected

    default_idx = available_files.index(current_selected)

    col_select, col_refresh = st.columns([4, 1], vertical_alignment="bottom")
    with col_select:
        selected_file = st.selectbox(
            "report",
            options=available_files,
            index=default_idx,
            format_func=format_result_label,
            help="Select a benchmark report from the results/ directory to inspect.",
        )

    with col_refresh:
        if st.button("Refresh", width="stretch"):
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
        variant_str = report_prompt_variant(report)

        with st.container():
            section_label("active report")
            html_block(
                '<p class="mp-sidebar-line" style="margin:0;">'
                f'<span class="mp-mono" style="color:{INK};">{esc(selected_file)}</span> · '
                f"model {esc(target_model)} · prompt {esc(variant_str)} · {esc(env_str)}</p>"
            )

        render_results(report)

        clean_name = target_model.replace(":", "_").replace("/", "_")
        st.download_button(
            "Export active report (JSON)",
            data=json.dumps(report, indent=2, default=str),
            file_name=f"markerphish_report_{clean_name}_{variant_str}.json",
            mime="application/json",
        )
