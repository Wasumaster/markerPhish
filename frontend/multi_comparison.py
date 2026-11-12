"""
MarkerPhish — Multi-Model Comparison Component
===============================================
Compares several evaluation runs: one summary table (best value per column in
bold), detection-vs-cost scatter plots, and per-class channel breakdowns.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend.utils import (
    ensure_report_compatibility,
    list_saved_result_files,
    load_result_by_filename,
    report_cold_start,
    report_prompt_variant,
)
from config import LEGACY_PROMPT_VARIANT
from ui.components import (
    DASH,
    best_flags,
    cell,
    data_table,
    esc,
    fmt_int,
    fmt_mb,
    fmt_pct,
    fmt_rate,
    fmt_seconds,
    marker_legend,
    page_title,
    plot,
    run_chip_label,
    run_timestamp,
    section_label,
    section_title,
)
from ui.theme import CLASS_COLORS, FONT_MONO, INK, SURFACE


def _extract_report_summary(filename: str, report: dict) -> dict:
    """Extract standard summary metrics from an evaluation report."""
    report = ensure_report_compatibility(report)
    cls = report.get("classification_metrics", {})
    res = report.get("resource_metrics", {})
    hw = report.get("hardware_profile", {})

    model_name = report.get("model_name", "Unknown Model")
    exec_mode = report.get("execution_mode", "CPU Safe Mode")
    prompt_variant = report_prompt_variant(report)

    # Channel metrics
    ch_metrics = cls.get("channel_metrics", {}).get("per_class", {})
    phishing_stats = ch_metrics.get("phishing", {})
    smishing_stats = ch_metrics.get("smishing", {})
    valid_stats = ch_metrics.get("valid", {})

    return {
        "filename": filename,
        "model_name": model_name,
        "prompt_variant": prompt_variant,
        "model_variant": f"{model_name} [{prompt_variant}]",
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
        "cold_start_s": report_cold_start(report),
        # Not recorded by the evaluator yet; shown as "—" when absent
        "peak_ram_mb": res.get("peak_ram_mb"),
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


def _summary_table(comp_df: pd.DataFrame) -> None:
    """One row per run; the best value in each metric column is set in bold."""
    # (header, column key, formatter, preference for bold or None)
    spec = [
        ("F1", "f1", fmt_pct, "max"),
        ("recall", "recall", fmt_pct, "max"),
        ("precision", "precision", fmt_pct, "max"),
        ("accuracy", "accuracy", fmt_pct, "max"),
        ("specificity", "specificity", fmt_pct, "max"),
        ("false alarm", "false_alarm_rate", fmt_pct, "min"),
        ("p95 latency", "p95_latency_s", fmt_seconds, "min"),
        ("mean latency", "avg_latency_s", fmt_seconds, "min"),
        ("throughput", "throughput_msgs_per_min", fmt_rate, "max"),
        ("cold start", "cold_start_s", fmt_seconds, "min"),
        ("peak RAM", "peak_ram_mb", fmt_mb, "min"),
        ("size", "model_size_mb", fmt_mb, "min"),
        ("unknown", "unknown_predictions", fmt_int, "min"),
        ("messages", "total_evaluated", fmt_int, None),
    ]
    best = {key: best_flags(comp_df[key].tolist(), pref) for _, key, _, pref in spec if pref}

    headers = ["model", "run", "prompt"] + [h for h, *_ in spec] + ["hardware"]
    rows = []
    for i, row in enumerate(comp_df.itertuples(index=False)):
        legacy = row.prompt_variant == LEGACY_PROMPT_VARIANT
        model_html = esc(row.model_name) + (' <span class="mp-faint">legacy</span>' if legacy else "")
        cells = [
            cell(model_html, raw=True),
            run_timestamp(row.filename) or DASH,
            DASH if legacy else row.prompt_variant,
        ]
        for _, key, fmt, pref in spec:
            value = getattr(row, key)
            cells.append(cell(fmt(value), bold=bool(pref and best[key][i])))
        cells.append(row.execution_mode)
        rows.append(cells)

    numeric = set(range(3, 3 + len(spec)))
    for r in rows:
        r[0]["html"] = f'<span class="nowrap">{r[0]["html"]}</span>'
    data_table(headers, rows, numeric=numeric)


def render_multi_comparison_view() -> None:
    """
    Render the multi-run benchmark comparison: run selection, summary table,
    trade-off charts and per-class breakdown.
    """
    page_title(
        "Compare runs",
        "Detection quality against cost, for the runs you pick. Best value in each column is set in bold.",
    )

    available_files = list_saved_result_files()
    if not available_files:
        st.info("No saved evaluation reports found in the results/ directory to compare.")
        return

    # Load every report once: used for chip labels and for the selected runs
    reports = {f: load_result_by_filename(f) for f in available_files}
    chip_labels = {
        f: run_chip_label(
            f,
            (rep or {}).get("model_name"),
            report_prompt_variant(rep or {}),
            LEGACY_PROMPT_VARIANT,
        )
        for f, rep in reports.items()
    }

    # ──────────────────────────────────────────────
    # 1. Run selection
    # ──────────────────────────────────────────────
    ctrl_col1, ctrl_col2 = st.columns([6, 1], vertical_alignment="bottom")
    with ctrl_col1:
        # Default to all available benchmark files (or the latest 6+ if many exist)
        default_selection = available_files[: max(6, len(available_files))]
        selected_files = st.multiselect(
            "runs",
            options=available_files,
            default=default_selection,
            format_func=lambda f: chip_labels.get(f, f),
            help="Select 2 or more evaluation reports to generate comparative analytics. Supports 6+ runs concurrently.",
        )

    with ctrl_col2:
        if st.button("select all", type="tertiary"):
            st.session_state["multi_selected_files"] = available_files
            st.rerun()

    if not selected_files or len(selected_files) < 1:
        st.warning("Select at least one benchmark report to inspect metrics.")
        return

    # Compile all selected reports
    compiled_data = []
    for f in selected_files:
        rep = reports.get(f)
        if rep:
            compiled_data.append(_extract_report_summary(f, rep))

    if not compiled_data:
        st.error("Failed to load data for selected evaluation files.")
        return

    comp_df = pd.DataFrame(compiled_data)

    # Short label: Model [variant] + run timestamp
    def make_run_label(row: pd.Series) -> str:
        fn = row["filename"]
        stem = fn[:-5] if fn.endswith(".json") else fn
        parts = stem.split("_")
        ts = f"{parts[-2]}_{parts[-1]}" if len(parts) >= 3 else stem
        return f"{row['model_variant']} ({ts})"

    comp_df["run_label"] = comp_df.apply(make_run_label, axis=1)

    # ──────────────────────────────────────────────
    # 2. Summary table (replaces the KPI cards and the summary matrix)
    # ──────────────────────────────────────────────
    _summary_table(comp_df)

    # CSV export of the comparison matrix for external thesis analysis
    matrix_cols = [
        "run_label",
        "prompt_variant",
        "execution_mode",
        "model_size_mb",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "p95_latency_s",
        "throughput_msgs_per_min",
        "cold_start_s",
    ]
    display_matrix = comp_df[matrix_cols].copy()
    display_matrix.columns = [
        "Model / Run",
        "Prompt Variant",
        "Hardware Mode",
        "Size (MB)",
        "Accuracy",
        "Threat Precision",
        "Threat Recall",
        "Threat F1",
        "Safe Specificity",
        "P95 Latency (s)",
        "Throughput (msg/min)",
        "Cold Start (s)",
    ]
    csv_data = display_matrix.to_csv(index=False, na_rep="n/a").encode("utf-8")
    st.download_button(
        "Export comparison (CSV)",
        data=csv_data,
        file_name="markerphish_multi_model_benchmark.csv",
        mime="text/csv",
    )

    # ──────────────────────────────────────────────
    # 3. Trade-off charts
    # ──────────────────────────────────────────────
    chart_col1, chart_col2 = st.columns(2, gap="large")
    max_size = max(comp_df["model_size_mb"].max(), 1.0)

    with chart_col1:
        # Threat F1 vs. P95 latency (marker size = model size)
        fig_f1_lat = go.Figure(
            go.Scatter(
                x=comp_df["p95_latency_s"],
                y=comp_df["f1"],
                mode="markers+text",
                text=comp_df["model_variant"],
                textposition="top center",
                textfont=dict(family=FONT_MONO, size=11, color=INK),
                marker=dict(
                    color=INK,
                    size=8 + 16 * comp_df["model_size_mb"] / max_size,
                    line=dict(width=1, color=SURFACE),
                ),
                customdata=comp_df[["execution_mode", "model_size_mb", "throughput_msgs_per_min"]],
                hovertemplate=(
                    "%{text}<br>F1 %{y:.2%} · p95 %{x:.2f} s<br>"
                    "%{customdata[0]} · %{customdata[1]:.1f} MB · %{customdata[2]:.1f} msg/min<extra></extra>"
                ),
                cliponaxis=False,
            )
        )
        fig_f1_lat.update_layout(xaxis_title="p95 latency (s)", yaxis_title="threat F1")
        fig_f1_lat.update_yaxes(tickformat=".0%")
        plot(fig_f1_lat, caption="F1 vs p95 latency — up and to the left is better", height=380)

    with chart_col2:
        # Throughput vs. model storage footprint (marker size = accuracy); one ink series per mode
        modes = list(dict.fromkeys(comp_df["execution_mode"]))
        traces = []
        for idx, mode in enumerate(modes):
            sub = comp_df[comp_df["execution_mode"] == mode]
            traces.append(
                go.Scatter(
                    name=mode,
                    x=sub["model_size_mb"],
                    y=sub["throughput_msgs_per_min"],
                    mode="markers+text",
                    text=sub["model_variant"],
                    textposition="top center",
                    textfont=dict(family=FONT_MONO, size=11, color=INK),
                    marker=dict(
                        color=INK,
                        symbol="circle" if idx == 0 else "circle-open",
                        size=8 + 16 * sub["accuracy"],
                        line=dict(width=1.5, color=INK if idx else SURFACE),
                    ),
                    customdata=sub[["accuracy"]],
                    hovertemplate=(
                        "%{text}<br>%{y:.1f} msg/min · %{x:.1f} MB<br>accuracy %{customdata[0]:.2%}"
                        f"<br>{mode}<extra></extra>"
                    ),
                    cliponaxis=False,
                )
            )
        fig_tput_size = go.Figure(traces)
        fig_tput_size.update_layout(
            xaxis_title="model size on disk (MB)",
            yaxis_title="throughput (msg/min)",
            showlegend=len(modes) > 2,
        )
        plot(fig_tput_size, caption="throughput vs model size — up and to the left is better", height=380)
        if 1 < len(modes) <= 2:
            marker_legend([(m, i == 0) for i, m in enumerate(modes)])

    # ──────────────────────────────────────────────
    # 4. Per-class performance breakdown
    # ──────────────────────────────────────────────
    section_title(
        "By class",
        "Channel-level F1 per run: phishing (email), smishing (SMS) and valid messages.",
    )

    class_series = [
        ("phishing", "phishing_f1", CLASS_COLORS["phishing"]),
        ("smishing", "smishing_f1", CLASS_COLORS["smishing"]),
        ("valid", "valid_f1", CLASS_COLORS["valid"]),
    ]
    fig_class = go.Figure(
        [
            go.Bar(
                name=name,
                x=comp_df["run_label"],
                y=comp_df[key],
                marker=dict(color=color, line=dict(width=0)),
                hovertemplate=f"%{{x}}<br>{name} F1 %{{y:.2%}}<extra></extra>",
            )
            for name, key, color in class_series
        ]
    )
    fig_class.update_layout(barmode="group", yaxis_title="F1", showlegend=True, bargroupgap=0.05)
    fig_class.update_yaxes(tickformat=".0%")
    fig_class.update_xaxes(tickangle=-15)
    plot(fig_class, caption="per-class F1", height=360)

    with st.expander("per-class data table", expanded=False):
        section_label("precision / recall / F1 per channel")
        data_table(
            [
                "run",
                "phishing P", "phishing R", "phishing F1",
                "smishing P", "smishing R", "smishing F1",
                "valid P", "valid R", "valid F1",
            ],
            [
                [
                    row.run_label,
                    fmt_pct(row.phishing_precision), fmt_pct(row.phishing_recall), fmt_pct(row.phishing_f1),
                    fmt_pct(row.smishing_precision), fmt_pct(row.smishing_recall), fmt_pct(row.smishing_f1),
                    fmt_pct(row.valid_precision), fmt_pct(row.valid_recall), fmt_pct(row.valid_f1),
                ]
                for row in comp_df.itertuples(index=False)
            ],
            numeric=set(range(1, 10)),
        )
