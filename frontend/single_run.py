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
from backend.manifest import (
    LATENCY_INFLATION_NOTE,
    current_environment,
    manifest_differences,
    manifest_summary,
    report_power_warning,
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
    fmt_ci,
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

    power_warning = report_power_warning(report)
    if power_warning:
        st.warning(power_warning)

    # ──────────────────────────────────────────────
    # Level 1: Binary threat detection metrics
    # ──────────────────────────────────────────────
    _render_detection(report.get("detection") or {})

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
    _render_timing(report)

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
    _render_resources(report.get("resources"))

    hw_prof = report.get("hardware") or report.get("hardware_profile")
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
            for key, label in (("os_version", "os"), ("python_version", "python"), ("ollama_version", "ollama"),
                               ("scanned_at", "scanned at")):
                if hw_prof.get(key):
                    pairs.append((label, hw_prof[key]))
            if hw_prof.get("ram_available_gb") is not None:
                pairs.insert(3, ("ram free at scan", f"{hw_prof['ram_available_gb']:.1f} GB"))
            kv_table(pairs, headers=("component", "value"))

    _render_manifest(report)

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


_METRIC_ROWS = [
    ("precision", "precision", "TP / (TP + FP)", True),
    ("recall", "recall", "TP / (TP + FN): share of threats caught", True),
    ("F1", "f1", "2 TP / (2 TP + FP + FN)", True),
    ("false positive rate", "fpr", "FP / (FP + TN): safe messages flagged", True),
    ("false negative rate", "fnr", "FN / (TP + FN): threats let through", False),
    ("specificity", "specificity", "TN / (TN + FP)", False),
    ("accuracy", "accuracy", "(TP + TN) / all evaluated", False),
]


def _render_detection(det: dict) -> None:
    """Binary detection in both unknown modes, overall and per channel, with bootstrap CIs."""
    boot = det.get("bootstrap") or {}
    level = int(round((boot.get("level") or 0.95) * 100))
    section_title(
        "Detection",
        "Malicious (phishing or smishing) vs. safe. Primary: unknown = error (a missed threat for a malicious "
        f"message, a false alarm for a safe one). Brackets: {level}% bootstrap CI "
        f"({boot.get('resamples', DASH)} resamples, stratified by class and channel"
        + (", group-aware" if boot.get("group_aware") else "") + ").",
    )
    overall = det.get("overall") or {}
    primary = overall.get("unknown_as_error") or {}
    excl = overall.get("excluding_unknown") or {}
    if not primary:
        note("No predictions to evaluate.")
        return
    rows = []
    for label, key, definition, has_ci in _METRIC_ROWS:
        rows.append([
            label,
            fmt_ci(primary.get(key), (primary.get("ci") or {}).get(key) if has_ci else None),
            fmt_ci(excl.get(key), (excl.get("ci") or {}).get(key) if has_ci else None),
            definition,
        ])
    unknown = fmt_int(det.get("unknown_predictions"))
    rows.append(["unknown predictions", f"{unknown} ({fmt_pct(det.get('unknown_rate'))})", f"{unknown} excluded",
                 "no parsable answer (or failed request)"])
    rows.append(["TP / FN / FP / TN",
                 f"{primary.get('tp')} / {primary.get('fn')} / {primary.get('fp')} / {primary.get('tn')}",
                 f"{excl.get('tp')} / {excl.get('fn')} / {excl.get('fp')} / {excl.get('tn')}", ""])
    with st.container():
        section_label("metrics")
        data_table(["metric", "unknown = error", "excluding unknown", "definition"], rows, numeric={1, 2}, sans={3})
        if det.get("excluded_unknown_ground_truth"):
            note(f"{det['excluded_unknown_ground_truth']} records without a ground-truth label are excluded.")

    per_channel = det.get("per_channel") or {}
    if not per_channel:
        return

    def channel_rows(mode: str) -> list:
        out = []
        for channel, modes in per_channel.items():
            m = modes.get(mode) or {}
            ci = m.get("ci") or {}
            out.append([
                channel,
                f"{fmt_int(m.get('support_malicious'))} / {fmt_int(m.get('support_safe'))}",
                fmt_ci(m.get("precision"), ci.get("precision")),
                fmt_ci(m.get("recall"), ci.get("recall")),
                fmt_ci(m.get("f1"), ci.get("f1")),
                fmt_ci(m.get("fpr"), ci.get("fpr")),
                fmt_ci(m.get("fnr")),
                fmt_int(m.get("unknown_predictions")),
            ])
        return out

    headers = ["channel", "malicious / safe", "precision", "recall", "F1", "FPR", "FNR", "unknown"]
    with st.container():
        section_label("per channel · unknown = error")
        data_table(headers, channel_rows("unknown_as_error"), numeric=set(range(1, 8)))
    with st.expander("per channel · excluding unknown", expanded=False):
        data_table(headers, channel_rows("excluding_unknown"), numeric=set(range(1, 8)))


def _render_timing(report: dict) -> None:
    """Latency over successful requests, failures, tokens/s and throughput."""
    t = report.get("timing") or {}
    lat = t.get("latency_s") or {}
    section_title(
        "Cost",
        "Latency per message over successful requests only (failed requests and the cold-start request are "
        "excluded). Throughput = messages / sum of request latencies; wall-clock throughput includes the UI.",
    )
    if report.get("latency_inflated"):
        st.warning(LATENCY_INFLATION_NOTE)
    failed = (
        f"{fmt_int(t.get('failed_requests'))} ({fmt_pct(t.get('failure_rate'))}) · "
        f"timeouts {fmt_int(t.get('timeouts'))} · errors {fmt_int(t.get('errors'))}"
    )
    rows = [
        ["mean latency", fmt_seconds(lat.get("mean"), 3)],
        ["median latency", fmt_seconds(lat.get("median"), 3)],
        ["p95 latency", fmt_seconds(lat.get("p95"), 3)],
        ["min / max latency", f"{fmt_seconds(lat.get('min'), 3)} / {fmt_seconds(lat.get('max'), 3)}"],
        ["throughput (inference time)", fmt_rate(t.get("throughput_msgs_per_min"))],
        ["throughput (wall clock)", fmt_rate(t.get("wall_throughput_msgs_per_min"))],
        ["failed requests", failed],
        ["prompt tokens (mean)", fmt_ratio(t.get("prompt_tokens_mean"), 1)],
        ["output tokens (mean)", fmt_ratio(t.get("output_tokens_mean"), 1)],
        ["prefill tokens/s", fmt_ratio(t.get("prefill_tokens_per_s"), 1)],
        ["generation tokens/s", fmt_ratio(t.get("generation_tokens_per_s"), 1)],
        ["total wall time", fmt_seconds(t.get("wall_time_s"), 1)],
        ["cold start (model load)", fmt_seconds(report_cold_start(report))],
    ]
    with st.container():
        section_label("computational overhead")
        data_table(["metric", "value"], rows, numeric={1}, max_width="640px")
        if t.get("note"):
            note(t["note"])


def _render_resources(res: dict | None) -> None:
    """Memory and CPU of the Ollama server during the run (runs with a resources block)."""
    section_title(
        "Resources",
        "Ollama server during the timed requests. Model memory is reported by Ollama (/api/ps); "
        "process RSS is secondary and noisy (memory-mapped weights, OS page cache).",
    )
    if not res:
        note("This run has no resource measurements (recorded since protocol 2).")
        return
    fits = res.get("fits_profile") or {}

    def pct(value) -> str:
        return DASH if value is None else f"{value:.1f} %"

    def fit_text(name: str) -> str:
        f = fits.get(name) or {}
        if f.get("fits_profile") is None:
            return DASH
        verdict = "fits" if f["fits_profile"] else "does not fit"
        return f"{verdict} (budget {fmt_mb(f.get('budget_mb'))})"

    rows = [
        ("model memory (Ollama)", fmt_mb(res.get("model_size_mb"))),
        ("model memory on GPU", fmt_mb(res.get("model_vram_mb"))),
        ("model file on disk", fmt_mb(res.get("model_disk_mb"))),
        ("peak process RSS", fmt_mb(res.get("peak_ram_mb"))),
        ("mean process RSS", fmt_mb(res.get("mean_ram_mb"))),
        ("RSS before first request (baseline)", fmt_mb(res.get("baseline_ram_mb"))),
        ("RSS before model load", fmt_mb(res.get("pre_warmup_ram_mb"))),
        ("peak / mean CPU (whole machine)", f"{pct(res.get('peak_cpu_pct'))} / {pct(res.get('mean_cpu_pct'))}"),
        ("start", f"{res.get('start_mode', DASH)} · cold start {fmt_seconds(res.get('cold_start_s'))}"),
        ("samples", f"{fmt_int(res.get('n_samples'))} every {res.get('sample_interval_s', DASH)} s · "
                    f"sampler cost {fmt_ratio(res.get('sampler_cost_ms'), 2)} ms"),
        ("models loaded before run", ", ".join(res.get("ollama_loaded_models") or []) or "none"),
    ]
    rows += [(f"fits {name}", fit_text(name)) for name in fits]
    if res.get("peak_vram_mb") is not None:
        rows.insert(2, ("peak VRAM (NVML)", fmt_mb(res.get("peak_vram_mb"))))
    kv_table(rows, headers=("measure", "value"))
    if res.get("other_models_loaded") and not res.get("other_models_unloaded"):
        st.warning("Other models were loaded in Ollama during this run; memory and latency include their effect.")


@st.cache_data(ttl=60, show_spinner=False)
def _current_environment_cached(model_name: str, prompt_variant: str, output_schema_json: str) -> dict:
    """Current code / model / preprocessing state (reads git and Ollama metadata; never loads a model)."""
    return current_environment(model_name, prompt_variant, json.loads(output_schema_json))


def _render_manifest(report: dict) -> None:
    """Collapsed manifest of the run and its differences from the current environment."""
    manifest = report.get("manifest")
    with st.expander("manifest", expanded=False):
        if not manifest:
            note(
                "This run has no manifest: it was produced under the legacy protocol (protocol 1), "
                "before dataset, sample, prompt, model and software versions were recorded."
            )
            return
        kv_table(manifest_summary(manifest), headers=("field", "value"), max_width="100%")

        prompt = manifest.get("prompt") or {}
        current = _current_environment_cached(
            report.get("model_name", ""),
            prompt.get("prompt_variant") or "",
            json.dumps(prompt.get("output_schema")),
        )
        diffs = manifest_differences(manifest, current)
        section_label("differences from the current environment")
        if diffs:
            data_table(
                ["field", "this run", "now"],
                [[field, str(run_value), str(now)] for field, run_value, now in diffs],
                max_width="100%",
            )
        else:
            note("Code commit, model digest, preprocessing version and prompt template match the current state.")

        templates = prompt.get("templates") or {}
        for channel, text in templates.items():
            section_label(f"prompt template · {channel}")
            st.code(text, language="text")


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
