"""
MarkerPhish — Compare tab
==========================
Compares saved runs: one table (best value per column in bold, only among runs
that may be best), the recommendation per constraint profile, trade-off
scatters and the legacy per-class breakdown. Rendering only: rows, rules and
analyses come from backend.compare and backend.optimum.
"""

from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend import utils as backend_utils
from backend.compare import best_values, build_comparison, hidden_reason, visible_runs
from backend.utils import ensure_report_compatibility, list_saved_result_files, load_result_by_filename
from config import CONSTRAINT_PROFILES
from frontend.recommendation import render_recommendation
from ui.components import (
    DASH,
    cell,
    data_table,
    esc,
    fmt_ci,
    fmt_int,
    fmt_mb,
    fmt_pct,
    fmt_rate,
    fmt_ratio,
    fmt_seconds,
    note,
    page_title,
    plot,
    run_chip_label,
    run_timestamp,
    section_label,
    section_title,
)
from ui.theme import CLASS_COLORS, INK, SURFACE

# Session-state key of the run multiselect.
_SELECTION_KEY = "compare_selected_files"
_SHOW_HIDDEN_KEY = "compare_show_hidden"


def select_all_runs(available_files: list[str]) -> None:
    """Button callback: select every available run in the Compare multiselect."""
    st.session_state[_SELECTION_KEY] = list(available_files)


@st.cache_data(show_spinner=False, max_entries=512)
def _load_report_cached(results_dir: str, filename: str, mtime: float) -> dict | None:
    """A loaded, compatibility-checked report; reloaded when the file changes."""
    report = load_result_by_filename(filename)
    return ensure_report_compatibility(report) if report else None


def load_reports(files: list[str]) -> dict[str, dict]:
    out = {}
    for f in files:
        path = os.path.join(backend_utils.RESULTS_DIR, f)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        rep = _load_report_cached(backend_utils.RESULTS_DIR, f, mtime)
        if rep:
            out[f] = rep
    return out


def _pct1(value) -> str:
    return DASH if value is None else f"{value:.1f} %"


def _fits_text(value) -> str:
    return DASH if value is None else ("yes" if value else "no")


def _summary_table(rows: list[dict], comparison: dict, profile_names: list[str]) -> None:
    """One row per run; bold = best value of the column among runs that may be best."""
    # (header, key, formatter, preference)
    spec = [
        ("F1 [95% CI]", "f1", None, "max"),
        ("recall", "recall", fmt_ratio, "max"),
        ("precision", "precision", fmt_ratio, "max"),
        ("FPR", "fpr", fmt_ratio, "min"),
        ("unknown", "unknown_rate", fmt_pct, "min"),
        ("p95 latency", "p95_latency_s", fmt_seconds, "min"),
        ("throughput", "throughput", fmt_rate, "max"),
        ("gen tok/s", "generation_tps", lambda v: fmt_ratio(v, 1), "max"),
        ("model memory", "model_memory_mb", fmt_mb, "min"),
        ("peak RSS", "peak_rss_mb", fmt_mb, "min"),
        ("mean CPU", "mean_cpu_pct", _pct1, "min"),
    ]
    if comparison["has_gpu"]:
        spec += [("peak VRAM", "peak_vram_mb", fmt_mb, "min"), ("GPU offload", "model_vram_mb", fmt_mb, None)]
    best = {key: best_values(rows, key, pref) for _, key, _, pref in spec if pref}
    latency_keys = {"p95_latency_s", "throughput", "generation_tps"}

    lead = ["model", "quantization", "prompt"]
    if comparison["differing_options"]:
        lead.append("options")
    lead += ["start", "n"]
    headers = lead + [h for h, *_ in spec] + [f"fits {p}" for p in profile_names]
    table_rows = []
    for i, row in enumerate(rows):
        marks = []
        if row["hidden"]:
            marks.append(row["hidden"])
        if row["baseline"]:
            marks.append("baseline · trained in-distribution")
        if not row["can_be_best"] and not row["hidden"]:
            marks.append("no valid prediction")
        model_html = esc(row["model"] or DASH) + "".join(f' <span class="mp-faint">{esc(m)}</span>' for m in marks)
        cells = [cell(f'<span class="nowrap">{model_html}</span>', raw=True), row["quantization"] or DASH,
                 row["prompt"] or DASH]
        if comparison["differing_options"]:
            cells.append(row["options"] or DASH)
        cells += [row["start"] or DASH, fmt_int(row["n"])]
        for _, key, fmt, pref in spec:
            text = fmt_ci(row["f1"], row["f1_ci"]) if key == "f1" else fmt(row[key])
            if row["latency_inflated"] and key in latency_keys and text != DASH:
                text += " *"
            cells.append(cell(text, bold=bool(pref and best[key][i])))
        cells += [_fits_text(row["fits"].get(p)) for p in profile_names]
        table_rows.append(cells)
    numeric = set(range(len(lead) - 1, len(headers)))
    data_table(headers, table_rows, numeric=numeric)
    footnotes = [
        "F1, recall, precision and FPR count an unknown answer as an error (missed threat or false alarm). "
        "Model memory is reported by Ollama (/api/ps); for the baseline it is the size of the fitted model.",
    ]
    if any(r["latency_inflated"] for r in rows):
        footnotes.append("* Latency inflated by about 2 s per request (run made before the Ollama URL fix).")
    if any(r["hidden"] for r in rows):
        footnotes.append("Legacy and dev runs are shown for reference only: never bold, never recommended.")
    note(" ".join(footnotes))


def _csv(rows: list[dict], profile_names: list[str]) -> bytes:
    flat = []
    for r in rows:
        item = {k: v for k, v in r.items() if k not in ("fits", "f1_ci")}
        item["f1_ci_low"] = (r["f1_ci"] or {}).get("low")
        item["f1_ci_high"] = (r["f1_ci"] or {}).get("high")
        for p in profile_names:
            item[f"fits_{p}"] = r["fits"].get(p)
        flat.append(item)
    return pd.DataFrame(flat).to_csv(index=False, na_rep="").encode("utf-8")


def _run_label(row: dict) -> str:
    label = f"{row['model']} [{row['prompt']}]"
    if row["options"]:
        label += f" {row['options']}"
    return label


def _tradeoff_charts(rows: list[dict]) -> None:
    chart_rows = [r for r in rows if r["f1"] is not None]
    if not chart_rows:
        return
    col1, col2 = st.columns(2, gap="large")
    with col1:
        pts = [r for r in chart_rows if r["p95_latency_s"] is not None]
        if pts:
            fig = go.Figure(go.Scatter(
                x=[r["p95_latency_s"] for r in pts], y=[r["f1"] for r in pts], mode="markers",
                text=[_run_label(r) for r in pts],
                marker=dict(color=[INK if r["can_be_best"] else SURFACE for r in pts], size=10,
                            line=dict(width=1.5, color=INK)),
                hovertemplate="%{text}<br>F1 %{y:.3f} · p95 %{x:.2f} s<extra></extra>", cliponaxis=False,
            ))
            fig.update_layout(xaxis_title="p95 latency (s)", yaxis_title="F1 (unknown = error)")
            plot(fig, caption="F1 vs p95 latency — up and to the left is better (open = legacy / dev)", height=380)
    with col2:
        pts = [r for r in chart_rows if r["throughput"] is not None and r["model_memory_mb"] is not None]
        if pts:
            fig = go.Figure(go.Scatter(
                x=[r["model_memory_mb"] for r in pts], y=[r["throughput"] for r in pts], mode="markers",
                text=[_run_label(r) for r in pts],
                marker=dict(color=INK, size=10, line=dict(width=1, color=SURFACE)),
                hovertemplate="%{text}<br>%{y:.1f} msg/min · %{x:.0f} MB<extra></extra>", cliponaxis=False,
            ))
            fig.update_layout(xaxis_title="model memory (MB)", yaxis_title="throughput (msg/min)")
            plot(fig, caption="throughput vs model memory — up and to the left is better", height=380)


def _per_class(reports: dict[str, dict], rows: list[dict]) -> None:
    section_title("By class", "Channel-level F1 per run: phishing (email), smishing (SMS) and valid messages.")
    names, per = [], {"phishing": [], "smishing": [], "valid": []}
    for r in rows:
        ch = ((reports[r["file"]].get("classification_metrics") or {}).get("channel_metrics") or {}).get("per_class", {})
        names.append(f"{_run_label(r)} ({run_timestamp(r['file']) or r['file']})")
        for k in per:
            per[k].append((ch.get(k) or {}).get("f1-score"))
    fig = go.Figure([
        go.Bar(name=k, x=names, y=v, marker=dict(color=CLASS_COLORS[k], line=dict(width=0)),
               hovertemplate=f"%{{x}}<br>{k} F1 %{{y:.2%}}<extra></extra>")
        for k, v in per.items()
    ])
    fig.update_layout(barmode="group", yaxis_title="F1", showlegend=True, bargroupgap=0.05)
    fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(tickangle=-15)
    plot(fig, caption="per-class F1 (excluding unknown)", height=360)


def render_multi_comparison_view() -> None:
    """Run selection, comparison table, recommendation, trade-off charts and per-class breakdown."""
    page_title(
        "Compare runs",
        "Detection quality against cost, for the runs you pick. Best value in each column is set in bold.",
    )
    all_files = list_saved_result_files()
    if not all_files:
        st.info("No saved evaluation reports found in the results/ directory to compare.")
        return
    all_reports = load_reports(all_files)

    show_hidden = st.toggle("show legacy and dev runs", key=_SHOW_HIDDEN_KEY, value=False)
    reports = visible_runs(all_reports, show_hidden)
    available_files = [f for f in all_files if f in reports]
    n_hidden = len(all_reports) - len(reports)
    if n_hidden:
        note(f"{n_hidden} legacy or dev run(s) hidden.")
    if not available_files:
        st.info("Every saved run is a legacy or dev run. Switch on \"show legacy and dev runs\" to see them, "
                "or make a new run in the Run tab.")
        return

    chip_labels = {
        f: run_chip_label(f, reports[f].get("model_name"),
                          (reports[f].get("manifest") or {}).get("prompt", {}).get("prompt_variant")
                          or reports[f].get("prompt_variant") or "legacy", "legacy")
        + (f" [{hidden_reason(reports[f])}]" if hidden_reason(reports[f]) else "")
        for f in available_files
    }

    # The multiselect owns its state under _SELECTION_KEY. Initialise it once, and
    # drop files that are not offered (a widget value outside its options raises).
    # Switching the legacy/dev toggle re-selects every run that is now offered.
    toggled = st.session_state.get("_compare_show_hidden_prev") != show_hidden
    st.session_state["_compare_show_hidden_prev"] = show_hidden
    if _SELECTION_KEY not in st.session_state or toggled:
        st.session_state[_SELECTION_KEY] = list(available_files)
    else:
        kept = [f for f in st.session_state[_SELECTION_KEY] if f in available_files]
        st.session_state[_SELECTION_KEY] = kept or list(available_files)

    ctrl_col1, ctrl_col2 = st.columns([6, 1], vertical_alignment="bottom")
    with ctrl_col1:
        selected_files = st.multiselect(
            "runs",
            options=available_files,
            key=_SELECTION_KEY,
            format_func=lambda f: chip_labels.get(f, f),
            help="Pick the runs to compare.",
        )
    with ctrl_col2:
        # on_click runs before the next script run, i.e. before the multiselect is
        # instantiated again, which is the only point where its key may be written.
        st.button("select all", type="tertiary", on_click=select_all_runs, args=(available_files,))

    if not selected_files:
        st.warning("Select at least one run.")
        return

    selected = {f: reports[f] for f in selected_files}
    profile_names = list(CONSTRAINT_PROFILES)
    comparison = build_comparison(selected, profile_names)
    rows = comparison["rows"]

    if comparison["mixed_samples"]:
        st.warning(
            "The selected runs were evaluated on different samples (dataset fingerprints differ or are missing): "
            "their numbers are not directly comparable, and pairwise tests are only made within one sample."
        )
    section_label("runs")
    _summary_table(rows, comparison, profile_names)
    st.download_button("Export comparison (CSV)", data=_csv(rows, profile_names),
                       file_name="markerphish_comparison.csv", mime="text/csv")

    render_recommendation(selected)
    section_title("Trade-offs", "Detection against latency and memory for the selected runs.")
    _tradeoff_charts(rows)
    _per_class(selected, rows)
