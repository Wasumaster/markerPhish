"""
MarkerPhish — Recommendation section of the Compare tab
========================================================
Renders backend.optimum.analyze: editable constraint profiles, the recommended
configuration per profile, the Pareto scatter (F1 vs model memory) and the
pairwise McNemar table. Rendering only; every decision is made in the backend.
"""

from __future__ import annotations

import copy

import plotly.graph_objects as go
import streamlit as st

from backend.optimum import ALPHA, analyze, pairwise_pvalues
from backend.stats import holm
from config import CONSTRAINT_PROFILES, MEMORY_HEADROOM_FACTOR
from ui.components import (
    DASH,
    cell,
    data_table,
    fmt_ci,
    fmt_mb,
    fmt_pct,
    fmt_seconds,
    note,
    plot,
    section_label,
    section_title,
)
from ui.theme import FAINT, FONT_MONO, INK, SURFACE

_PROFILES_KEY = "constraint_profiles_edit"
_HEADROOM_KEY = "constraint_headroom_edit"


def run_label(point: dict) -> str:
    """'model [variant] options' (no fixed names, works for any model)."""
    label = f"{point.get('model') or DASH} [{point.get('prompt_variant') or DASH}]"
    if point.get("options"):
        label += f" {point['options']}"
    return label


def _edit_profiles() -> tuple[dict, float]:
    """Constraint profiles editable for this session (defaults from config)."""
    if _PROFILES_KEY not in st.session_state:
        st.session_state[_PROFILES_KEY] = copy.deepcopy(CONSTRAINT_PROFILES)
    profiles = st.session_state[_PROFILES_KEY]
    with st.expander("constraint profiles", expanded=False):
        headroom = st.number_input(
            "headroom factor (share of RAM the model may use)", min_value=0.05, max_value=1.0, step=0.05,
            value=float(st.session_state.get(_HEADROOM_KEY, MEMORY_HEADROOM_FACTOR)), key=_HEADROOM_KEY,
        )
        for name, profile in profiles.items():
            c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom")
            with c1:
                note(f"{name}: {profile.get('description', '')}")
            with c2:
                profile["ram_gb"] = st.number_input(f"RAM GB · {name}", min_value=1.0, step=1.0,
                                                    value=float(profile["ram_gb"]), key=f"prof_ram_{name}")
            with c3:
                limit = st.number_input(f"P95 limit s · {name} (0 = none)", min_value=0.0, step=0.5,
                                        value=float(profile.get("max_p95_latency_s") or 0.0), key=f"prof_lat_{name}")
                profile["max_p95_latency_s"] = limit or None
        note("Changes apply to this session only; defaults are in config.CONSTRAINT_PROFILES.")
    return profiles, float(headroom)


def render_recommendation(reports: dict[str, dict]) -> None:
    """Recommendation, Pareto scatter and pairwise McNemar table for the selected runs."""
    section_title(
        "Recommendation",
        "Per constraint profile: the feasible configuration with the highest F1 (unknown = error); ties go to "
        "lower model memory, then lower P95 latency. Only protocol-2, non-dev runs on one dataset sample count.",
    )
    profiles, headroom = _edit_profiles()
    result = analyze(reports, profiles, headroom)
    points = result["points"]

    if not points:
        note("No run qualifies for the recommendation yet (see the excluded runs below).")
    else:
        rows = []
        for name, prof in result["per_profile"].items():
            winner = points.get(prof["winner"]) if prof["winner"] else None
            limit = f" · P95 ≤ {prof['max_p95_latency_s']} s" if prof.get("max_p95_latency_s") else ""
            if not winner:
                rows.append([name, f"{fmt_mb(prof['budget_mb'])}{limit}", "no feasible run", DASH, DASH, DASH, DASH])
                continue
            similar = [run_label(points[f]) for f in prof["alternatives_not_significantly_different"]]
            why = (f"highest F1 of {len(prof['feasible'])} feasible run(s); "
                   f"{len(prof['not_feasible'])} over budget or limit")
            rows.append([
                name,
                f"{fmt_mb(prof['budget_mb'])}{limit}",
                cell(run_label(winner), bold=True),
                fmt_ci(winner["f1"], winner["f1_ci"]),
                f"{fmt_mb(winner['model_memory_mb'])} · p95 {fmt_seconds(winner['p95_latency_s'])}",
                why,
                ", ".join(similar) or "none",
            ])
        data_table(
            ["profile", "budget", "recommended", "F1 [95% CI]", "memory · latency", "why",
             f"not significantly different (Holm, α {ALPHA})"],
            rows,
            sans={5, 6},
        )
        note(
            "\"Not significantly different\" means McNemar's test on the paired answers (Holm-adjusted over the "
            "comparisons of one profile) cannot separate the run from the winner on this sample; it does not "
            "mean the two are equivalent. Confidence-interval overlap is not used as a test."
        )
        _pareto_chart(points, result)
        _pairwise_table(reports, points)

    if result["excluded"]:
        with st.expander(f"runs excluded from the recommendation ({len(result['excluded'])})", expanded=False):
            data_table(["run", "reason"], [[e["file"], e["reason"]] for e in result["excluded"]], sans={1})


def _pareto_chart(points: dict, result: dict) -> None:
    complete = [p for p in points.values() if p["f1"] is not None and p["model_memory_mb"] is not None]
    if not complete:
        return
    front = [p for p in complete if p["pareto"]]
    rest = [p for p in complete if not p["pareto"]]
    traces = []
    for group, filled, name in ((front, True, "Pareto front"), (rest, False, "dominated")):
        if not group:
            continue
        traces.append(go.Scatter(
            name=name,
            x=[p["model_memory_mb"] for p in group],
            y=[p["f1"] for p in group],
            mode="markers",
            marker=dict(color=INK if filled else SURFACE, size=11, symbol="circle",
                        line=dict(width=1.5, color=INK)),
            text=[run_label(p) for p in group],
            customdata=[[p["p95_latency_s"], fmt_pct(p["unknown_rate"])] for p in group],
            hovertemplate="%{text}<br>F1 %{y:.3f} · %{x:.0f} MB · p95 %{customdata[0]:.2f} s"
                          " · unknown %{customdata[1]}<extra></extra>",
            cliponaxis=False,
        ))
    fig = go.Figure(traces)
    for name, prof in result["per_profile"].items():
        fig.add_vline(x=prof["budget_mb"], line_dash="dash", line_color=FAINT, line_width=1,
                      annotation_text=f"{name} budget", annotation_position="top",
                      annotation_font=dict(family=FONT_MONO, size=10, color=FAINT))
    fig.update_layout(xaxis_title="model memory (MB, Ollama)", yaxis_title="F1 (unknown = error)",
                      showlegend=True)
    plot(fig, caption="F1 vs model memory · filled = Pareto front (F1, memory, P95) · dashed = memory budget",
         height=420)


def _pairwise_table(reports: dict[str, dict], points: dict) -> None:
    names = [n for n in reports if n in points]
    if len(names) < 2:
        return
    table = pairwise_pvalues({n: reports[n] for n in names})
    pairs = [(a, b, p) for (a, b), p in table.items()]
    testable = [p for _, _, p in pairs if p is not None]
    adjusted = iter(holm(testable))
    rows = []
    for a, b, p in pairs:
        p_holm = next(adjusted) if p is not None else None
        rows.append([
            run_label(points[a]), run_label(points[b]),
            DASH if p is None else f"{p:.4f}",
            DASH if p_holm is None else cell(f"{p_holm:.4f}", bold=p_holm < ALPHA),
        ])
    with st.expander(f"pairwise McNemar tests ({len(rows)} pairs)", expanded=False):
        data_table(["run A", "run B", "p", "p (Holm, all pairs)"], rows, numeric={2, 3})
        note(f"Bold: Holm-adjusted p < {ALPHA}. Pairs are the eligible runs on the reference sample; "
             "records are paired by source id; an unknown answer counts as wrong.")
