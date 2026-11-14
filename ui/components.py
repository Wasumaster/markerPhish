"""
MarkerPhish — UI Components
============================
Small HTML render helpers shared by all tabs: top bar, headings, section labels,
data tables, composition bar, legends, run chip labels, number formatting and
the Plotly chart wrapper. Presentation only: no evaluation logic.

All HTML is emitted as a single line (no blank lines or indentation), so
Streamlit's Markdown renderer treats it as one raw HTML block.
"""

from __future__ import annotations

import html
import math
from datetime import datetime

import streamlit as st

from ui.theme import FAINT, INK, MALICIOUS, ONLINE, SAFE, SURFACE

DASH = "—"


# ──────────────────────────────────────────────
# Escaping & formatting
# ──────────────────────────────────────────────
def esc(value) -> str:
    """Escape text for inline HTML (also neutralises '$' math and line breaks)."""
    text = html.escape(str(value), quote=True)
    return text.replace("$", "&#36;").replace("\r", " ").replace("\n", " ")


def _number(value) -> float | None:
    """Return value as float, or None if it is missing / not numeric / NaN."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def fmt_pct(value) -> str:
    """0.1234 -> '12.34%'; missing -> '—'."""
    number = _number(value)
    return DASH if number is None else f"{number:.2%}"


def fmt_seconds(value, decimals: int = 2) -> str:
    """9.153 -> '9.15 s'; missing -> '—'."""
    number = _number(value)
    return DASH if number is None else f"{number:.{decimals}f} s"


def fmt_rate(value) -> str:
    """12.46 -> '12.5 msg/min'; missing -> '—'."""
    number = _number(value)
    return DASH if number is None else f"{number:.1f} msg/min"


def fmt_mb(value) -> str:
    """940.4 -> '940.4 MB'; missing or zero -> '—'."""
    number = _number(value)
    return DASH if not number else f"{number:,.1f} MB"


def fmt_int(value) -> str:
    """1234 -> '1,234'; missing -> '—'."""
    number = _number(value)
    return DASH if number is None else f"{int(number):,}"


def fmt_ratio(value, decimals: int = 4) -> str:
    """0.81234 -> '0.8123'; missing -> '—'."""
    number = _number(value)
    return DASH if number is None else f"{number:.{decimals}f}"


def fmt_ci(value, ci: dict | None = None, decimals: int = 3) -> str:
    """0.893 with ci {low, high} -> '0.893 [0.825; 0.946]'; missing -> '—'."""
    number = _number(value)
    if number is None:
        return DASH
    text = f"{number:.{decimals}f}"
    low = _number((ci or {}).get("low"))
    high = _number((ci or {}).get("high"))
    if low is not None and high is not None:
        text += f" [{low:.{decimals}f}; {high:.{decimals}f}]"
    return text


# ──────────────────────────────────────────────
# Page furniture
# ──────────────────────────────────────────────
def _html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def top_bar(online: bool, endpoint: str) -> None:
    """Full-width top bar: product name left, Ollama status right."""
    host = endpoint.split("://", 1)[-1].rstrip("/")
    dot = ONLINE if online else MALICIOUS
    status = f"ollama · {host}" if online else "ollama · offline"
    _html(
        '<div class="mp-topbar">'
        '<div class="mp-topbar-brand"><span class="mp-topbar-name">markerphish</span>'
        '<span class="mp-topbar-desc">LLM phishing &amp; smishing detection benchmark</span></div>'
        f'<div class="mp-status"><span class="mp-dot" style="background:{dot};"></span>{esc(status)}</div>'
        "</div>"
    )


def page_title(title: str, description: str | None = None) -> None:
    """h1 page title with an optional muted description underneath."""
    desc = f'<p class="mp-desc">{esc(description)}</p>' if description else ""
    _html(f'<div><h1 class="mp-h1">{esc(title)}</h1>{desc}</div>')


def section_title(title: str, description: str | None = None) -> None:
    """h2 section title with an optional muted description."""
    desc = f'<p class="mp-desc">{esc(description)}</p>' if description else ""
    _html(f'<div><h2 class="mp-h2">{esc(title)}</h2>{desc}</div>')


def section_label(text: str) -> None:
    """Lowercase mono label placed above a control, table or chart."""
    _html(f'<p class="mp-label">{esc(text)}</p>')


def note(text: str, mono: bool = False) -> None:
    """Muted 12px note line."""
    cls = "mp-sidebar-line" if mono else "mp-note"
    _html(f'<p class="{cls}">{esc(text)}</p>')


def html_block(markup: str) -> None:
    """Render pre-built (already escaped) HTML."""
    _html(markup)


# ──────────────────────────────────────────────
# Tables
# ──────────────────────────────────────────────
def cell(text, *, color: str | None = None, bold: bool = False, raw: bool = False) -> dict:
    """A table cell with optional colour / weight. raw=True means text is trusted HTML."""
    return {"html": str(text) if raw else esc(text), "color": color, "bold": bold}


def data_table(
    headers: list[str],
    rows: list[list],
    *,
    numeric: set[int] | None = None,
    sans: set[int] | None = None,
    wrap: set[int] | None = None,
    total: list | None = None,
    max_width: str | None = None,
    max_height: str | None = None,
) -> None:
    """
    Render an HTML data table.

    headers   column titles (rendered uppercase, mono 12px)
    rows      list of rows; each value is a plain value (escaped) or a cell() dict
    numeric   column indexes to right-align
    sans      column indexes set in IBM Plex Sans (free text)
    wrap      column indexes whose (mono) text may wrap
    total     optional summary row (weight 600, ink top border)
    """
    numeric = numeric or set()
    sans = sans or set()
    wrap = wrap or set()

    def _classes(idx: int) -> str:
        cls = []
        if idx in numeric and idx != 0:
            cls.append("num")
        if idx in sans:
            cls.append("sans")
        if idx in wrap:
            cls.append("wrap")
        return f' class="{" ".join(cls)}"' if cls else ""

    def _td(idx: int, value) -> str:
        if isinstance(value, dict):
            body = value["html"]
            style = f' style="color:{value["color"]};"' if value.get("color") else ""
            cls = _classes(idx)
            if value.get("bold"):
                cls = cls[:-1] + ' best"' if cls else ' class="best"'
            return f"<td{cls}{style}>{body}</td>"
        return f"<td{_classes(idx)}>{esc(value)}</td>"

    head = "".join(f"<th{_classes(i)}>{esc(h)}</th>" for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(_td(i, v) for i, v in enumerate(row)) + "</tr>" for row in rows)
    foot = ""
    if total is not None:
        foot = '<tr class="total">' + "".join(_td(i, v) for i, v in enumerate(total)) + "</tr>"

    style = []
    if max_width:
        style.append(f"max-width:{max_width};")
    if max_height:
        style.append(f"max-height:{max_height};")
    wrap_style = f' style="{"".join(style)}"' if style else ""
    _html(
        f'<div class="mp-table-wrap"{wrap_style}><table class="mp-table">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}{foot}</tbody></table></div>"
    )


def kv_table(pairs: list[tuple], *, headers: tuple[str, str] = ("item", "value"), max_width: str = "640px") -> None:
    """Two-column key / value table for descriptive settings (left-aligned, values wrap)."""
    data_table(list(headers), [[k, v] for k, v in pairs], wrap={1}, max_width=max_width)


def best_flags(values: list, prefer: str) -> list[bool]:
    """Mark the best value(s) in a column: prefer='max' or 'min'. Missing values never win."""
    numbers = [_number(v) for v in values]
    present = [n for n in numbers if n is not None]
    if not present:
        return [False] * len(values)
    target = max(present) if prefer == "max" else min(present)
    return [n is not None and math.isclose(n, target, rel_tol=1e-9, abs_tol=1e-12) for n in numbers]


# ──────────────────────────────────────────────
# Composition bar & legends
# ──────────────────────────────────────────────
def stacked_bar(segments: list[tuple[str, int, str]]) -> None:
    """One horizontal stacked bar; segment widths proportional to counts, legend below."""
    total = sum(count for _, count, _ in segments) or 1
    parts = "".join(
        f'<div title="{esc(label)} {count}" style="width:{count / total * 100:.4f}%;background:{color};"></div>'
        for label, count, color in segments
        if count > 0
    )
    _html(f'<div class="mp-stack">{parts}</div>{_legend_html([(f"{lbl} {cnt}", col) for lbl, cnt, col in segments])}')


def _legend_html(items: list[tuple[str, str]]) -> str:
    entries = "".join(
        f'<span class="mp-legend-item"><span class="mp-swatch" style="background:{color};"></span>{esc(label)}</span>'
        for label, color in items
    )
    return f'<div class="mp-legend">{entries}</div>'


def legend_row(items: list[tuple[str, str]]) -> None:
    """Row of 10px colour squares with mono labels."""
    _html(_legend_html(items))


def marker_legend(items: list[tuple[str, bool]]) -> None:
    """Legend for ink point series: (label, filled) -> filled disc or hollow ring."""
    entries = "".join(
        f'<span class="mp-legend-item"><span class="{"mp-disc" if filled else "mp-ring"}"></span>{esc(label)}</span>'
        for label, filled in items
    )
    _html(f'<div class="mp-legend">{entries}</div>')


def colored(text, color: str, bold: bool = True) -> str:
    """Inline coloured mono text (returns HTML)."""
    weight = "600" if bold else "400"
    return f'<span class="mp-mono" style="color:{color};font-weight:{weight};">{esc(text)}</span>'


def strong(text) -> str:
    """Inline mono 600 text (returns HTML)."""
    return f'<span class="mp-strong">{esc(text)}</span>'


# ──────────────────────────────────────────────
# Run labels
# ──────────────────────────────────────────────
def run_timestamp(filename: str) -> str | None:
    """'eval_..._20261017_140404.json' -> '17 Oct 14:04' (None if no timestamp)."""
    stem = filename[:-5] if filename.endswith(".json") else filename
    parts = stem.split("_")
    if len(parts) >= 3:
        try:
            return datetime.strptime(f"{parts[-2]}{parts[-1]}", "%Y%m%d%H%M%S").strftime("%d %b %H:%M")
        except ValueError:
            return None
    return None


def run_chip_label(filename: str, model_name: str | None, prompt_variant: str | None, legacy_name: str) -> str:
    """Chip text 'model · 17 Oct 14:04'; non-legacy runs also show the prompt variant."""
    parts = [model_name or filename]
    if prompt_variant and prompt_variant != legacy_name:
        parts.append(prompt_variant)
    stamp = run_timestamp(filename)
    if stamp:
        parts.append(stamp)
    return " · ".join(parts)


# ──────────────────────────────────────────────
# Charts
# ──────────────────────────────────────────────
def chart_caption(text: str) -> None:
    """Mono caption placed above a chart (charts carry no internal title)."""
    _html(f'<p class="mp-chart-caption">{esc(text)}</p>')


def plot(fig, caption: str | None = None, height: int | None = None) -> None:
    """Render a Plotly figure with the markerphish template, no modebar, no Streamlit theme."""
    if caption:
        chart_caption(caption)
    # Background set explicitly: Streamlit otherwise paints charts with the page colour
    fig.update_layout(template="markerphish", title=dict(text=""), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE)
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})


__all__ = [
    "DASH", "INK", "FAINT", "SAFE", "MALICIOUS",
    "esc", "fmt_pct", "fmt_seconds", "fmt_rate", "fmt_mb", "fmt_int", "fmt_ratio",
    "top_bar", "page_title", "section_title", "section_label", "note", "html_block",
    "cell", "data_table", "kv_table", "best_flags",
    "stacked_bar", "legend_row", "marker_legend", "colored", "strong",
    "run_timestamp", "run_chip_label", "chart_caption", "plot",
]
