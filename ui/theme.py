"""
MarkerPhish — Theme
====================
Design tokens, global CSS injection and the Plotly template.

The look is a "lab report": paper background, black ink, thin rules, monospace
numbers, one orange / blue pair for the two classes. Every colour used anywhere
in the UI comes from the tokens below.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ──────────────────────────────────────────────
# Design tokens
# ──────────────────────────────────────────────
PAPER = "#F6F5F1"          # page background
SURFACE = "#FFFFFF"        # inputs, chart background
INK = "#1A1A18"            # text, strong borders, primary button
MUTED = "#5A5A55"          # secondary text
FAINT = "#8A8880"          # tertiary text
RULE = "#DCDAD3"           # dividers
HOVER = "#EFEDE7"          # table row hover
MALICIOUS = "#B4480F"      # malicious class / smishing
MALICIOUS_LIGHT = "#E08A4F"  # phishing (email) subtype
SAFE = "#1F4E9C"           # safe class / valid SMS
SAFE_LIGHT = "#7FA3DA"     # valid email subtype
ONLINE = "#2F7D4F"         # status-online dot

FONT_SANS = "'IBM Plex Sans', -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
FONT_MONO = "'IBM Plex Mono', ui-monospace, 'SFMono-Regular', Consolas, monospace"

RADIUS = "2px"
CONTROL_HEIGHT = "44px"

# Channel-level label -> colour (used by tables, legends and charts)
CLASS_COLORS = {
    "malicious": MALICIOUS,
    "safe": SAFE,
    "valid": SAFE,
    "smishing": MALICIOUS,
    "phishing": MALICIOUS_LIGHT,
    "valid sms": SAFE,
    "valid email": SAFE_LIGHT,
}


# ──────────────────────────────────────────────
# Global CSS
# ──────────────────────────────────────────────
_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

/* Global type and page background */
html, body, .stApp, [data-testid="stAppViewContainer"] {{
    font-family: {FONT_SANS};
    font-size: 14px;
    line-height: 1.5;
    color: {INK};
    background: {PAPER};
}}
.stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp li, .stApp label, .stApp span:not([data-testid="stIconMaterial"]), .stApp div, .stApp input, .stApp textarea, .stApp button {{
    font-family: {FONT_SANS};
}}
.stApp code, .stApp pre, .stApp kbd {{
    font-family: {FONT_MONO} !important;
}}

/* Material icons (expander arrows, help) keep their icon font */
.stApp [data-testid="stIconMaterial"] {{
    font-family: "Material Symbols Rounded" !important;
}}

/* No shadows anywhere */
.stApp *, [data-testid="stSelectboxVirtualDropdown"] *, [role="listbox"] * {{
    box-shadow: none !important;
    text-shadow: none !important;
}}

/* Streamlit chrome: decoration bar, toolbar, deploy button, menu, footer */
[data-testid="stDecoration"],
[data-testid="stToolbar"],
[data-testid="stStatusWidget"],
[data-testid="manage-app-button"],
.stDeployButton,
#MainMenu,
footer {{
    display: none !important;
}}
header[data-testid="stHeader"] {{
    background: transparent !important;
    height: 0 !important;
}}

/* Main column: padding and max width */
[data-testid="stMainBlockContainer"], .block-container {{
    padding: 24px 32px 48px 32px !important;
    max-width: 1120px !important;
}}
[data-testid="stMain"] [data-testid="stVerticalBlock"] {{
    gap: 16px;
}}

/* Sidebar: paper background, right rule, padding and group spacing */
section[data-testid="stSidebar"] {{
    width: 248px !important;
    min-width: 248px !important;
    max-width: 248px !important;
    background: {PAPER} !important;
    border-right: 1px solid {RULE} !important;
}}
section[data-testid="stSidebar"] > div {{
    background: {PAPER} !important;
}}
[data-testid="stSidebarUserContent"] {{
    padding: 24px 20px !important;
}}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
    gap: 22px;
}}
[data-testid="stSidebarHeader"] {{
    height: 8px !important;
    min-height: 0 !important;
    padding: 0 20px !important;
}}

/* Widget labels act as mono section labels ("model", "prompt", ...) */
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {{
    font-family: {FONT_MONO} !important;
    font-size: 12px !important;
    font-weight: 400 !important;
    color: {MUTED} !important;
    text-transform: lowercase;
    min-height: 0 !important;
    margin-bottom: 6px !important;
}}

/* Select and multiselect boxes (react-aria ComboBox: the bordered box is role="group") */
[data-testid="stSelectbox"] [role="group"],
[data-testid="stMultiSelect"] [role="group"] {{
    background: {SURFACE} !important;
    border: 1px solid {INK} !important;
    border-radius: {RADIUS} !important;
    min-height: {CONTROL_HEIGHT} !important;
}}
[data-testid="stSelectbox"] input,
[data-testid="stMultiSelect"] input {{
    font-family: {FONT_MONO} !important;
    font-size: 14px !important;
    color: {INK} !important;
    -webkit-text-fill-color: {INK} !important;
    background: transparent !important;
}}
[data-testid="stSelectbox"] [role="group"] svg,
[data-testid="stMultiSelect"] [role="group"] > button svg {{
    color: {INK} !important;
    fill: {INK} !important;
}}

/* Dropdown menus of selects / multiselects (role="listbox") */
[data-testid="stSelectboxVirtualDropdown"], [role="listbox"] {{
    background: {SURFACE} !important;
    border-radius: {RADIUS} !important;
}}
[role="listbox"] {{
    border: 1px solid {INK} !important;
}}
[role="listbox"] [role="option"], [role="listbox"] [role="option"] * {{
    font-family: {FONT_MONO} !important;
    font-size: 14px !important;
    color: {INK} !important;
    border-radius: 0 !important;
}}
[role="listbox"] [role="option"][data-focused="true"],
[role="listbox"] [role="option"][data-hovered="true"],
[role="listbox"] [role="option"][data-selected="true"],
[role="listbox"] [role="option"]:hover {{
    background: {HOVER} !important;
}}

/* Multiselect chips (selected runs): white, 1px ink border, mono 13px */
[data-testid="stMultiSelect"] [data-tag] {{
    background: {SURFACE} !important;
    border: 1px solid {INK} !important;
    border-radius: {RADIUS} !important;
    color: {INK} !important;
}}
[data-testid="stMultiSelect"] [data-tag] span {{
    font-family: {FONT_MONO} !important;
    font-size: 13px !important;
    color: {INK} !important;
    background: transparent !important;
}}
[data-testid="stMultiSelect"] [data-tag] button,
[data-testid="stMultiSelect"] [data-tag] button svg {{
    color: {INK} !important;
    fill: {INK} !important;
    background: transparent !important;
}}

/* Text inputs and text areas */
[data-testid="stTextInputRootElement"] {{
    background: {SURFACE} !important;
    border: 1px solid {INK} !important;
    border-radius: {RADIUS} !important;
    min-height: {CONTROL_HEIGHT} !important;
}}
[data-testid="stTextInputField"], [data-testid="stTextArea"] textarea {{
    font-family: {FONT_MONO} !important;
    font-size: 14px !important;
    color: {INK} !important;
    -webkit-text-fill-color: {INK} !important;
    background: {SURFACE} !important;
}}
[data-testid="stTextArea"] textarea,
[data-testid="stTextArea"] [data-testid="stTextAreaRootElement"] {{
    border-color: {RULE} !important;
    border-radius: {RADIUS} !important;
}}

/* Buttons: primary (ink), secondary (white, ink border), tertiary (underlined text) */
.stButton button, .stDownloadButton button, [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"] {{
    border-radius: {RADIUS} !important;
    min-height: {CONTROL_HEIGHT} !important;
    font-family: {FONT_SANS} !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    transition: none !important;
}}
[data-testid="stBaseButton-primary"] {{
    background: {INK} !important;
    border: 1px solid {INK} !important;
    color: {PAPER} !important;
}}
[data-testid="stBaseButton-primary"] * {{
    color: {PAPER} !important;
}}
[data-testid="stBaseButton-primary"]:disabled {{
    background: {FAINT} !important;
    border-color: {FAINT} !important;
}}
[data-testid="stBaseButton-secondary"] {{
    background: {SURFACE} !important;
    border: 1px solid {INK} !important;
    color: {INK} !important;
}}
[data-testid="stBaseButton-secondary"]:hover {{
    background: {HOVER} !important;
}}
[data-testid="stBaseButton-tertiary"] {{
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    min-height: {CONTROL_HEIGHT} !important;
    color: {INK} !important;
    text-decoration: underline !important;
    text-underline-offset: 3px;
}}
[data-testid="stBaseButton-tertiary"] * {{
    color: {INK} !important;
    font-family: {FONT_MONO} !important;
    font-size: 13px !important;
}}

/* Tabs (react-aria): plain text links, 2px ink underline on the active one */
[role="tablist"] {{
    gap: 24px !important;
    border-bottom: 1px solid {RULE} !important;
    background: transparent !important;
}}
[data-testid="stTab"] {{
    background: transparent !important;
    padding: 0 0 10px 0 !important;
    margin-bottom: -1px !important;
    height: auto !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
}}
[data-testid="stTab"] p {{
    font-size: 14px !important;
    font-weight: 400 !important;
    color: {MUTED} !important;
}}
[data-testid="stTab"][aria-selected="true"] {{
    border-bottom: 2px solid {INK} !important;
}}
[data-testid="stTab"][aria-selected="true"] p {{
    color: {INK} !important;
    font-weight: 600 !important;
}}
[data-testid="stTab"]:hover p {{
    color: {INK} !important;
}}
/* Hide react-aria's own tab rule (::after) and sliding indicator; the 2px border above replaces them */
[role="tablist"]::after {{
    display: none !important;
}}
[role="tablist"] > div:not([data-testid="stTab"]) {{
    display: none !important;
}}
[role="tabpanel"] {{
    padding-top: 24px !important;
}}

/* Radio buttons (device selection) */
[data-testid="stRadio"] label p {{
    font-size: 13px !important;
}}

/* Progress bar: thin 4px ink bar on a rule track */
[data-testid="stProgressBarTrack"] {{
    height: 4px !important;
    background: {RULE} !important;
    border-radius: 0 !important;
}}
[data-testid="stProgressBarTrack"] > div {{
    background: {INK} !important;
    border-radius: 0 !important;
}}
[data-testid="stProgress"] p {{
    font-family: {FONT_MONO} !important;
    font-size: 12px !important;
    color: {MUTED} !important;
}}

/* Spinner: ink, no colour */
[data-testid="stSpinnerIcon"] {{
    border-color: {RULE} !important;
    border-top-color: {INK} !important;
}}
[data-testid="stSpinner"] p {{
    font-family: {FONT_MONO} !important;
    font-size: 12px !important;
    color: {MUTED} !important;
}}

/* Alerts (info / warning / error / success): white box, thin rule, no tint, no icon */
[data-testid="stAlert"] > div, [data-testid="stAlertContainer"] {{
    background: {SURFACE} !important;
    border: 1px solid {RULE} !important;
    border-left: 2px solid {INK} !important;
    border-radius: {RADIUS} !important;
    color: {INK} !important;
}}
[data-testid="stAlert"] p {{
    color: {INK} !important;
    font-size: 13px !important;
}}
[data-testid="stAlertDynamicIcon"] {{
    display: none !important;
}}

/* Expanders: thin rule, no fill */
[data-testid="stExpander"] details {{
    border: 1px solid {RULE} !important;
    border-radius: {RADIUS} !important;
    background: transparent !important;
}}
[data-testid="stExpander"] summary {{
    background: transparent !important;
}}
[data-testid="stExpander"] summary p {{
    font-family: {FONT_MONO} !important;
    font-size: 13px !important;
    color: {INK} !important;
}}
[data-testid="stExpander"] summary:hover {{
    background: {HOVER} !important;
}}

/* Captions */
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{
    color: {MUTED} !important;
    font-size: 12px !important;
}}

/* Code blocks (scikit-learn report) */
[data-testid="stCode"] pre, .stCode pre {{
    background: {SURFACE} !important;
    border: 1px solid {RULE} !important;
    border-radius: {RADIUS} !important;
}}
[data-testid="stCode"] code {{
    font-size: 12px !important;
    color: {INK} !important;
}}

/* Interactive dataframes (audit log, large message lists) */
[data-testid="stDataFrame"], [data-testid="stDataFrameResizable"] {{
    border: 1px solid {RULE} !important;
    border-radius: 0 !important;
}}

/* Plotly chart container */
[data-testid="stPlotlyChart"] {{
    border: 1px solid {RULE};
    background: {SURFACE};
}}

/* Horizontal dividers */
.stApp hr {{
    border-color: {RULE} !important;
    margin: 8px 0 !important;
}}

/* ── Custom components (ui.components) ── */
.mp-topbar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
    padding: 14px 28px;
    margin: -24px -32px 8px -32px;
    border-bottom: 1px solid {INK};
}}
.mp-topbar-brand {{ display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; }}
.mp-topbar-name {{ font-family: {FONT_MONO}; font-size: 16px; font-weight: 600; color: {INK}; }}
.mp-topbar-desc {{ font-size: 13px; color: {MUTED}; }}
.mp-status {{ display: flex; align-items: center; gap: 8px; font-family: {FONT_MONO}; font-size: 12px; color: {MUTED}; }}
.mp-dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}

.stApp h1.mp-h1 {{ font-family: {FONT_SANS} !important; font-size: 22px !important; font-weight: 600 !important; color: {INK}; margin: 0 !important; padding: 0 !important; line-height: 1.3 !important; letter-spacing: 0 !important; }}
.stApp h2.mp-h2 {{ font-family: {FONT_SANS} !important; font-size: 16px !important; font-weight: 600 !important; color: {INK}; margin: 18px 0 0 0 !important; padding: 0 !important; line-height: 1.4 !important; letter-spacing: 0 !important; }}
.mp-desc {{ font-size: 14px; color: {MUTED}; max-width: 680px; margin: 6px 0 0 0; }}
.mp-label {{ font-family: {FONT_MONO}; font-size: 12px; color: {MUTED}; text-transform: lowercase; margin: 0 0 6px 0; }}
.mp-note {{ font-size: 12px; color: {MUTED}; margin: 0; }}
.mp-mono {{ font-family: {FONT_MONO}; }}
.mp-faint {{ color: {FAINT}; }}
.mp-muted {{ color: {MUTED}; }}
.mp-strong {{ font-family: {FONT_MONO}; font-weight: 600; }}
.mp-sentence {{ font-size: 16px; color: {INK}; margin: 0; }}

.mp-table-wrap {{ overflow-x: auto; overflow-y: auto; width: 100%; }}
.mp-table {{ border-collapse: collapse; width: 100%; font-family: {FONT_MONO}; font-size: 14px; color: {INK}; }}
.mp-table th {{
    font-family: {FONT_MONO}; font-size: 12px; font-weight: 400; text-transform: uppercase; letter-spacing: .04em;
    color: {MUTED}; padding: 10px 12px; border-bottom: 1px solid {INK}; text-align: left; white-space: nowrap;
    position: sticky; top: 0; background: {PAPER};
}}
.mp-table td {{ padding: 11px 12px; border-bottom: 1px solid {RULE}; vertical-align: top; text-align: left; white-space: nowrap; }}
.mp-table td.sans {{ white-space: normal; min-width: 280px; }}
.mp-table td.wrap {{ white-space: normal; }}
.mp-table tbody tr:hover td {{ background: {HOVER}; }}
.mp-table .num {{ text-align: right; white-space: nowrap; }}
.mp-table .sans {{ font-family: {FONT_SANS}; }}
.mp-table .nowrap {{ white-space: nowrap; }}
.mp-table .best {{ font-weight: 600; }}
.mp-table tr.total td {{ font-weight: 600; border-top: 1px solid {INK}; }}

.mp-stack {{ display: flex; height: 36px; border: 1px solid {INK}; width: 100%; }}
.mp-stack > div {{ height: 100%; }}
.mp-legend {{ display: flex; flex-wrap: wrap; gap: 6px 20px; margin-top: 10px; font-family: {FONT_MONO}; font-size: 12px; color: {INK}; }}
.mp-legend-item {{ display: inline-flex; align-items: center; gap: 6px; }}
.mp-swatch {{ width: 10px; height: 10px; display: inline-block; }}
.mp-ring {{ width: 10px; height: 10px; display: inline-block; border: 1.5px solid {INK}; border-radius: 50%; box-sizing: border-box; }}
.mp-disc {{ width: 10px; height: 10px; display: inline-block; background: {INK}; border-radius: 50%; }}
.mp-chart-caption {{ font-family: {FONT_MONO}; font-size: 12px; color: {MUTED}; margin: 0 0 6px 0; }}
.mp-sidebar-line {{ font-family: {FONT_MONO}; font-size: 12px; color: {MUTED}; margin: 6px 0 0 0; line-height: 1.5; }}
.mp-sidebar-line-sans {{ font-size: 12px; color: {MUTED}; margin: 6px 0 0 0; line-height: 1.5; }}

/* Mono for numbers, labels, model names, IDs (beats the global sans rule above) */
.stApp .mp-mono, .stApp .mp-mono *, .stApp .mp-table, .stApp .mp-table *,
.stApp .mp-topbar-name, .stApp .mp-status, .stApp .mp-status *, .stApp .mp-label,
.stApp .mp-legend, .stApp .mp-legend *, .stApp .mp-chart-caption, .stApp .mp-sidebar-line,
.stApp .mp-strong {{
    font-family: {FONT_MONO} !important;
}}
.stApp .mp-table .sans, .stApp .mp-table .sans * {{
    font-family: {FONT_SANS} !important;
}}

/* Narrow screens: columns stack, top bar wraps */
@media (max-width: 760px) {{
    [data-testid="stMainBlockContainer"], .block-container {{ padding: 16px 16px 40px 16px !important; }}
    .mp-topbar {{ margin: -16px -16px 8px -16px; padding: 12px 16px; }}
    [data-testid="stHorizontalBlock"] {{ flex-direction: column !important; }}
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{ width: 100% !important; flex: 1 1 100% !important; }}
}}
</style>
"""


# ──────────────────────────────────────────────
# Plotly template
# ──────────────────────────────────────────────
def _register_plotly_template() -> None:
    """Register the 'markerphish' Plotly template and make it the default."""
    axis = dict(
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor=INK,
        linewidth=1,
        mirror=False,
        ticks="outside",
        tickcolor=INK,
        ticklen=4,
        tickfont=dict(family=FONT_MONO, size=11, color=MUTED),
        title=dict(font=dict(family=FONT_MONO, size=12, color=MUTED)),
        automargin=True,
    )
    pio.templates["markerphish"] = go.layout.Template(
        layout=dict(
            paper_bgcolor=SURFACE,
            plot_bgcolor=SURFACE,
            font=dict(family=FONT_MONO, size=11, color=MUTED),
            title=dict(text=""),
            colorway=[INK, MALICIOUS, SAFE, MALICIOUS_LIGHT, SAFE_LIGHT, FAINT],
            xaxis=axis,
            yaxis=axis,
            showlegend=False,
            legend=dict(
                font=dict(family=FONT_MONO, size=11, color=INK),
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                bgcolor=SURFACE,
            ),
            hoverlabel=dict(
                bgcolor=SURFACE,
                bordercolor=INK,
                font=dict(family=FONT_MONO, size=12, color=INK),
            ),
            margin=dict(t=20, r=20, b=48, l=56),
            bargap=0.35,
            coloraxis=dict(colorbar=dict(outlinewidth=0, tickfont=dict(family=FONT_MONO, size=11, color=MUTED))),
        )
    )
    pio.templates.default = "markerphish"


_register_plotly_template()


def inject_css() -> None:
    """Inject fonts and the global stylesheet. Call once at the top of the app."""
    st.markdown(_CSS, unsafe_allow_html=True)
