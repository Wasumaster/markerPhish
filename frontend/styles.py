"""
MarkerPhish — Design System & Professional Styling
====================================================
Centralised CSS design tokens, component styles, and Streamlit element overrides.
Injects global styles and hides non-essential Streamlit chrome (Deploy button,
hamburger menu) for a polished, enterprise-grade research interface.
"""

import streamlit as st

# ──────────────────────────────────────────────
# Global CSS Design System
# ──────────────────────────────────────────────
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.stApp {
    background-color: #0b0f19;
    color: #f1f5f9;
}

/* ─── Hide Streamlit Deploy Button & Action Chrome ─── */
.stDeployButton,
[data-testid="stToolbar"] > div:first-child,
button[data-testid="stBaseButton-header"],
div[data-testid="stStatusWidget"],
[data-testid="manage-app-button"],
.viewerBadge_link__1S137,
.viewerBadge_container__r5tak {
    display: none !important;
    visibility: hidden !important;
    pointer-events: none !important;
    height: 0 !important;
    width: 0 !important;
    overflow: hidden !important;
}

/* ─── Hide Streamlit Hamburger / Main Menu & Header Chrome ─── */
#MainMenu {
    visibility: hidden !important;
    display: none !important;
}

header[data-testid="stHeader"] [data-testid="stToolbar"] {
    display: none !important;
}

header[data-testid="stHeader"] {
    background-color: transparent !important;
}

footer {
    visibility: hidden !important;
    display: none !important;
}

/* ─── Header Styling ─── */
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

/* ─── Metric Card Overrides ─── */
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

/* ─── Sidebar Styling ─── */
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

/* ─── Status Badges ─── */
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

/* ─── Container Cards ─── */
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

/* ─── Table Styling ─── */
.dataframe {
    border-radius: 6px;
    overflow: hidden;
}

/* ─── Tab Styling ─── */
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

/* ─── Primary Button ─── */
.stButton > button {
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 0.5rem 1.5rem;
    transition: all 0.2s ease;
}

/* ─── Callout Notice ─── */
.notice-box {
    background: #0f172a;
    border-left: 3px solid #3b82f6;
    padding: 12px 16px;
    border-radius: 0 6px 6px 0;
    font-size: 0.85rem;
    color: #cbd5e1;
    margin: 12px 0;
}

/* ─── Comparison Table Enhancements ─── */
.comparison-header {
    font-size: 0.95rem;
    font-weight: 600;
    color: #f8fafc;
    margin-bottom: 4px;
}

.comparison-subtitle {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-bottom: 16px;
}

/* ─── Multi-select Chip Styling ─── */
div[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background: rgba(59, 130, 246, 0.15);
    border: 1px solid rgba(59, 130, 246, 0.4);
    color: #60a5fa;
    border-radius: 4px;
    font-size: 0.78rem;
    font-weight: 500;
}
</style>
"""


def inject_styles():
    """Inject the global CSS design system and Streamlit element overrides."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
