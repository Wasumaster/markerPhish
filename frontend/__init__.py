"""
MarkerPhish — Frontend Package
================================
Modular Streamlit UI components: styles, sidebar, header, and views.
"""

from frontend.header import render_header
from frontend.multi_comparison import render_multi_comparison_view
from frontend.overview import render_data_overview
from frontend.runner import render_evaluation_runner
from frontend.sidebar import render_sidebar
from frontend.single_run import render_results, render_single_run_view
from frontend.styles import inject_styles

__all__ = [
    "inject_styles",
    "render_sidebar",
    "render_header",
    "render_data_overview",
    "render_evaluation_runner",
    "render_results",
    "render_single_run_view",
    "render_multi_comparison_view",
]
