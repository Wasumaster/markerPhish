"""
MarkerPhish — Frontend Views Package
=====================================
Modular view components re-exported for convenience.
"""

from frontend.overview import render_data_overview
from frontend.runner import render_evaluation_runner
from frontend.single_run import render_results, render_single_run_view
from frontend.multi_comparison import render_multi_comparison_view

__all__ = [
    "render_data_overview",
    "render_evaluation_runner",
    "render_results",
    "render_single_run_view",
    "render_multi_comparison_view",
]
