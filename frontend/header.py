"""
MarkerPhish — Header Component
================================
Renders the top bar: product name on the left, live Ollama status on the right.
"""

from backend.evaluator import check_ollama_connection
from config import OLLAMA_BASE_URL
from ui.components import top_bar


def render_header():
    """Render the full-width top bar with the real Ollama endpoint and status."""
    top_bar(online=check_ollama_connection(), endpoint=OLLAMA_BASE_URL)
