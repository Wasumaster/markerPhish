"""
MarkerPhish — Styles (compatibility shim)
==========================================
The design system now lives in ui/theme.py. inject_styles() is kept so existing
imports (frontend.inject_styles) keep working; it simply injects the theme CSS.
"""

from ui.theme import inject_css


def inject_styles():
    """Inject the global stylesheet (delegates to ui.theme.inject_css)."""
    inject_css()
