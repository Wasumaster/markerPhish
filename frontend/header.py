"""
MarkerPhish — Header Component
================================
Renders the static main page header with platform branding and academic context.
"""

import streamlit as st


def render_header():
    """Render the static platform header with title, subtitle, and academic tag."""
    st.markdown(
        """
        <div class="platform-header">
            <h1 class="platform-title">MarkerPhish</h1>
            <p class="platform-subtitle">
                Automated evaluation platform for large language models in text-based social engineering attack detection
                (phishing & smishing) with concurrent hardware resource profiling.
            </p>
            <span class="academic-tag">Engineering Thesis Research System | Academic & Business Benchmark</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
