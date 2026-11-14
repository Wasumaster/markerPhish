"""Documentation must describe what the code does: key numbers match config, stale claims stay gone."""

import os

import pytest

import config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return f.read()


@pytest.mark.parametrize("text", [
    f"{config.MESSAGE_MAX_CHARS:,} characters",
    f"{config.PROMPT_TEMPLATE_MAX_CHARS:,} characters",
    f"c = {config.CHARS_PER_TOKEN_ESTIMATE}",
    f"`LINKS_MIN_BODY_CHARS = {config.LINKS_MIN_BODY_CHARS}`",
    f"`LINKS_MOVED_MAX_CHARS = {config.LINKS_MOVED_MAX_CHARS}`",
    f"headroom = {config.MEMORY_HEADROOM_FACTOR}",
    f'`config.PREPROCESSING_VERSION = "{config.PREPROCESSING_VERSION}"`',
    f"{config.DEFAULT_NUM_PREDICT} for every model",
    "8.4%",
    "## 14. Adding a new model",
    "trained **in-distribution**",
    "does **not** mean equivalent",
])
def test_overview_states_the_current_values(text):
    assert text in _read("OVERVIEW.md")


def test_budgets_in_overview_match_the_profiles():
    overview = _read("OVERVIEW.md")
    for profile in config.CONSTRAINT_PROFILES.values():
        budget = int(profile["ram_gb"] * 1024 * config.MEMORY_HEADROOM_FACTOR)
        assert f"{budget:,} MB" in overview


@pytest.mark.parametrize("stale", ["F1-Score (Macro)", "f1_weighted", "KPI", "pie chart", "in real time"])
def test_stale_claims_are_gone(stale):
    assert stale not in _read("OVERVIEW.md") and stale not in _read("README.md")


def test_app_docstring_is_accurate():
    assert "concurrent hardware resource profiling" not in _read("app.py")
