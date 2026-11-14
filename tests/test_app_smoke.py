"""
Streamlit AppTest smoke tests: every tab renders against the fake Ollama.

These tests load the real datasets from Data/ (read-only) but never contact a
real Ollama server and never run an evaluation.
"""

import os

import pytest
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app.py")
TABS = ["Dataset", "Run", "Results", "Compare"]


@pytest.fixture
def app(fake_ollama, legacy_results_dir):
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    return at


def test_all_tabs_render_without_exceptions(app):
    assert not app.exception, [e.value for e in app.exception]
    assert [t.label for t in app.tabs] == TABS


def test_offline_ollama_shows_message_instead_of_crashing(fake_ollama, legacy_results_dir):
    fake_ollama.online = False
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    assert any("offline" in e.value.lower() for e in at.error)


def test_compare_select_all_selects_every_run(app, legacy_results_dir):
    files = sorted(f for f in os.listdir(legacy_results_dir) if f.endswith(".json"))
    runs = next(m for m in app.multiselect if m.label == "runs")
    runs.set_value([files[0]]).run()
    runs = next(m for m in app.multiselect if m.label == "runs")
    assert runs.value == [files[0]]

    next(b for b in app.button if b.label == "select all").click().run()
    assert not app.exception
    runs = next(m for m in app.multiselect if m.label == "runs")
    assert sorted(runs.value) == files


def test_failed_run_resets_is_running(app, monkeypatch):
    import frontend.runner as runner

    def explode(*args, **kwargs):
        raise RuntimeError("simulated failure inside the run")

    monkeypatch.setattr(runner, "_execute_run", explode)
    next(b for b in app.button if b.label == "Run evaluation").click().run()

    assert app.session_state["is_running"] is False
    assert any("simulated failure" in e.value for e in app.error)
    app.run()
    assert not next(b for b in app.button if b.label == "Run evaluation").disabled
