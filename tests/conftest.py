"""
Shared pytest fixtures.

Safety rule: the unit test suite never talks to a real Ollama server. The
autouse fixture below makes every real HTTP request fail loudly; tests that
need Ollama use the `fake_ollama` fixture, which replaces requests.get/post.
"""

from __future__ import annotations

import os
import shutil
import sys

import pandas as pd
import pytest
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tests.fake_ollama import FakeOllama  # noqa: E402


@pytest.fixture(autouse=True)
def _block_real_http(monkeypatch):
    """Fail any request that would leave the process (e.g. to a live Ollama)."""

    def _refuse(*args, **kwargs):
        raise RuntimeError("Real HTTP is disabled in unit tests; use the fake_ollama fixture.")

    monkeypatch.setattr(requests.Session, "request", _refuse)
    monkeypatch.setattr(requests, "get", _refuse)
    monkeypatch.setattr(requests, "post", _refuse)


@pytest.fixture(autouse=True)
def _no_real_process_scan(monkeypatch):
    """The profiler never reads the machine's real process list in unit tests."""
    from backend import profiler

    monkeypatch.setattr(profiler, "list_processes", lambda: [])


@pytest.fixture(autouse=True)
def _clear_evaluator_caches():
    """The evaluator caches per-model capabilities; isolate tests from each other."""
    from backend import evaluator

    evaluator._THINKING_SUPPORT_CACHE.clear()
    yield
    evaluator._THINKING_SUPPORT_CACHE.clear()


@pytest.fixture
def fake_ollama(monkeypatch) -> FakeOllama:
    """A fake Ollama server wired into requests.get / requests.post."""
    fake = FakeOllama()
    monkeypatch.setattr(requests, "get", fake.get)
    monkeypatch.setattr(requests, "post", fake.post)
    return fake


@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    """An isolated, empty results directory used by backend.utils."""
    from backend import utils

    target = tmp_path / "results"
    target.mkdir()
    monkeypatch.setattr(utils, "RESULTS_DIR", str(target))
    return target


@pytest.fixture
def legacy_results_dir(results_dir):
    """The isolated results directory filled with copies of the repo's stored runs."""
    src = os.path.join(ROOT, "results")
    if os.path.isdir(src):
        for name in os.listdir(src):
            if name.endswith(".json"):
                shutil.copy(os.path.join(src, name), results_dir / name)
    return results_dir


@pytest.fixture
def toy_df() -> pd.DataFrame:
    """A small unified-schema dataset with all four (channel x class) cells."""
    rows = []
    rid = 1
    for source, label, n in [
        ("phishfuzzer_seed", "phishing", 30),
        ("phishfuzzer_seed", "valid", 25),
        ("mendeley_sms", "smishing", 40),
        ("mendeley_sms", "valid", 60),
    ]:
        for i in range(n):
            rows.append({"id": rid, "text": f"{source} {label} message {i}", "true_label": label, "source": source})
            rid += 1
    return pd.DataFrame(rows)
