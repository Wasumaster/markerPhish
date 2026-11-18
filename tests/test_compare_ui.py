"""
Compare tab AppTests: dataset column and filter, the dataset / sample mismatch
warning.

The runs are made against the fake Ollama and saved to an isolated results
directory; no real Ollama server is contacted.
"""

import os
import re

import pytest
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app.py")
COMPARE = 3


def _text(at, index=COMPARE):
    return " ".join(re.sub(r"<[^>]+>", " ", m.value) for m in at.tabs[index].markdown)


def _save_run(fake_ollama, toy_df, model, dataset_choice, size=20, mtime=None):
    """One protocol-2 run of `model` on a balanced sample labelled with `dataset_choice`."""
    from backend.evaluator import run_evaluation
    from backend.profiler import ResourceSampler
    from backend.run_report import build_run_report
    from backend.sampling import balanced_sample
    from backend.utils import save_results

    sample = balanced_sample(toy_df.assign(source_id=[f"t:{i}" for i in toy_df["id"]]), size)
    fake_ollama.responder = lambda m, p, b: '{"label": "valid"}'
    run = run_evaluation(model, sample.df, num_gpu=0, prompt_variant="zeroshot", tree_rss_reader=lambda: 1.0,
                         sampler_factory=lambda m: ResourceSampler(m, interval_s=0.01, gpu_reader=lambda: (None, None),
                                                                    power_reader=lambda: True))
    report = build_run_report(run, df_sample=sample.df, dataset_choice=dataset_choice, sample_info=sample.info,
                              code={"commit": None, "dirty": None, "changed_files": None})
    path = save_results(report, model, prompt_variant="zeroshot")
    if mtime:
        os.utime(path, (mtime, mtime))
    return os.path.basename(path)


@pytest.fixture
def two_dataset_runs(fake_ollama, results_dir, toy_df):
    """seed+sms: two runs on one sample and one on another sample; sms: the most recent run."""
    files = {
        "a": _save_run(fake_ollama, toy_df, "acme-small:1b", "seed+sms", mtime=3_000_000_000),
        "b": _save_run(fake_ollama, toy_df, "acme-small:7b", "seed+sms", mtime=3_000_000_100),
        "other_sample": _save_run(fake_ollama, toy_df, "omega-big:8b", "seed+sms", size=40, mtime=3_000_000_150),
        "sms": _save_run(fake_ollama, toy_df, "zeta-chat:latest", "sms", mtime=3_000_000_200),
    }
    return files


def _app():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _multiselect(at, label):
    return next(m for m in at.tabs[COMPARE].multiselect if m.label == label)


# ── A. dataset column, filter and mismatch warning ──

def test_dataset_filter_defaults_to_the_most_recent_runs_dataset(two_dataset_runs):
    at = _app()
    assert _multiselect(at, "dataset").value == ["sms"]
    assert sorted(_multiselect(at, "dataset").options) == ["seed+sms", "sms"]
    assert _multiselect(at, "runs").value == [two_dataset_runs["sms"]]
    assert not any("different samples" in w.value for w in at.tabs[COMPARE].warning)


def test_dataset_column_follows_prompt_and_chips_name_the_dataset(two_dataset_runs):
    at = _app()
    text = _text(at)
    assert re.search(r"prompt\s+dataset\s+start", text)  # table header order
    assert "zeta-chat:latest" in text and " sms " in text
    runs = _multiselect(at, "runs")
    assert all("sms" in o for o in runs.options)  # chip labels carry the dataset


def test_mixed_datasets_warn_and_name_the_runs_that_differ(two_dataset_runs):
    at = _app()
    _multiselect(at, "dataset").set_value(["sms", "seed+sms"]).run()
    assert not at.exception, [e.value for e in at.exception]
    assert sorted(_multiselect(at, "runs").value) == sorted(two_dataset_runs.values())  # filter change re-selects
    warning = next(w.value for w in at.tabs[COMPARE].warning if "different samples" in w.value)
    assert "Reference: dataset seed+sms" in warning
    assert "zeta-chat:latest" in warning and "(dataset sms)" in warning
    assert "omega-big:8b" in warning  # same dataset, other sample
    assert "acme-small:7b" not in warning and "acme-small:1b" not in warning
    text = _text(at)
    assert text.count("other sample") >= 2 and "never bold, never recommended" in text


def test_results_header_names_the_dataset(two_dataset_runs):
    at = _app()
    results = _text(at, 2)
    assert "dataset sms" in results
