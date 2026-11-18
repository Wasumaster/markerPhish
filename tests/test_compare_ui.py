"""
Compare tab AppTests: dataset column and filter, the dataset / sample mismatch
warning, the analysis sections behind config flags, and the CSV export.

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


# ── C. analysis sections behind config flags, CSV export ──

_FLAGGED_HEADINGS = {
    "SHOW_RECOMMENDATION": ["Recommendation"],
    "SHOW_TRADEOFF_CHARTS": ["Trade-offs"],
    "SHOW_COMPARE_ANALYSIS": ["By class", "Pareto front and pairwise tests"],
}


def test_hidden_sections_are_absent_by_default(two_dataset_runs):
    import config

    assert not (config.SHOW_RECOMMENDATION or config.SHOW_TRADEOFF_CHARTS or config.SHOW_COMPARE_ANALYSIS)
    at = _app()
    _multiselect(at, "dataset").set_value(["sms", "seed+sms"]).run()
    text = _text(at)
    for headings in _FLAGGED_HEADINGS.values():
        for h in headings:
            assert h not in text
    assert "constraint profiles" not in [e.label for e in at.tabs[COMPARE].expander]
    assert not any(e.label.startswith("runs excluded") for e in at.tabs[COMPARE].expander)
    assert len(at.tabs[COMPARE].get("plotly_chart")) == 0
    # the kept parts are still there
    assert "Compare runs" in text and "F1 [95% CI]" in text and "unknown answer as an error" in text
    assert "Model memory is reported by Ollama" in text


@pytest.mark.parametrize("flag", list(_FLAGGED_HEADINGS))
def test_each_flag_shows_its_section(two_dataset_runs, monkeypatch, flag):
    import config

    monkeypatch.setattr(config, flag, True)
    at = _app()
    _multiselect(at, "dataset").set_value(["sms", "seed+sms"]).run()
    assert not at.exception, [e.value for e in at.exception]
    text = _text(at)
    for h in _FLAGGED_HEADINGS[flag]:
        assert h in text
    others = [h for f, hs in _FLAGGED_HEADINGS.items() if f != flag for h in hs]
    assert not any(h in text for h in others)
    if flag == "SHOW_RECOMMENDATION":
        labels = [e.label for e in at.tabs[COMPARE].expander]
        assert "constraint profiles" in labels
        excluded = next(e for e in at.tabs[COMPARE].expander if e.label.startswith("runs excluded"))
        assert excluded.proto.expanded  # opened because runs on another dataset were excluded
        assert "recommendation uses seed+sms" in text and "excluded from the recommendation" in text
        assert len(at.tabs[COMPARE].get("plotly_chart")) == 0  # Pareto chart belongs to the analysis flag
        assert not any(e.label.startswith("pairwise McNemar") for e in at.tabs[COMPARE].expander)
    if flag == "SHOW_COMPARE_ANALYSIS":
        assert len(at.tabs[COMPARE].get("plotly_chart")) == 2  # per class + Pareto
        assert any(e.label.startswith("pairwise McNemar") for e in at.tabs[COMPARE].expander)
        assert "constraint profiles" not in [e.label for e in at.tabs[COMPARE].expander]
    if flag == "SHOW_TRADEOFF_CHARTS":
        assert len(at.tabs[COMPARE].get("plotly_chart")) == 2


def test_csv_export_still_works(two_dataset_runs):
    import io

    import pandas as pd

    from frontend.multi_comparison import _csv
    from backend.compare import build_comparison
    from frontend.multi_comparison import load_reports

    at = _app()
    buttons = [b for b in at.tabs[COMPARE].get("download_button")]
    assert len(buttons) == 1 and "Export comparison (CSV)" in buttons[0].proto.label

    reports = load_reports(list(two_dataset_runs.values()))
    rows = build_comparison(reports, ["laptop-8gb"])["rows"]
    df = pd.read_csv(io.BytesIO(_csv(rows, ["laptop-8gb"])))
    assert {"dataset", "fingerprint", "sample_mismatch", "f1", "fits_laptop-8gb"} <= set(df.columns)
    assert sorted(df["dataset"]) == ["seed+sms", "seed+sms", "seed+sms", "sms"]
