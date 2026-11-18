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
    app.toggle(key="compare_show_hidden").set_value(True).run()  # the stored runs are all legacy
    datasets = next(m for m in app.multiselect if m.label == "dataset")
    datasets.set_value(list(datasets.options)).run()  # the filter offers only the newest run's dataset by default
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


def test_results_tab_shows_the_manifest_of_a_new_run(fake_ollama, legacy_results_dir, toy_df):
    import re

    from backend.evaluator import run_evaluation
    from backend.run_report import build_run_report
    from backend.sampling import balanced_sample
    from backend.utils import save_results

    df = toy_df.assign(source_id=[f"t:{i}" for i in toy_df["id"]], group_id=[f"g:{i}" for i in toy_df["id"]])
    sample = balanced_sample(df, 20)
    run = run_evaluation("acme-small:1b", sample.df, num_gpu=0, prompt_variant="zeroshot")
    report = build_run_report(
        run, df_sample=sample.df, dataset_choice="seed+sms", sample_info=sample.info,
        code={"commit": "0" * 40, "dirty": False, "changed_files": []},
    )
    path = save_results(report, "acme-small:1b", prompt_variant="zeroshot")
    newest = os.path.basename(path)
    os.utime(path, (4_000_000_000, 4_000_000_000))  # newest file -> selected by default

    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["loaded_filename"] == newest
    results_text = " ".join(re.sub(r"<[^>]+>", " ", m.value) for m in at.tabs[2].markdown)
    assert "dataset fingerprint" in results_text
    assert "differences from the current environment" in results_text
    assert "code_commit" in results_text  # the fake commit differs from HEAD
    assert any("{{MESSAGE}}" in c.value for c in at.tabs[2].code)


def test_results_tab_marks_legacy_runs(app):
    import re

    text = " ".join(re.sub(r"<[^>]+>", " ", m.value) for m in app.tabs[2].markdown)
    assert "legacy protocol (protocol 1)" in text


def test_run_tab_warns_about_other_loaded_models(fake_ollama, legacy_results_dir):
    fake_ollama.load("omega-big:8b")
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception
    assert any("Other models are loaded" in w.value and "omega-big:8b" in w.value for w in at.warning)
    box = next(c for c in at.checkbox if c.label.startswith("unload other models"))
    assert box.value is True
    assert next(r for r in at.radio if r.label == "start").value == "warm"


def test_results_tab_shows_power_warning_and_resources(fake_ollama, legacy_results_dir, toy_df, monkeypatch):
    import re

    from backend import evaluator
    from backend.profiler import ResourceSampler
    from backend.run_report import build_run_report
    from backend.sampling import balanced_sample
    from backend.utils import save_results

    states = iter([{"on_ac_power": True}, {"on_ac_power": False}])
    monkeypatch.setattr(evaluator, "power_state", lambda: next(states))
    sample = balanced_sample(toy_df.assign(source_id=[f"t:{i}" for i in toy_df["id"]]), 20)
    run = evaluator.run_evaluation(
        "acme-small:1b", sample.df, num_gpu=0, prompt_variant="zeroshot", tree_rss_reader=lambda: 10.0,
        sampler_factory=lambda m: ResourceSampler(m, interval_s=0.01, gpu_reader=lambda: (None, None),
                                                  power_reader=lambda: None),
    )
    report = build_run_report(run, df_sample=sample.df, dataset_choice="seed+sms", sample_info=sample.info,
                              code={"commit": None, "dirty": None, "changed_files": None})
    path = save_results(report, "acme-small:1b", prompt_variant="zeroshot")
    os.utime(path, (4_000_000_000, 4_000_000_000))

    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("AC power changed during this run" in w.value for w in at.tabs[2].warning)
    text = " ".join(re.sub(r"<[^>]+>", " ", m.value) for m in at.tabs[2].markdown)
    assert "model memory (Ollama)" in text and "fits laptop-8gb" in text


def _tab_text(at, index):
    import re

    return " ".join(re.sub(r"<[^>]+>", " ", m.value) for m in at.tabs[index].markdown)


def test_results_shows_both_unknown_modes_with_confidence_intervals(fake_ollama, legacy_results_dir, toy_df):
    import re

    from backend.evaluator import run_evaluation
    from backend.profiler import ResourceSampler
    from backend.run_report import build_run_report
    from backend.sampling import balanced_sample
    from backend.utils import save_results

    sample = balanced_sample(toy_df.assign(source_id=[f"t:{i}" for i in toy_df["id"]]), 20)
    run = run_evaluation(
        "acme-small:1b", sample.df, num_gpu=0, prompt_variant="zeroshot", tree_rss_reader=lambda: 10.0,
        sampler_factory=lambda m: ResourceSampler(m, interval_s=0.01, gpu_reader=lambda: (None, None),
                                                  power_reader=lambda: True),
    )
    report = build_run_report(run, df_sample=sample.df, dataset_choice="seed+sms", sample_info=sample.info,
                              code={"commit": None, "dirty": None, "changed_files": None})
    path = save_results(report, "acme-small:1b", prompt_variant="zeroshot")
    os.utime(path, (4_000_000_000, 4_000_000_000))

    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    text = _tab_text(at, 2)
    assert "unknown = error" in text and "excluding unknown" in text
    assert re.search(r"\d\.\d{3} \[\d\.\d{3}; \d\.\d{3}\]", text)  # value [low; high]
    assert "per channel" in text and "generation tokens/s" in text
    assert not any("Measured before the Ollama URL fix" in w.value for w in at.tabs[2].warning)
    compare = _tab_text(at, 3)
    assert "F1 [95% CI]" in compare and "Latency inflated" not in compare  # legacy runs hidden by default
    assert "6 legacy or dev run(s) hidden" in compare
    at.toggle(key="compare_show_hidden").set_value(True).run()
    compare = _tab_text(at, 3)
    assert "Latency inflated" in compare and "never bold, never recommended" in compare


def test_legacy_run_in_results_is_flagged_as_latency_inflated(app):
    assert any("Measured before the Ollama URL fix" in w.value for w in app.tabs[2].warning)


def test_invalid_advanced_option_disables_the_run(app):
    app.number_input(key="opt_num_ctx").set_value(1024).run()  # too small for the longest prompt
    assert not app.exception
    assert any("does not fit" in e.value for e in app.error)
    assert next(b for b in app.button if b.label == "Run evaluation").disabled


def test_missing_calibration_is_only_a_note(app):
    import re

    text = " ".join(re.sub(r"<[^>]+>", " ", m.value) for m in app.tabs[1].markdown)
    assert "No calibration for" in text and "python -m backend.calibrate" in text
    assert not next(b for b in app.button if b.label == "Run evaluation").disabled


def test_end_to_end_run_from_the_ui(fake_ollama, results_dir):
    """Pick a model, choose 50 records, start: a complete report is saved and shown."""
    import json

    fake_ollama.responder = lambda m, p, b: '{"label": "malicious"}' if "verify" in p.lower() else '{"label": "valid"}'
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    next(s for s in at.sidebar.selectbox if s.label == "model").set_value("zeta-chat:latest")
    next(s for s in at.sidebar.selectbox if s.label == "sample size").set_value(50)
    at.run()
    next(b for b in at.button if b.label == "Run evaluation").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Evaluation session completed" in m.value for m in at.success)
    files = [f for f in os.listdir(results_dir) if f.endswith(".json")]
    assert len(files) == 1
    report = json.load(open(results_dir / files[0], encoding="utf-8"))
    assert report["model_name"] == "zeta-chat:latest"
    assert report["manifest"]["dataset"]["sample_size"] == 50
    assert [c["count"] for c in report["manifest"]["dataset"]["cells"]] == [13, 13, 12, 12]
    assert report["detection"]["n_records"] == 50
    assert report["manifest"]["inference"]["options"]["num_predict"] == 256
    assert report["resources"]["start_mode"] == "warm"
    assert at.session_state["loaded_filename"] == files[0]
    assert at.session_state["is_running"] is False


def _save_fake_runs(fake_ollama, toy_df, accuracies):
    """One protocol-2 run per (model, share of correct answers), all on the same sample."""
    import json as _json

    from backend.evaluator import run_evaluation
    from backend.profiler import ResourceSampler
    from backend.run_report import build_run_report
    from backend.sampling import balanced_sample
    from backend.utils import save_results

    sample = balanced_sample(toy_df.assign(source_id=[f"t:{i}" for i in toy_df["id"]]), 40)
    truth = {r.text: r.true_label for r in sample.df.itertuples()}
    texts = list(truth)
    for model, share in accuracies.items():
        right = set(texts[: int(len(texts) * share)])

        def responder(m, prompt, body, right=right):
            message = prompt.split("<message>")[-1].split("</message>")[0].strip()
            malicious = truth.get(message) in ("phishing", "smishing")
            if message not in right:
                malicious = not malicious
            return _json.dumps({"label": "malicious" if malicious else "valid"})

        fake_ollama.responder = responder
        run = run_evaluation(model, sample.df, num_gpu=0, prompt_variant="zeroshot", tree_rss_reader=lambda: 1.0,
                             sampler_factory=lambda m: ResourceSampler(m, interval_s=0.01, gpu_reader=lambda: (None, None),
                                                                        power_reader=lambda: True))
        report = build_run_report(run, df_sample=sample.df, dataset_choice="seed+sms", sample_info=sample.info,
                                  code={"commit": None, "dirty": None, "changed_files": None})
        save_results(report, model, prompt_variant="zeroshot")


def test_compare_recommends_per_profile(fake_ollama, legacy_results_dir, toy_df):
    _save_fake_runs(fake_ollama, toy_df, {"acme-small:1b": 0.9, "acme-small:7b": 0.95, "zeta-chat:latest": 0.6})
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    text = _tab_text(at, 3)
    assert "Recommendation" in text and "laptop-8gb" in text and "laptop-16gb" in text
    assert "acme-small:7b [zeroshot]" in text  # the most accurate model fits both budgets
    assert "not significantly different" in text
    at.toggle(key="compare_show_hidden").set_value(True).run()
    assert "runs excluded from the recommendation (6)" in " ".join(e.label for e in at.tabs[3].expander)
    assert len(at.tabs[3].get("plotly_chart")) >= 3

    # editing a profile changes its budget
    at.number_input(key="prof_ram_laptop-8gb").set_value(2.0).run()
    assert "1,024.0 MB" in _tab_text(at, 3)  # 2 GB x headroom 0.5



def test_baseline_button_saves_a_baseline_run(fake_ollama, results_dir):
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    next(s for s in at.sidebar.selectbox if s.label == "sample size").set_value(50)
    at.run()
    next(b for b in at.button if b.label == "Run TF-IDF baseline on this sample").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Baseline saved" in m.value for m in at.success)
    files = os.listdir(results_dir)
    assert len(files) == 1 and files[0].startswith("eval_baseline-tfidf-lr_baseline_")
    assert fake_ollama.generate_calls() == []  # the baseline never talks to Ollama


def test_compare_marks_hidden_runs_and_warns_about_mixed_samples(fake_ollama, legacy_results_dir, toy_df):
    _save_fake_runs(fake_ollama, toy_df, {"acme-small:1b": 0.9})
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not any("different samples" in w.value for w in at.tabs[3].warning)  # one sample only
    at.toggle(key="compare_show_hidden").set_value(True).run()
    assert not at.exception
    assert any("different samples" in w.value for w in at.tabs[3].warning)  # legacy runs have no fingerprint
    text = _tab_text(at, 3)
    assert text.count(" legacy ") >= 6


def test_run_tab_shows_cells_purpose_and_estimate(fake_ollama, legacy_results_dir):
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    next(s for s in at.sidebar.selectbox if s.label == "sample size").set_value(50)
    at.run()
    text = _tab_text(at, 1)
    assert "50 · smoke test, not for results" in text
    assert "email phishing 13 · email valid 13 · sms smishing 12 · sms valid 12" in text
    assert "estimated time" in text and "rough: no earlier run" in text


def test_results_shows_timeline_and_score_curves_for_a_scored_run(fake_ollama, results_dir):
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    next(s for s in at.sidebar.selectbox if s.label == "sample size").set_value(50)
    at.run()
    next(b for b in at.button if b.label == "Run TF-IDF baseline on this sample").click().run()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    text = _tab_text(at, 2)
    assert "RAM / CPU over time" in text and "Scores" in text
    assert len(at.tabs[2].get("plotly_chart")) >= 3

