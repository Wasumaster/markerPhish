"""Phase 6: experiment files, matrix expansion, estimates and the headless batch runner (fake Ollama)."""

import json
import os

import pytest
import yaml

from backend import batch as batch_mod
from backend import experiment as ex
from backend.utils import list_saved_result_files, load_result_by_filename, save_results

BASE = {"models": ["acme-small:1b", "zeta-chat"], "prompt_variants": ["zeroshot"], "dataset": "seed+sms",
        "sample_size": 20}


def _write(tmp_path, data, name="exp.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return str(path)


# ── parsing ──
def test_parse_defaults():
    e = ex.parse_experiment(dict(BASE))
    assert e.repetitions == 1 and e.start_mode == "warm" and e.seed == 42 and e.unload_other_models is True
    assert e.constraint_profiles == ["laptop-8gb", "laptop-16gb"]


@pytest.mark.parametrize("change, message", [
    ({"colour": "red"}, "Unknown keys"),
    ({"prompt_variants": ["fancy"]}, "Unknown prompt variants"),
    ({"dataset": "everything"}, "Unknown dataset"),
    ({"sample_size": 10}, "sample_size must be"),
    ({"repetitions": 0}, "repetitions must be"),
    ({"start_mode": "hot"}, "start_mode"),
    ({"options": {"top_k": 40}}, "Not configurable"),
    ({"options": {"num_thread": [4, "x"]}}, "options.num_thread"),
    ({"constraint_profiles": ["phone"]}, "Unknown constraint profiles"),
    ({"models": []}, "models must be"),
])
def test_parse_errors(change, message):
    with pytest.raises(ex.ExperimentError, match=message):
        ex.parse_experiment({**BASE, **change})


def test_missing_key():
    with pytest.raises(ex.ExperimentError, match="Missing required key: sample_size"):
        ex.parse_experiment({k: v for k, v in BASE.items() if k != "sample_size"})


def test_all_variants_and_installed_keyword():
    e = ex.parse_experiment({**BASE, "models": "installed", "prompt_variants": "all"})
    assert e.models == "installed" and e.prompt_variants == ["zeroshot", "fewshot", "fewshotcot"]
    assert ex.resolve_models(e, ["m1:1b", "m2:latest"]) == ["m1:1b", "m2:latest"]
    with pytest.raises(ex.ExperimentError, match="lists no models"):
        ex.resolve_models(e, [])


def test_resolve_models_matches_latest_and_reports_missing():
    e = ex.parse_experiment(dict(BASE))
    assert ex.resolve_models(e, ["acme-small:1b", "zeta-chat:latest"]) == ["acme-small:1b", "zeta-chat:latest"]
    with pytest.raises(ex.ExperimentError, match="Not installed in Ollama: zeta-chat"):
        ex.resolve_models(e, ["acme-small:1b"])


def test_load_experiment_errors(tmp_path):
    with pytest.raises(ex.ExperimentError, match="Cannot read"):
        ex.load_experiment(str(tmp_path / "missing.yaml"))
    bad = tmp_path / "bad.yaml"
    bad.write_text("models: [unclosed", encoding="utf-8")
    with pytest.raises(ex.ExperimentError, match="not valid YAML"):
        ex.load_experiment(str(bad))


# ── expansion ──
def test_option_grid():
    assert ex.option_grid({}) == [{}]
    assert ex.option_grid({"num_thread": [4, 8], "num_ctx": 4096}) == [
        {"num_ctx": 4096, "num_thread": 4}, {"num_ctx": 4096, "num_thread": 8}]


def test_expand_cells_and_ids():
    e = ex.parse_experiment({**BASE, "prompt_variants": "all", "repetitions": 3, "options": {"num_thread": [4, 8]}})
    models = ["acme-small:1b", "acme-small:7b"]  # two tags of one model: a quantization / size comparison
    cells = ex.expand_cells(e, models)
    assert len(cells) == 2 * 3 * 2 * 3
    assert len({c.cell_id for c in cells}) == len(cells)
    assert [c.cell_id for c in cells] == [c.cell_id for c in ex.expand_cells(e, models)]  # stable
    assert ex.batch_id(e, models) == ex.batch_id(e, models)
    assert ex.batch_id(e, models) != ex.batch_id(e, models[:1])
    assert cells[0].label().startswith("acme-small:1b | zeroshot | num_thread=4 | rep 1")


def test_estimate_seconds():
    e = ex.parse_experiment(dict(BASE, prompt_variants=["zeroshot", "fewshot"]))
    cells = ex.expand_cells(e, ["m:1", "n:1"])
    secs, quality = ex.estimate_seconds(cells, 100, {("m:1", "zeroshot"): 2.0, ("m:1", "fewshot"): 3.0})
    assert quality == "partly rough"
    assert secs == pytest.approx(100 * (2 + 3 + 2 * ex.ROUGH_SECONDS_PER_RECORD))
    assert ex.estimate_seconds(cells, 10, {})[1] == "rough"


def test_measured_latencies_ignore_legacy_and_inflated_runs():
    good = {"model_name": "m:1", "timing": {"latency_s": {"mean": 2.0}},
            "manifest": {"protocol_version": 2, "prompt": {"prompt_variant": "fewshot"},
                         "software": {"ollama_url": "http://127.0.0.1:11434"}}}
    inflated = json.loads(json.dumps(good))
    inflated["manifest"]["software"]["ollama_url"] = "http://localhost:11434"
    legacy = {"model_name": "m:1", "timing": {"latency_s": {"mean": 9.0}}}
    assert ex.measured_seconds_per_record([good, inflated, legacy]) == {("m:1", "fewshot"): 2.0}


# ── headless runner ──
def _quick_sampler(model_name):
    from backend.profiler import ResourceSampler

    return ResourceSampler(model_name, interval_s=0.01, gpu_reader=lambda: (None, None), power_reader=lambda: True)


@pytest.fixture
def exp_file(tmp_path):
    return _write(tmp_path, {**BASE, "name": "unit", "options": {"num_thread": [1, 2]}})


def test_dry_run_sends_nothing(fake_ollama, results_dir, exp_file):
    lines = []
    plan = batch_mod.run_batch(exp_file, dry_run=True, out=lines.append)
    assert plan["cells"] == 4 and plan["todo"] == 4
    assert fake_ollama.generate_calls() == []
    assert any("Cells: 4 (0 already complete, 4 to run)" in line for line in lines)
    assert any("rough estimate" in line for line in lines)


def test_batch_runs_all_cells_then_resumes_without_rerunning(fake_ollama, results_dir, exp_file):
    summary = batch_mod.run_batch(exp_file, sampler_factory=_quick_sampler, out=lambda *_: None)
    assert [c["status"] for c in summary["cells"]] == ["complete"] * 4
    runs = list_saved_result_files()
    assert len(runs) == 4
    rep = load_result_by_filename(runs[0])
    assert rep["batch"]["batch_id"] == summary["batch_id"] and rep["manifest"]["dataset"]["sample_size"] == 20
    assert {load_result_by_filename(r)["manifest"]["inference"]["options"].get("num_thread") for r in runs} == {1, 2}
    summary_file = os.path.join(results_dir, "batches", f"batch_{summary['batch_id']}.json")
    assert json.load(open(summary_file))["cells"][0]["power_before"] is not None
    assert os.path.isfile(summary_file.replace(".json", ".log"))

    calls = len(fake_ollama.generate_calls())
    lines = []
    again = batch_mod.run_batch(exp_file, sampler_factory=_quick_sampler, out=lines.append)
    assert len(fake_ollama.generate_calls()) == calls  # nothing re-run
    assert all(c["status"] == "complete (earlier)" for c in again["cells"])
    assert any("4 already complete, 0 to run" in line for line in lines)


def test_a_failed_cell_does_not_stop_the_batch(fake_ollama, results_dir, tmp_path, monkeypatch):
    path = _write(tmp_path, dict(BASE))
    real = batch_mod.run_evaluation

    def flaky(**kw):
        if kw["model_name"] == "acme-small:1b":
            raise RuntimeError("model crashed")
        return real(**kw)

    monkeypatch.setattr(batch_mod, "run_evaluation", flaky)
    summary = batch_mod.run_batch(path, sampler_factory=_quick_sampler, out=lambda *_: None)
    assert [c["status"] for c in summary["cells"]] == ["failed", "complete"]
    assert "model crashed" in summary["cells"][0]["error"]
    # a restart retries only the failed cell
    monkeypatch.setattr(batch_mod, "run_evaluation", real)
    again = batch_mod.run_batch(path, sampler_factory=_quick_sampler, out=lambda *_: None)
    assert [c["status"] for c in again["cells"]] == ["complete", "complete (earlier)"]


def test_battery_warning_and_unknown_model(fake_ollama, results_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(batch_mod, "power_state", lambda: {"on_ac_power": False})
    lines = []
    batch_mod.run_batch(_write(tmp_path, dict(BASE)), dry_run=True, out=lines.append)
    assert any("on battery power" in line for line in lines)
    with pytest.raises(ex.ExperimentError, match="Not installed"):
        batch_mod.run_batch(_write(tmp_path, {**BASE, "models": ["nope:1b"]}, "x.yaml"), dry_run=True)


def test_cli_exit_codes(fake_ollama, results_dir, tmp_path, capsys):
    assert batch_mod.main([_write(tmp_path, {**BASE, "dataset": "nope"})]) == 2
    assert "ERROR" in capsys.readouterr().err
    assert batch_mod.main([_write(tmp_path, dict(BASE), "ok.yaml"), "--dry-run"]) == 0


def test_save_results_never_overwrites(results_dir):
    a = save_results({"x": 1}, "m:1", prompt_variant="zeroshot")
    b = save_results({"x": 2}, "m:1", prompt_variant="zeroshot")
    assert a != b and os.path.isfile(a) and os.path.isfile(b)
