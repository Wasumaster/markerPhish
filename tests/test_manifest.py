"""backend.manifest and backend.run_report: the reproducibility manifest."""

import json

import pytest

from backend import manifest as mf
from backend.evaluator import build_output_schema, run_evaluation
from backend.run_report import build_run_report
from backend.sampling import balanced_sample
from config import DEV_BUILD, OLLAMA_SEED, PREPROCESSING_VERSION, PROTOCOL_VERSION


@pytest.fixture
def sample(toy_df):
    df = toy_df.assign(source_id=[f"t:{i}" for i in toy_df["id"]], group_id=[f"g:{i}" for i in toy_df["id"]])
    return balanced_sample(df, 20)


# ── dataset ──
def test_fingerprint_ignores_order_and_detects_changes(sample):
    df = sample.df
    fp = mf.dataset_fingerprint(df)
    assert fp == mf.dataset_fingerprint(df.sample(frac=1, random_state=3))
    changed_text = df.copy()
    changed_text.loc[0, "text"] += "!"
    assert mf.dataset_fingerprint(changed_text) != fp
    changed_label = df.copy()
    changed_label.loc[0, "true_label"] = "valid" if df.loc[0, "true_label"] != "valid" else "phishing"
    assert mf.dataset_fingerprint(changed_label) != fp
    assert len(fp) == 64


def test_sampled_records_keep_ids_in_order(sample):
    recs = mf.sampled_records(sample.df)
    assert [r["id"] for r in recs] == list(range(1, 21))
    assert all(r["source_id"].startswith("t:") and r["group_id"].startswith("g:") for r in recs)


# ── prompt ──
def test_prompt_template_and_hash():
    schema = build_output_schema("fewshotcot")
    t = mf.prompt_template("fewshotcot", schema)
    assert set(t["templates"]) == {"email", "sms"}
    assert all(mf.MESSAGE_PLACEHOLDER in text for text in t["templates"].values())
    h = mf.prompt_template_hash(t)
    assert h == mf.prompt_template_hash(mf.prompt_template("fewshotcot", schema))
    assert h != mf.prompt_template_hash(mf.prompt_template("fewshot", schema))
    assert h != mf.prompt_template_hash(mf.prompt_template("fewshotcot", build_output_schema("zeroshot")))


# ── software ──
def test_code_version_ignores_saved_runs(monkeypatch):
    outputs = {"rev-parse": "abc123\n", "status": " M results/eval_x.json\n?? results/eval_y.json\n"}
    monkeypatch.setattr(mf, "_git", lambda args: outputs[args[0]])
    assert mf.code_version() == {"commit": "abc123", "dirty": False, "changed_files": []}
    outputs["status"] += " M backend/metrics.py\n"
    v = mf.code_version()
    assert v["dirty"] is True and v["changed_files"] == ["backend/metrics.py"]


def test_code_version_without_git_is_null(monkeypatch):
    monkeypatch.setattr(mf, "_git", lambda args: None)
    assert mf.code_version() == {"commit": None, "dirty": None, "changed_files": None}


def test_git_failure_never_raises(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("git not installed")

    monkeypatch.setattr(mf.subprocess, "run", boom)
    assert mf._git(["status"]) is None


def test_ollama_version_and_model_details(fake_ollama):
    assert mf.ollama_version() == "0.40.13"
    d = mf.model_details("acme-small:1b")
    assert d["tag"] == "1b"
    assert d["digest"] and len(d["digest"]) == 64
    assert d["quantization"] == "Q4_K_M" and d["parameter_size"] == "1.5B"
    assert d["context_length"] == 32768 and d["parameter_count"] == 1_543_714_304
    assert mf.model_details("zeta-chat")["digest"] is not None  # untagged = :latest


def test_offline_ollama_gives_nulls(fake_ollama):
    fake_ollama.online = False
    assert mf.ollama_version() is None
    d = mf.model_details("acme-small:1b")
    assert d["digest"] is None and d["quantization"] is None


# ── power ──
def test_power_state_shape_and_failure_tolerance(monkeypatch):
    state = mf.power_state()
    assert set(state) == {"on_ac_power", "battery_percent", "power_plan", "power_mode"}

    import psutil

    def broken():
        raise RuntimeError("no sensor")

    monkeypatch.setattr(psutil, "sensors_battery", broken)
    state = mf.power_state()
    assert state["on_ac_power"] is None and state["battery_percent"] is None


def test_power_state_reads_battery(monkeypatch):
    import psutil

    class Battery:
        percent = 55.0
        power_plugged = False

    monkeypatch.setattr(psutil, "sensors_battery", lambda: Battery())
    state = mf.power_state()
    assert state["on_ac_power"] is False and state["battery_percent"] == 55.0


# ── full report ──
def test_run_report_contains_a_complete_manifest(fake_ollama, sample):
    run = run_evaluation("acme-small:1b", sample.df, num_gpu=0, prompt_variant="fewshot")
    report = build_run_report(
        run,
        df_sample=sample.df,
        dataset_choice="seed+sms",
        sample_info=sample.info,
        code={"commit": "deadbeef", "dirty": False, "changed_files": []},
    )
    json.dumps(report)  # serializable
    m = report["manifest"]
    assert m["protocol_version"] == PROTOCOL_VERSION == 2
    assert m["dev"] is DEV_BUILD
    assert m["preprocessing_version"] == PREPROCESSING_VERSION
    ds = m["dataset"]
    assert ds["dataset_choice"] == "seed+sms" and ds["sample_size"] == 20 and ds["sample_seed"] == 42
    assert [c["count"] for c in ds["cells"]] == [5, 5, 5, 5]
    assert ds["dataset_fingerprint"] == mf.dataset_fingerprint(sample.df)
    assert len(ds["records"]) == 20
    assert m["prompt"]["prompt_variant"] == "fewshot"
    assert m["prompt"]["prompt_template_hash"] == mf.prompt_template_hash(
        mf.prompt_template("fewshot", build_output_schema("fewshot"))
    )
    assert m["model"]["digest"] and m["model"]["quantization"] == "Q4_K_M"
    opts = m["inference"]["options"]
    assert opts["temperature"] == 0.0 and opts["seed"] == OLLAMA_SEED and opts["num_gpu"] == 0
    assert m["inference"]["options_changed_requests"] == 0
    assert m["software"]["ollama_version"] == "0.40.13"
    assert m["software"]["code"]["commit"] == "deadbeef"
    assert set(m["power"]) == {"power_at_start", "power_at_end", "power_changed", "on_battery_during_run"}
    assert set(m["power"]["power_at_start"]) == {"on_ac_power", "battery_percent", "power_plan", "power_mode"}
    started, ended = m["timestamps"]["started_at"], m["timestamps"]["ended_at"]
    assert started <= ended and ("+" in started or "-" in started[10:])  # has a timezone offset
    # every request carried the seed
    assert all(b["options"]["seed"] == OLLAMA_SEED for b in fake_ollama.generate_calls())
    assert [r["source_id"] for r in report["raw_predictions"]] == [r["source_id"] for r in ds["records"]]


# ── reading saved runs ──
def test_protocol_and_dev_flags_of_saved_runs():
    assert mf.report_protocol_version({}) == 1
    assert mf.report_protocol_version({"manifest": {"protocol_version": 2}}) == 2
    assert mf.report_protocol_version({"manifest": {"protocol_version": "x"}}) == 1
    assert mf.report_is_dev({"manifest": {"dev": True}}) is True
    assert mf.report_is_dev({}) is False


def test_manifest_differences():
    manifest = {
        "preprocessing_version": "3",
        "software": {"code": {"commit": "aaa", "dirty": False}},
        "model": {"digest": "d1"},
        "prompt": {"prompt_template_hash": "h1"},
    }
    same = {"code_commit": "aaa", "code_dirty": False, "model_digest": "d1",
            "preprocessing_version": "3", "prompt_template_hash": "h1"}
    assert mf.manifest_differences(manifest, same) == []
    changed = dict(same, code_commit="bbb", model_digest="d2", code_dirty=None)
    assert mf.manifest_differences(manifest, changed) == [("code_commit", "aaa", "bbb"), ("model_digest", "d1", "d2")]


def test_manifest_summary_handles_partial_manifests():
    rows = dict(mf.manifest_summary({"protocol_version": 2, "dev": True}))
    assert rows["protocol"] == "2 (dev)"
    assert rows["model digest"] == "not recorded"


def test_model_without_metadata_gives_nulls_not_errors(fake_ollama):
    d = mf.model_details("bare-model:latest")
    assert d["name"] == "bare-model:latest" and d["tag"] == "latest"
    for key in ("family", "parameter_size", "parameter_count", "quantization", "context_length", "digest"):
        assert d[key] is None, key
    assert mf.model_details("not-installed:x")["size_bytes"] is None


def test_runs_made_from_now_on_are_protocol_2_and_not_dev():
    import config

    assert config.PROTOCOL_VERSION == 2 and config.DEV_BUILD is False
