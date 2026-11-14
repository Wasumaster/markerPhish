"""Phase 5b: inference options, context check, overrides, calibration, output / input truncation."""

import json

import pandas as pd
import pytest

import config
from backend import calibrate, evaluator, model_overrides
from backend.options import (
    OptionsError,
    differing_options,
    longest_prompt,
    options_label,
    required_num_ctx,
    resolve_options,
    resolved_from_runner,
    run_options,
)
from backend.run_report import build_run_report
from backend.sampling import balanced_sample


@pytest.fixture
def sample(toy_df):
    return balanced_sample(toy_df.assign(source_id=[f"t:{i}" for i in toy_df["id"]]), 20).df


def _resolve(df=None, ctx=32768, cores=16, user=None, model="acme-small:1b", variant="fewshotcot"):
    return resolve_options(model_name=model, prompt_variant=variant, df_sample=df, model_context_length=ctx,
                           logical_cores=cores, num_gpu=0, user=user)


# ── resolution ──
def test_defaults(sample):
    r = _resolve(sample)
    assert r.options == {"temperature": 0.0, "top_p": 1.0, "seed": config.OLLAMA_SEED, "num_ctx": 4096,
                         "num_predict": config.DEFAULT_NUM_PREDICT, "num_gpu": 0}
    assert r.sources["num_thread"] == "ollama default" and r.sources["num_batch"] == "ollama default"
    assert r.chars_per_token_source == "default"
    assert any("No calibration" in n for n in r.notes)  # a note, never an error
    assert r.longest_prompt["source_id"].startswith("t:")


def test_user_options_are_sent(sample):
    r = _resolve(sample, user={"num_ctx": 8192, "num_thread": 4, "num_predict": 100, "num_batch": 256})
    assert (r.options["num_ctx"], r.options["num_thread"], r.options["num_predict"], r.options["num_batch"]) == (
        8192, 4, 100, 256)
    assert set(r.sources.values()) == {"user"}


def test_zero_or_empty_means_default(sample):
    r = _resolve(sample, user={"num_thread": 0, "num_batch": None, "num_ctx": ""})
    assert "num_thread" not in r.options and "num_batch" not in r.options and r.options["num_ctx"] == 4096


@pytest.mark.parametrize("user, message", [
    ({"num_thread": 17}, "more than the 16 logical cores"),
    ({"num_thread": "x"}, "whole number"),
    ({"num_predict": -5}, "at least 1"),
    ({"top_k": 40}, "Not configurable"),
    ({"num_ctx": 65536}, "exceeds the model's context length"),
])
def test_invalid_user_options(sample, user, message):
    with pytest.raises(OptionsError, match=message):
        _resolve(sample, user=user)


def test_num_ctx_never_exceeds_the_model_context(sample):
    r = _resolve(None, ctx=2048)
    assert r.options["num_ctx"] == 2048 and r.sources["num_ctx"] == "capped at model context length"


def test_too_small_model_context_refuses_and_names_the_record(sample):
    with pytest.raises(OptionsError, match=r"Record t:\d+ does not fit"):
        _resolve(sample, ctx=512)


def test_unknown_context_length_does_not_crash(sample):
    r = _resolve(sample, ctx=None)
    assert r.options["num_ctx"] == 4096 and r.model_context_length is None


def test_denser_calibrated_tokenizer_raises_num_ctx(toy_df):
    long_df = toy_df.head(25).assign(
        text=["x" * config.MESSAGE_MAX_CHARS] * 25, source_id=[f"t:{i}" for i in range(25)]
    )
    model_overrides.save_override("dense-model:3b", chars_per_token=1.2)
    r = _resolve(long_df, model="dense-model:3b")
    assert r.chars_per_token == 1.2 and r.chars_per_token_source == "calibrated"
    assert r.sources["num_ctx"] == "auto-raised" and r.options["num_ctx"] > 4096
    assert r.options["num_ctx"] % 512 == 0
    with pytest.raises(OptionsError, match="does not fit"):  # explicit num_ctx is never changed
        _resolve(long_df, model="dense-model:3b", user={"num_ctx": 4096})


def test_default_budget_fits_every_prompt_without_calibration(toy_df):
    long_df = toy_df.head(4).assign(text=["y" * config.MESSAGE_MAX_CHARS] * 4, source_id=list("abcd"))
    r = _resolve(long_df)
    assert r.options["num_ctx"] == config.OLLAMA_NUM_CTX  # never raised with the default estimate


def test_model_override_num_predict(sample):
    model_overrides.save_override("verbose-model:9b", num_predict=512)
    r = _resolve(sample, model="verbose-model:9b")
    assert r.options["num_predict"] == 512 and r.sources["num_predict"] == "model override"
    assert _resolve(sample, model="verbose-model:9b", user={"num_predict": 64}).options["num_predict"] == 64


def test_required_num_ctx():
    assert required_num_ctx(prompt_chars=2400, num_predict=256, chars_per_token=2.4) == 1536  # (1000+256)/0.9
    assert required_num_ctx(prompt_chars=10, num_predict=1, chars_per_token=2.4) == 512


def test_longest_prompt(toy_df):
    df = toy_df.head(3).assign(text=["a", "b" * 50, "c"], source_id=["s1", "s2", "s3"])
    assert longest_prompt(df, "zeroshot")["source_id"] == "s2"
    assert longest_prompt(df.iloc[0:0], "zeroshot") is None


def test_resolved_from_runner():
    assert resolved_from_runner(["llama-server --model m --threads 8 -b 512"]) == {"num_thread": 8, "num_batch": 512}
    assert resolved_from_runner(["llama-server --model m --port 5300"]) == {"num_thread": None, "num_batch": None}


# ── run identity ──
def _rep(**opts):
    return {"manifest": {"inference": {"options": {"temperature": 0.0, "seed": 42, **opts}}}}


def test_differing_options_and_labels():
    a, b = _rep(num_ctx=4096, num_predict=256), _rep(num_ctx=8192, num_predict=256, num_thread=4)
    assert differing_options([a, b]) == ["num_ctx", "num_thread"]
    assert differing_options([a, a]) == [] and differing_options([a]) == []
    assert options_label(a, ["num_ctx", "num_thread"]) == "num_ctx=4096, num_thread=default"
    assert run_options({}) == {}


# ── overrides file ──
def test_overrides_file_is_optional_and_tolerant(tmp_path):
    path = str(tmp_path / "o.json")
    assert model_overrides.load_overrides(path) == {}
    open(path, "w").write("{broken")
    assert model_overrides.load_overrides(path) == {}
    model_overrides.save_override("m", path=path, chars_per_token=2.0)
    assert model_overrides.override_chars_per_token("m:latest", path=path) == 2.0  # m == m:latest
    with pytest.raises(ValueError):
        model_overrides.save_override("m", path=path, colour="red")


# ── evaluation with options ──
def test_output_cut_at_num_predict_is_never_parsed(fake_ollama, sample):
    fake_ollama.eval_count = 999
    fake_ollama.responder = lambda m, p, b: '{"label": "malicious"}'
    run = evaluator.run_evaluation("acme-small:1b", sample, num_gpu=0, prompt_variant="zeroshot",
                                   tree_rss_reader=lambda: 1.0, user_options={"num_predict": 8},
                                   sampler_factory=_sampler)
    assert all(r.truncated_output for r in run.results)
    assert all(r.predicted_label == "unknown" for r in run.results)
    report = build_run_report(run, df_sample=sample, dataset_choice="x", sample_info=None,
                              code={"commit": None, "dirty": None, "changed_files": None})
    assert report["timing"]["truncated_output"] == 20
    assert report["detection"]["unknown_predictions"] == 20
    assert report["manifest"]["inference"]["options"]["num_predict"] == 8
    assert report["manifest"]["inference"]["options_resolution"]["option_sources"]["num_predict"] == "user"


def test_input_reaching_num_ctx_is_flagged(fake_ollama, sample):
    fake_ollama.chars_per_token = 0.05  # absurdly dense: every prompt reaches num_ctx
    run = evaluator.run_evaluation("acme-small:1b", sample, num_gpu=0, prompt_variant="zeroshot",
                                   tree_rss_reader=lambda: 1.0, sampler_factory=_sampler)
    assert all(r.input_possibly_truncated for r in run.results)
    report = build_run_report(run, df_sample=sample, dataset_choice="x", sample_info=None,
                              code={"commit": None, "dirty": None, "changed_files": None})
    assert report["timing"]["input_possibly_truncated"] == 20


def test_normal_inputs_are_not_flagged(fake_ollama, sample):
    fake_ollama.chars_per_token = 3.5
    run = evaluator.run_evaluation("acme-small:1b", sample, num_gpu=0, prompt_variant="fewshot",
                                   tree_rss_reader=lambda: 1.0, sampler_factory=_sampler)
    assert not any(r.input_possibly_truncated or r.truncated_output for r in run.results)
    assert {b["options"]["num_predict"] for b in fake_ollama.generate_calls()} == {1, config.DEFAULT_NUM_PREDICT}


def test_invalid_options_refuse_before_any_request(fake_ollama, sample):
    with pytest.raises(OptionsError):
        evaluator.run_evaluation("acme-small:1b", sample, user_options={"num_thread": 10_000})
    assert fake_ollama.generate_calls() == []


def test_model_without_metadata_runs(fake_ollama, sample):
    run = evaluator.run_evaluation("bare-model:latest", sample, num_gpu=0, prompt_variant="zeroshot",
                                   tree_rss_reader=lambda: 1.0, sampler_factory=_sampler)
    assert run.completed == 20 and run.options_resolution["model_context_length"] is None


def _sampler(model_name):
    from backend.profiler import ResourceSampler

    return ResourceSampler(model_name, interval_s=0.01, gpu_reader=lambda: (None, None), power_reader=lambda: True)


# ── calibration ──
def _cal_df():
    return pd.DataFrame([
        {"source_id": "seed:1", "channel": "email", "text": "From: a\nSubject: s\nLinks: none\n\n" + "w" * 900},
        {"source_id": "seed:2", "channel": "email",
         "text": "From: a\nSubject: s\nLinks: " + " ".join(f"https://x.example/{i}" for i in range(20)) + "\n\nhi"},
        {"source_id": "sms:1", "channel": "sms", "text": "short sms"},
        {"source_id": "sms:2", "channel": "sms", "text": "a longer sms " * 5},
    ])


def test_calibration_cases():
    cases = calibration_cases = calibrate.calibration_cases(_cal_df())
    assert [(c["case"], c["row"]["source_id"]) for c in cases] == [
        ("longest email", "seed:1"), ("longest Links header", "seed:2"), ("longest sms", "sms:2")]
    assert len(calibration_cases) <= 3


def test_chars_per_token_from_uses_the_densest_case():
    assert calibrate.chars_per_token_from([{"chars_per_token": 3.0}, {"chars_per_token": 2.0}], 0.95) == 1.9
    assert calibrate.chars_per_token_from([{"chars_per_token": None}]) is None


def test_calibrate_stores_an_override_with_at_most_three_requests(fake_ollama, monkeypatch):
    fake_ollama.chars_per_token = 2.0  # a denser tokenizer than the global estimate
    from backend import data_loader

    monkeypatch.setattr(data_loader, "load_dataset_with_report", lambda choice: (_cal_df(), {}))
    entry = calibrate.calibrate("acme-small:1b")
    assert entry["chars_per_token"] == pytest.approx(1.9, abs=0.01)
    assert len(entry["measurements"]) == 3
    prompts = [b for b in fake_ollama.generate_calls() if "prompt" in b]
    assert len(prompts) <= 3 and all(b["options"]["num_predict"] == 1 for b in prompts)
    assert "acme-small:1b" not in fake_ollama.loaded  # unloaded at the end
    stored = json.load(open(config.MODEL_OVERRIDES_PATH))
    assert stored["acme-small:1b"]["chars_per_token"] == entry["chars_per_token"]
    assert model_overrides.override_chars_per_token("acme-small:1b") == entry["chars_per_token"]


def test_calibrate_unknown_model(fake_ollama):
    with pytest.raises(SystemExit, match="not installed"):
        calibrate.calibrate("missing-model:1b")
