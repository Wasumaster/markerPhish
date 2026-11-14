"""Evaluator against the fake Ollama: model lookup, request payload, full run."""

import json

import pytest

from backend.evaluator import (
    check_ollama_connection,
    classify_message,
    find_model_size,
    list_available_models,
    run_evaluation,
)


def test_exact_model_name_match():
    models = [
        {"name": "qwen2.5:7b", "size": 7},
        {"name": "qwen2.5:1.5b", "size": 1},
        {"name": "llama3.2:latest", "size": 3},
    ]
    assert find_model_size(models, "qwen2.5:1.5b") == 1
    assert find_model_size(models, "qwen2.5:7b") == 7
    # A bare prefix must not match another tag
    assert find_model_size(models, "qwen2.5") == 0
    assert find_model_size(models, "qwen2.5:1") == 0
    # Untagged name means ":latest"
    assert find_model_size(models, "llama3.2") == 3
    assert find_model_size([], "x") == 0


def test_connection_and_model_listing(fake_ollama):
    assert check_ollama_connection()
    names = [m["name"] for m in list_available_models()]
    assert "qwen2.5:1.5b" in names
    fake_ollama.online = False
    assert not check_ollama_connection()
    assert list_available_models() == []


def test_classify_message_payload_is_deterministic(fake_ollama):
    fake_ollama.responder = lambda model, prompt, body: '{"label": "malicious"}'
    predicted, raw, latency, meta = classify_message(
        "qwen2.5:1.5b", "Pay the customs fee", num_gpu=0, channel="sms", prompt_variant="zeroshot"
    )
    assert predicted == "smishing"
    assert meta["binary_label"] == "malicious"
    body = fake_ollama.generate_calls()[-1]
    assert body["options"]["temperature"] == 0.0
    assert body["options"]["num_gpu"] == 0
    assert body["format"]["properties"]["label"]["enum"] == ["malicious", "valid"]
    assert "think" not in body  # the fake model has no "thinking" capability
    assert "Pay the customs fee" in body["prompt"]


def test_thinking_models_get_think_false(fake_ollama):
    fake_ollama.capabilities = ["completion", "thinking"]
    classify_message("qwen2.5:1.5b", "hi", channel="email", prompt_variant="zeroshot")
    assert fake_ollama.generate_calls()[-1]["think"] is False


def test_http_error_becomes_unknown(fake_ollama, monkeypatch):
    from tests.fake_ollama import FakeResponse

    original = fake_ollama._route

    def route(method, url, body):
        if url.endswith("/api/generate"):
            return FakeResponse({"error": "boom"}, status_code=500)
        return original(method, url, body)

    monkeypatch.setattr(fake_ollama, "_route", route)
    predicted, raw, _, meta = classify_message("qwen2.5:1.5b", "x", num_gpu=0, channel="email")
    assert predicted == "unknown"
    assert raw.startswith("[ERROR")
    assert meta == {}


def test_run_evaluation_end_to_end(fake_ollama, toy_df):
    # Malicious iff the record text contains "phishing" or "smishing" -> perfect classifier
    def responder(model, prompt, body):
        message = prompt.split("<message>")[-1]
        return json.dumps({"label": "malicious" if ("phishing" in message or "smishing" in message) else "valid"})

    fake_ollama.responder = responder
    seen = []
    run = run_evaluation(
        "qwen2.5:1.5b",
        toy_df.head(12),
        num_gpu=0,
        prompt_variant="fewshot",
        progress_callback=lambda cur, tot, res: seen.append(cur),
    )
    assert run.completed == 12 and run.failed == 0
    assert seen == list(range(1, 13))
    assert all(r.is_binary_correct for r in run.results)
    assert run.model_size_bytes == 986_000_000  # exact match, not qwen2.5:7b
    # warm-up request + 12 evaluation requests
    assert len(fake_ollama.generate_calls()) == 13
    assert run.cold_start_s == pytest.approx(2.0)  # the warm-up request loaded the model
