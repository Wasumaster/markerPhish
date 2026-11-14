"""Phase 9: comparison table rules (backend.compare)."""

from backend.compare import best_values, build_comparison, hidden_reason, visible_runs


def rep(model, f1, protocol=2, dev=False, unknown=0, n=100, fp="fp", vram=0, options=None):
    manifest = {"protocol_version": protocol, "dev": dev, "dataset": {"dataset_fingerprint": fp},
                "prompt": {"prompt_variant": "fewshot"}, "model": {"quantization": "Q8_0"},
                "inference": {"options": {"num_ctx": 4096, **(options or {})}},
                "software": {"ollama_url": "http://127.0.0.1:11434"}}
    out = {
        "model_name": model,
        "detection": {"n_records": n, "unknown_predictions": unknown, "unknown_rate": unknown / n,
                      "overall": {"unknown_as_error": {"f1": f1, "recall": f1, "precision": f1, "fpr": 1 - f1,
                                                       "ci": {"f1": {"low": f1 - 0.05, "high": f1 + 0.05}}}}},
        "timing": {"latency_s": {"p95": 2.0}, "throughput_msgs_per_min": 30.0},
        "resources": {"model_size_mb": 1000.0, "peak_ram_mb": 1200.0, "mean_cpu_pct": 40.0, "start_mode": "warm",
                      "model_vram_mb": vram, "fits_profile": {"laptop-8gb": {"fits_profile": True}}},
    }
    if protocol >= 2:
        out["manifest"] = manifest
    return out


def test_hidden_runs():
    assert hidden_reason(rep("a", 0.8)) is None
    assert hidden_reason(rep("a", 0.8, protocol=1)) == "legacy"
    assert hidden_reason(rep("a", 0.8, dev=True)) == "dev"
    reports = {"a": rep("a", 0.8), "l": rep("l", 0.9, protocol=1)}
    assert set(visible_runs(reports, False)) == {"a"} and set(visible_runs(reports, True)) == {"a", "l"}


def test_legacy_dev_and_all_unknown_runs_are_never_best():
    reports = {"good": rep("good", 0.80), "legacy": rep("old", 0.99, protocol=1), "dev": rep("dev", 0.98, dev=True),
               "mute": rep("mute", 0.95, unknown=100)}
    c = build_comparison(reports, ["laptop-8gb"])
    flags = dict(zip([r["file"] for r in c["rows"]], best_values(c["rows"], "f1", "max")))
    assert flags == {"good": True, "legacy": False, "dev": False, "mute": False}
    assert [r["can_be_best"] for r in c["rows"]] == [True, False, False, False]


def test_min_columns_and_missing_values():
    rows = build_comparison({"a": rep("a", 0.8), "b": rep("b", 0.7)}, [])["rows"]
    rows[1]["p95_latency_s"] = None
    assert best_values(rows, "p95_latency_s", "min") == [True, False]
    assert best_values([dict(rows[0], f1=None)], "f1", "max") == [False]


def test_mixed_samples_gpu_and_differing_options():
    c = build_comparison({"a": rep("a", 0.8, fp="x"), "b": rep("b", 0.8, fp="y")}, [])
    assert c["mixed_samples"] is True and c["has_gpu"] is False
    c = build_comparison({"a": rep("a", 0.8), "b": rep("b", 0.8, vram=512, options={"num_thread": 4})}, ["laptop-8gb"])
    assert c["mixed_samples"] is False and c["has_gpu"] is True
    assert c["differing_options"] == ["num_thread"]
    assert c["rows"][1]["options"] == "num_thread=4" and c["rows"][0]["options"] == "num_thread=default"
    assert c["rows"][0]["fits"] == {"laptop-8gb": True}


def test_many_runs():
    reports = {f"r{i}": rep(f"model-{i}:{i}b", 0.5 + i / 100) for i in range(40)}
    c = build_comparison(reports, ["laptop-8gb", "laptop-16gb"])
    assert len(c["rows"]) == 40 and sum(best_values(c["rows"], "f1", "max")) == 1
