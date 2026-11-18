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


def _with_dataset(report, dataset, started_at=None):
    report["manifest"]["dataset"]["dataset_choice"] = dataset
    if started_at:
        report["manifest"]["timestamps"] = {"started_at": started_at}
    return report


def test_rows_carry_the_dataset_choice_and_legacy_runs_show_a_dash():
    from backend.compare import NO_DATASET, dataset_label

    reports = {"a": _with_dataset(rep("a", 0.8), "seed+sms"), "l": rep("l", 0.9, protocol=1)}
    rows = {r["file"]: r for r in build_comparison(reports, [])["rows"]}
    assert rows["a"]["dataset"] == "seed+sms" and rows["l"]["dataset"] is None
    assert dataset_label(reports["a"]) == "seed+sms" and dataset_label(reports["l"]) == NO_DATASET == "\u2014"


def test_runs_on_another_dataset_are_named_and_never_best():
    reports = {
        "a": _with_dataset(rep("a", 0.80, fp="x"), "seed+sms", "2026-11-01T10:00:00"),
        "b": _with_dataset(rep("b", 0.82, fp="x"), "seed+sms", "2026-11-02T10:00:00"),
        "c": _with_dataset(rep("c", 0.99, fp="y"), "sms", "2026-11-03T10:00:00"),  # most recent, but alone
    }
    c = build_comparison(reports, [])
    assert c["mixed_samples"] is True
    assert c["reference"] == {"dataset": "seed+sms", "fingerprint": "x"}
    assert c["mismatched"] == ["c"]
    flags = dict(zip([r["file"] for r in c["rows"]], best_values(c["rows"], "f1", "max")))
    assert flags == {"a": False, "b": True, "c": False}  # c has the highest F1 but another dataset
    assert [r["sample_mismatch"] for r in c["rows"]] == [False, False, True]


def test_same_dataset_but_other_sample_is_a_mismatch_too():
    reports = {"a": _with_dataset(rep("a", 0.8, fp="x"), "sms", "2026-11-01"),
               "b": _with_dataset(rep("b", 0.9, fp="y"), "sms", "2026-11-02")}
    c = build_comparison(reports, [])
    assert c["reference"] == {"dataset": "sms", "fingerprint": "y"}  # tie: the most recent run
    assert c["mismatched"] == ["a"] and c["datasets"] == ["sms"]


def test_one_sample_has_no_mismatch():
    reports = {"a": _with_dataset(rep("a", 0.8), "seed"), "b": _with_dataset(rep("b", 0.7), "seed")}
    c = build_comparison(reports, [])
    assert c["mismatched"] == [] and c["mixed_samples"] is False
    assert all(r["can_be_best"] for r in c["rows"])


def test_default_dataset_filter_is_the_most_recent_run():
    from frontend.multi_comparison import default_dataset_filter

    reports = {"new": _with_dataset(rep("a", 0.8), "sms"), "old": _with_dataset(rep("b", 0.8), "seed"),
               "legacy": rep("l", 0.8, protocol=1)}
    assert default_dataset_filter(["new", "old", "legacy"], reports) == ["sms"]
    assert default_dataset_filter(["legacy", "new"], reports) == ["\u2014"]
    assert default_dataset_filter([], reports) == []


def test_recommendation_exclusions_name_the_dataset():
    from frontend.recommendation import excluded_rows

    reports = {"c": _with_dataset(rep("c", 0.9, fp="y"), "sms"), "d": _with_dataset(rep("d", 0.9, fp="z"), "seed"),
               "l": rep("l", 0.9, protocol=1)}
    excluded = [{"file": "c", "reason": "different dataset fingerprint (other sample)"},
                {"file": "d", "reason": "different dataset fingerprint (other sample)"},
                {"file": "l", "reason": "no dataset fingerprint"}]
    rows = excluded_rows(excluded, reports, "seed")
    assert rows[0] == ["c", "sms", "different dataset fingerprint (other sample): dataset sms, recommendation uses seed"]
    assert "same dataset (seed) but a different sample" in rows[1][2]
    assert rows[2][1] == "\u2014" and "legacy run without a manifest" in rows[2][2]
