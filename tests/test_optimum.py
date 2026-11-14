"""Phase 7: optimum analysis (eligibility, Pareto front, per-profile winner, McNemar + Holm)."""

import pytest

from backend.detection import compute_detection
from backend.optimum import analyze, eligibility, pareto_front, pairwise_pvalues

PROFILES = {
    "small": {"ram_gb": 8, "gpu": False, "max_p95_latency_s": None},
    "large": {"ram_gb": 16, "gpu": False, "max_p95_latency_s": None},
}
N = 200


def make_report(model, n_correct, memory_mb, p95, fp="fp1", protocol=2, dev=False, start="2026-11-15T10:00:00+01:00",
                options=None, unknown=0):
    """A synthetic run: the first n_correct of N records are right, `unknown` records unanswered."""
    raw = []
    for i in range(N):
        truth = "malicious" if i % 2 == 0 else "safe"
        if i < unknown:
            pred = "unknown"
        elif i < unknown + n_correct:
            pred = truth
        else:
            pred = "safe" if truth == "malicious" else "malicious"
        raw.append({"source_id": f"s{i}", "true_binary": truth, "predicted_binary": pred,
                    "channel": "email" if i % 4 < 2 else "sms"})
    manifest = {"protocol_version": protocol, "dev": dev, "dataset": {"dataset_fingerprint": fp},
                "prompt": {"prompt_variant": "fewshot"}, "model": {"quantization": "Q4_K_M"},
                "inference": {"options": {"temperature": 0.0, "seed": 42, "num_ctx": 4096, **(options or {})}},
                "timestamps": {"started_at": start}}
    if protocol < 2:
        manifest = None
    rep = {"model_name": model, "raw_predictions": raw, "detection": compute_detection(raw, n_resamples=50),
           "resources": {"model_size_mb": memory_mb, "peak_ram_mb": memory_mb * 1.1, "model_vram_mb": 0},
           "timing": {"latency_s": {"p95": p95}}}
    if manifest:
        rep["manifest"] = manifest
    return rep


# ── eligibility ──
def test_eligibility_reasons():
    reports = {
        "a": make_report("m-a:1b", 180, 1000, 2.0),
        "legacy": make_report("m-b:1b", 180, 1000, 2.0, protocol=1),
        "dev": make_report("m-c:1b", 180, 1000, 2.0, dev=True),
        "allunk": make_report("m-d:1b", 0, 1000, 2.0, unknown=N),
        "other": make_report("m-e:1b", 180, 1000, 2.0, fp="fp2"),
    }
    eligible, excluded, ref = eligibility(reports)
    assert set(eligible) == {"a"} and ref == "fp1"
    reasons = {e["file"]: e["reason"] for e in excluded}
    assert "legacy" in reasons["legacy"] and "dev" in reasons["dev"]
    assert "all unknown" in reasons["allunk"] and "different dataset fingerprint" in reasons["other"]


def test_reference_fingerprint_is_the_most_common():
    reports = {"a": make_report("x:1", 150, 1, 1, fp="A"), "b": make_report("y:1", 150, 1, 1, fp="B"),
               "c": make_report("z:1", 150, 1, 1, fp="B")}
    assert eligibility(reports)[2] == "B"


# ── Pareto ──
def test_pareto_front():
    pts = [
        {"file": "best_f1", "f1": 0.9, "model_memory_mb": 5000, "p95_latency_s": 9},
        {"file": "small", "f1": 0.8, "model_memory_mb": 1000, "p95_latency_s": 3},
        {"file": "dominated", "f1": 0.7, "model_memory_mb": 2000, "p95_latency_s": 4},
        {"file": "fast", "f1": 0.75, "model_memory_mb": 1500, "p95_latency_s": 1},
        {"file": "unknown_mem", "f1": 0.99, "model_memory_mb": None, "p95_latency_s": 1},
    ]
    assert pareto_front(pts) == {"best_f1", "small", "fast"}


def test_equal_points_are_both_on_the_front():
    pts = [{"file": "a", "f1": 0.8, "model_memory_mb": 1, "p95_latency_s": 1},
           {"file": "b", "f1": 0.8, "model_memory_mb": 1, "p95_latency_s": 1}]
    assert pareto_front(pts) == {"a", "b"}


# ── per profile ──
def test_winner_per_profile_and_feasibility():
    reports = {
        "big": make_report("big-model:8b", 190, 6000, 8.0),     # only fits the 16 GB profile (budget 8192)
        "mid": make_report("mid-model:3b", 186, 3000, 3.0),
        "tiny": make_report("tiny-model:1b", 150, 900, 1.0),
    }
    out = analyze(reports, PROFILES, headroom=0.5)
    small, large = out["per_profile"]["small"], out["per_profile"]["large"]
    assert small["budget_mb"] == 4096.0 and large["budget_mb"] == 8192.0
    assert small["winner"] == "mid" and large["winner"] == "big"
    assert {"file": "big", "reason": "exceeds the memory budget"} in small["not_feasible"]
    assert set(large["feasible"]) == {"big", "mid", "tiny"}
    assert out["pareto"] == ["big", "mid", "tiny"]


def test_holm_adjusted_mcnemar_marks_close_runs():
    reports = {
        "w": make_report("w:1", 190, 1000, 2.0),
        "close": make_report("c:1", 188, 1000, 2.0),   # 2 discordant records: not significant
        "far": make_report("f:1", 120, 1000, 2.0),     # 70 discordant records: significant
    }
    prof = analyze(reports, {"p": PROFILES["large"]}, 0.5)["per_profile"]["p"]
    assert prof["winner"] == "w"
    comp = {c["file"]: c for c in prof["comparisons"]}
    assert comp["close"]["not_significantly_different"] is True
    assert comp["far"]["not_significantly_different"] is False
    assert comp["far"]["p_holm"] >= comp["far"]["p_value"]  # Holm never lowers a p-value
    assert prof["alternatives_not_significantly_different"] == ["close"]


def test_ties_broken_by_memory_then_latency():
    reports = {"heavy": make_report("h:1", 180, 3000, 1.0), "light": make_report("l:1", 180, 1000, 5.0),
               "light_fast": make_report("lf:1", 180, 1000, 2.0)}
    assert analyze(reports, {"p": PROFILES["large"]}, 0.5)["per_profile"]["p"]["winner"] == "light_fast"


def test_latency_limit_of_a_profile():
    reports = {"slow": make_report("s:1", 190, 1000, 9.0), "quick": make_report("q:1", 170, 1000, 2.0)}
    prof = analyze(reports, {"p": {"ram_gb": 16, "max_p95_latency_s": 5.0}}, 0.5)["per_profile"]["p"]
    assert prof["winner"] == "quick"
    assert {"file": "slow", "reason": "exceeds the latency limit"} in prof["not_feasible"]


def test_unknown_answers_count_as_errors_in_the_ranking():
    # "evasive" refuses the hard half: excluding unknown it would look perfect
    reports = {"evasive": make_report("e:1", 100, 1000, 1.0, unknown=100), "honest": make_report("h:1", 170, 1000, 1.0)}
    assert analyze(reports, {"p": PROFILES["large"]}, 0.5)["per_profile"]["p"]["winner"] == "honest"


def test_differing_options_are_labelled():
    reports = {"t4": make_report("m:1", 180, 1000, 2.0, options={"num_thread": 4}),
               "t8": make_report("m:1", 182, 1000, 1.5, options={"num_thread": 8})}
    out = analyze(reports, PROFILES, 0.5)
    assert out["differing_options"] == ["num_thread"]
    assert out["points"]["t8"]["options"] == "num_thread=8"


def test_many_models_scale(benchmark=None):
    reports = {f"r{i}": make_report(f"model-{i}:1b", 100 + i, 500 + 50 * i, 1 + i / 10) for i in range(30)}
    out = analyze(reports, PROFILES, 0.5)
    assert len(out["points"]) == 30 and out["per_profile"]["large"]["winner"] == "r29"


def test_empty_and_pairwise():
    out = analyze({}, PROFILES, 0.5)
    assert out["per_profile"]["small"]["winner"] is None and out["reference_fingerprint"] is None
    table = pairwise_pvalues({"a": make_report("a:1", 190, 1, 1), "b": make_report("b:1", 120, 1, 1),
                              "x": make_report("x:1", 190, 1, 1, fp="other")})
    assert table[("a", "b")] < 0.001 and table[("a", "x")] is None
    assert table[("a", "b")] == pytest.approx(table[("a", "b")])
