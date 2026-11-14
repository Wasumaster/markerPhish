"""backend.stats and backend.detection: metrics, unknown modes, bootstrap CI, McNemar, timing."""

import math

import numpy as np
import pytest

from backend.detection import compare_runs, compute_detection, compute_timing, request_status
from backend.stats import bootstrap_ci, bootstrap_weights, counts, holm, mcnemar, point_metrics, rates

TRUE = np.array([True, True, True, True, False, False, False, False])
PRED = np.array(["malicious", "malicious", "safe", "unknown", "safe", "malicious", "unknown", "safe"], dtype=object)


def test_counts_excluding_unknown():
    c = counts(TRUE, PRED, "excluding_unknown")
    assert (c["tp"], c["fn"], c["fp"], c["tn"], c["unknown"]) == (2, 1, 1, 2, 2)


def test_counts_unknown_as_error():
    c = counts(TRUE, PRED, "unknown_as_error")
    # unknown on a threat = missed threat (FN); unknown on a safe message = false alarm (FP)
    assert (c["tp"], c["fn"], c["fp"], c["tn"]) == (2, 2, 2, 2)


def test_rates_and_formulas():
    m = point_metrics(TRUE, PRED, "unknown_as_error")
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5
    assert m["fpr"] == 0.5 and m["fnr"] == 0.5
    m2 = point_metrics(TRUE, PRED, "excluding_unknown")
    assert m2["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert m2["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert m2["evaluated"] == 6 and m2["support"] == 8


def test_undefined_rates_are_none_and_f1_zero_when_nothing_detected():
    m = point_metrics(np.array([True, False]), np.array(["safe", "safe"], dtype=object), "unknown_as_error")
    assert m["precision"] is None  # nothing flagged
    assert m["recall"] == 0.0 and m["f1"] == 0.0
    only_safe = point_metrics(np.array([False]), np.array(["safe"], dtype=object), "unknown_as_error")
    assert only_safe["recall"] is None and only_safe["f1"] is None and only_safe["fpr"] == 0.0


def test_rates_vectorised():
    r = rates({"tp": np.array([1.0, 0.0]), "fn": np.array([1.0, 0.0]), "fp": np.array([0.0, 0.0]),
               "tn": np.array([2.0, 2.0])})
    assert r["recall"][0] == 0.5 and math.isnan(r["recall"][1])


# ── bootstrap ──
def test_bootstrap_weights_preserve_stratum_sizes_and_are_deterministic():
    strata = np.array(["a"] * 6 + ["b"] * 4, dtype=object)
    w = bootstrap_weights(strata, None, n_resamples=50, seed=1)
    assert w.shape == (50, 10)
    assert np.all(w[:, :6].sum(axis=1) == 6) and np.all(w[:, 6:].sum(axis=1) == 4)
    assert np.array_equal(w, bootstrap_weights(strata, None, n_resamples=50, seed=1))


def test_bootstrap_is_group_aware():
    strata = np.array(["a"] * 6, dtype=object)
    groups = np.array(["g1", "g1", "g1", "g2", "g2", "g3"], dtype=object)
    w = bootstrap_weights(strata, groups, n_resamples=200, seed=2)
    # members of one group always carry the same weight
    assert np.all(w[:, 0] == w[:, 1]) and np.all(w[:, 1] == w[:, 2]) and np.all(w[:, 3] == w[:, 4])


def test_bootstrap_ci_contains_the_estimate_and_narrows_with_n():
    rng = np.random.default_rng(0)

    def ci_width(n):
        is_mal = np.array([True] * (n // 2) + [False] * (n // 2))
        correct = rng.random(n) < 0.85
        right = np.where(is_mal, "malicious", "safe")
        wrong = np.where(is_mal, "safe", "malicious")
        pred = np.where(correct, right, wrong).astype(object)
        strata = np.where(is_mal, "m", "s").astype(object)
        w = bootstrap_weights(strata, None, 500, 3)
        ci = bootstrap_ci(is_mal, pred, w, "unknown_as_error")["f1"]
        f1 = point_metrics(is_mal, pred, "unknown_as_error")["f1"]
        assert ci["low"] <= f1 <= ci["high"]
        return ci["high"] - ci["low"]

    assert ci_width(1000) < ci_width(100)


# ── McNemar & Holm ──
def test_mcnemar_exact_small_counts():
    a = np.array([True] * 10 + [False] * 2 + [True] * 50)
    b = np.array([False] * 10 + [True] * 2 + [True] * 50)
    r = mcnemar(a, b)
    assert (r["only_a_correct"], r["only_b_correct"], r["method"]) == (10, 2, "exact")
    assert r["p_value"] == pytest.approx(0.038574, abs=1e-5)


def test_mcnemar_chi2_large_counts():
    a = np.array([True] * 30 + [False] * 10)
    b = np.array([False] * 30 + [True] * 10)
    r = mcnemar(a, b)
    assert r["method"] == "chi2_cc" and r["statistic"] == pytest.approx(9.025)
    assert r["p_value"] == pytest.approx(0.002663, abs=1e-5)


def test_mcnemar_identical_runs():
    a = np.array([True, False, True])
    assert mcnemar(a, a)["p_value"] == 1.0


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([]) == []
    assert holm([0.6, 0.9]) == pytest.approx([1.0, 1.0])


# ── detection block ──
def _raw(true, pred, channel, gid=None, sid=None):
    return {"true_binary": true, "predicted_binary": pred, "channel": channel, "group_id": gid, "source_id": sid}


def test_compute_detection_overall_and_per_channel():
    raw = (
        [_raw("malicious", "malicious", "email", sid=f"e{i}") for i in range(8)]
        + [_raw("malicious", "unknown", "email", sid="e8")]
        + [_raw("safe", "safe", "email", sid=f"ev{i}") for i in range(9)]
        + [_raw("malicious", "malicious", "sms", sid=f"s{i}") for i in range(5)]
        + [_raw("safe", "malicious", "sms", sid=f"sv{i}") for i in range(5)]
    )
    d = compute_detection(raw, n_resamples=200)
    assert d["unknown_predictions"] == 1
    assert d["overall"]["unknown_as_error"]["fn"] == 1
    assert d["overall"]["excluding_unknown"]["fn"] == 0
    email = d["per_channel"]["email"]["unknown_as_error"]
    sms = d["per_channel"]["sms"]["unknown_as_error"]
    assert email["fpr"] == 0.0 and sms["fpr"] == 1.0
    assert email["support"] == 18 and sms["support"] == 10
    ci = d["overall"]["unknown_as_error"]["ci"]["f1"]
    assert ci["low"] <= d["overall"]["unknown_as_error"]["f1"] <= ci["high"]


def test_compute_detection_on_legacy_records():
    raw = [{"true": "phishing", "predicted": "smishing"}, {"true": "valid", "predicted": "unknown"},
           {"predicted": "valid"}]
    d = compute_detection(raw, n_resamples=50)
    assert d["excluded_unknown_ground_truth"] == 1
    assert d["overall"]["unknown_as_error"]["tp"] == 1 and d["overall"]["unknown_as_error"]["fp"] == 1
    assert "email" in d["per_channel"]  # phishing -> email


def test_compute_detection_empty():
    assert compute_detection([])["overall"] == {}


# ── timing ──
def test_timing_excludes_failures_and_cold_request():
    raw = [
        {"latency_s": 9.0, "is_cold_request": True, "request_status": "ok"},
        {"latency_s": 2.0, "request_status": "ok", "prompt_eval_count": 1000, "eval_count": 10,
         "prompt_eval_duration_s": 1.0, "eval_duration_s": 0.5, "total_duration_s": 1.6},
        {"latency_s": 4.0, "request_status": "ok", "prompt_eval_count": 1000, "eval_count": 10,
         "prompt_eval_duration_s": 3.0, "eval_duration_s": 0.5, "total_duration_s": 3.6},
        {"latency_s": 300.0, "raw_response": "[TIMEOUT]"},
        {"latency_s": 0.1, "raw_response": "[ERROR: boom]"},
    ]
    t = compute_timing(raw, wall_time_s=400.0)
    assert t["failed_requests"] == 2 and t["timeouts"] == 1 and t["errors"] == 1
    assert t["failure_rate"] == 0.4
    assert t["cold_requests_excluded"] == 1
    assert t["latency_s"]["mean"] == 3.0 and t["latency_s"]["max"] == 4.0
    assert t["throughput_msgs_per_min"] == pytest.approx(2 / 6 * 60)
    assert t["prefill_tokens_per_s"] == 500.0 and t["generation_tokens_per_s"] == 20.0
    assert t["wall_throughput_msgs_per_min"] == pytest.approx(5 / 400 * 60)


def test_request_status_from_old_records():
    assert request_status({"raw_response": "[CONNECTION_ERROR: x]"}) == "error"
    assert request_status({"raw_response": '{"label": "valid"}'}) == "ok"


# ── pairwise ──
def _report(fp, preds):
    return {
        "manifest": {"dataset": {"dataset_fingerprint": fp}},
        "raw_predictions": [_raw(t, p, "email", sid=str(i)) for i, (t, p) in enumerate(preds)],
    }


def test_compare_runs_requires_the_same_fingerprint():
    a = _report("fp1", [("malicious", "malicious")])
    assert compare_runs(a, _report("fp2", [("malicious", "safe")]))["comparable"] is False
    assert "legacy" in compare_runs(a, {"raw_predictions": []})["reason"]


def test_compare_runs_pairs_by_source_id():
    a = _report("fp", [("malicious", "malicious")] * 12 + [("safe", "safe")] * 10)
    b = _report("fp", [("malicious", "unknown")] * 12 + [("safe", "safe")] * 10)
    r = compare_runs(a, b)
    assert r["comparable"] and r["only_a_correct"] == 12 and r["only_b_correct"] == 0
    assert r["p_value"] < 0.001
