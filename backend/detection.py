"""
MarkerPhish — Detection and timing summaries from stored predictions
=====================================================================
Both summaries are computed from a run's raw_predictions only, at save time and
again on load when the stored metrics_version is older than METRICS_VERSION, so
a later fix of a metric never makes stored runs unusable.

detection: binary metrics in both unknown modes (backend.stats), overall and
per channel (email / SMS), with stratified, group-aware bootstrap CIs.

timing: latency statistics over successful requests only (failed requests and
timeouts are counted separately), excluding the cold-start request; token
counts and tokens/s for prefill and generation from Ollama's own durations;
throughput from the sum of inference latencies; wall-clock throughput kept as a
separate field.
"""

from __future__ import annotations

import numpy as np

from config import channel_for_source

from .stats import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CI_LEVEL,
    UNKNOWN_MODES,
    bootstrap_ci,
    bootstrap_weights,
    point_metrics,
)

# Bump when the computation of the detection or timing summary changes.
METRICS_VERSION = 1
PRIMARY_UNKNOWN_MODE = "unknown_as_error"
CHANNELS = ("email", "sms")

_BINARY_FROM_CHANNEL_LABEL = {"phishing": "malicious", "smishing": "malicious", "valid": "safe"}


def _true_binary(r: dict) -> str | None:
    tb = r.get("true_binary")
    if tb in ("malicious", "safe"):
        return tb
    return _BINARY_FROM_CHANNEL_LABEL.get(str(r.get("true", "")).strip().lower())


def _pred_binary(r: dict) -> str:
    pb = r.get("predicted_binary")
    if pb in ("malicious", "safe"):
        return pb
    mapped = _BINARY_FROM_CHANNEL_LABEL.get(str(r.get("predicted", "")).strip().lower())
    return mapped or "unknown"


def _channel(r: dict) -> str:
    ch = r.get("channel")
    if ch in CHANNELS:
        return ch
    return channel_for_source(None, r.get("true"))


def detection_arrays(raw: list[dict]) -> dict:
    """Aligned numpy arrays of the records with a known ground truth."""
    rows = [(r, _true_binary(r)) for r in raw]
    kept = [(r, tb) for r, tb in rows if tb is not None]
    groups = [str(r.get("group_id") or "") for r, _ in kept]
    return {
        "is_mal": np.array([tb == "malicious" for _, tb in kept], dtype=bool),
        "pred": np.array([_pred_binary(r) for r, _ in kept], dtype=object),
        "channel": np.array([_channel(r) for r, _ in kept], dtype=object),
        "groups": np.array(groups, dtype=object) if any(groups) else None,
        "source_ids": [str(r.get("source_id") or r.get("id")) for r, _ in kept],
        "excluded_unknown_ground_truth": len(rows) - len(kept),
    }


def compute_detection(
    raw: list[dict],
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """The detection block of a run."""
    a = detection_arrays(raw)
    is_mal, pred, channel = a["is_mal"], a["pred"], a["channel"]
    n = len(is_mal)
    out = {
        "metrics_version": METRICS_VERSION,
        "primary_unknown_mode": PRIMARY_UNKNOWN_MODE,
        "n_records": n,
        "unknown_predictions": int(np.sum(pred == "unknown")) if n else 0,
        "unknown_rate": round(float(np.mean(pred == "unknown")), 4) if n else None,
        "excluded_unknown_ground_truth": a["excluded_unknown_ground_truth"],
        "bootstrap": {
            "resamples": n_resamples,
            "seed": seed,
            "level": CI_LEVEL,
            "stratified_by": "class x channel",
            "group_aware": a["groups"] is not None,
        },
        "overall": {},
        "per_channel": {},
    }
    if n == 0:
        return out
    strata = np.array([f"{'m' if m else 's'}|{c}" for m, c in zip(is_mal, channel)], dtype=object)
    weights = bootstrap_weights(strata, a["groups"], n_resamples, seed)
    for mode in UNKNOWN_MODES:
        block = point_metrics(is_mal, pred, mode)
        block["ci"] = bootstrap_ci(is_mal, pred, weights, mode)
        out["overall"][mode] = block
    for ch in CHANNELS:
        mask = channel == ch
        if not mask.any():
            continue
        out["per_channel"][ch] = {}
        for mode in UNKNOWN_MODES:
            block = point_metrics(is_mal[mask], pred[mask], mode)
            block["ci"] = bootstrap_ci(is_mal, pred, weights, mode, mask=mask)
            block["unknown_predictions"] = int(np.sum(pred[mask] == "unknown"))
            out["per_channel"][ch][mode] = block
    return out


# ──────────────────────────────────────────────
# Timing
# ──────────────────────────────────────────────
def request_status(r: dict) -> str:
    """'ok', 'timeout' or 'error' (older runs: derived from the stored raw response)."""
    status = r.get("request_status")
    if status in ("ok", "timeout", "error"):
        return status
    raw = str(r.get("raw_response") or "")
    if raw.startswith("[TIMEOUT"):
        return "timeout"
    if raw.startswith("[ERROR") or raw.startswith("[CONNECTION_ERROR"):
        return "error"
    return "ok"


def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "median": None, "p95": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": round(float(arr.mean()), 4),
        "median": round(float(np.median(arr)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
    }


def _sum(records: list[dict], key: str) -> float | None:
    values = [r.get(key) for r in records if r.get(key) is not None]
    return float(sum(values)) if values else None


def compute_timing(raw: list[dict], wall_time_s: float | None = None) -> dict:
    """The timing block of a run."""
    statuses = [request_status(r) for r in raw]
    ok = [r for r, s in zip(raw, statuses) if s == "ok"]
    timed = [r for r in ok if not r.get("is_cold_request")]
    latencies = [float(r["latency_s"]) for r in timed if r.get("latency_s") is not None]
    sum_latency = float(sum(latencies)) if latencies else None

    prompt_tokens = _sum(timed, "prompt_eval_count")
    output_tokens = _sum(timed, "eval_count")
    prefill_s = _sum(timed, "prompt_eval_duration_s")
    gen_s = _sum(timed, "eval_duration_s")
    n = len(raw)
    failed = n - len(ok)
    return {
        "metrics_version": METRICS_VERSION,
        "requests": n,
        "successful_requests": len(ok),
        "failed_requests": failed,
        "timeouts": statuses.count("timeout"),
        "errors": statuses.count("error"),
        "failure_rate": round(failed / n, 4) if n else None,
        "cold_requests_excluded": sum(1 for r in ok if r.get("is_cold_request")),
        "truncated_output": sum(1 for r in raw if r.get("truncated_output")),
        "input_possibly_truncated": sum(1 for r in raw if r.get("input_possibly_truncated")),
        "latency_s": _stats(latencies),
        "sum_latency_s": round(sum_latency, 3) if sum_latency is not None else None,
        "throughput_msgs_per_min": round(len(latencies) / sum_latency * 60, 3) if sum_latency else None,
        "wall_time_s": round(wall_time_s, 3) if wall_time_s else None,
        "wall_throughput_msgs_per_min": round(n / wall_time_s * 60, 3) if wall_time_s else None,
        "prompt_tokens_mean": round(prompt_tokens / len(timed), 1) if prompt_tokens is not None and timed else None,
        "output_tokens_mean": round(output_tokens / len(timed), 2) if output_tokens is not None and timed else None,
        "prefill_tokens_per_s": round(prompt_tokens / prefill_s, 2) if prompt_tokens and prefill_s else None,
        "generation_tokens_per_s": round(output_tokens / gen_s, 2) if output_tokens and gen_s else None,
        "server_total_duration_s": _stats([float(r["total_duration_s"]) for r in timed if r.get("total_duration_s") is not None]),
        "note": (
            "Latency = client wall time of one /api/generate request (HTTP + queue + prefill + generation). "
            "Prefill tokens/s uses Ollama's prompt_eval_duration, which is shortened when the shared prompt "
            "prefix is served from Ollama's prompt cache."
        ),
    }


# ──────────────────────────────────────────────
# Pairwise comparison of two runs
# ──────────────────────────────────────────────
def report_fingerprint(report: dict) -> str | None:
    return (((report or {}).get("manifest") or {}).get("dataset") or {}).get("dataset_fingerprint")


def correctness_by_record(report: dict) -> dict[str, bool]:
    """source_id -> correct (unknown counts as wrong, i.e. the 'unknown = error' mode)."""
    a = detection_arrays(report.get("raw_predictions") or [])
    truth = np.where(a["is_mal"], "malicious", "safe")
    return {sid: bool(t == p) for sid, t, p in zip(a["source_ids"], truth, a["pred"])}


def compare_runs(report_a: dict, report_b: dict) -> dict:
    """
    McNemar's test between two runs. Only runs on the same dataset_fingerprint are
    comparable (same records, same texts); otherwise {"comparable": False, ...}.
    Records are paired by source_id.
    """
    from .stats import mcnemar

    fa, fb = report_fingerprint(report_a), report_fingerprint(report_b)
    if fa is None or fb is None:
        return {"comparable": False, "reason": "not comparable: a run has no dataset fingerprint (legacy run)"}
    if fa != fb:
        return {"comparable": False, "reason": "not comparable: different dataset fingerprints"}
    ca, cb = correctness_by_record(report_a), correctness_by_record(report_b)
    ids = sorted(set(ca) & set(cb))
    if len(ids) != len(ca) or len(ids) != len(cb):
        return {"comparable": False, "reason": "not comparable: the runs do not contain the same records"}
    result = mcnemar(np.array([ca[i] for i in ids]), np.array([cb[i] for i in ids]))
    return {"comparable": True, "reason": None, **result}


# ──────────────────────────────────────────────
# Score curves (runs whose records carry a probability score)
# ──────────────────────────────────────────────
def score_curves(raw: list[dict]) -> dict | None:
    """
    ROC and precision-recall curves with AUC / average precision, from the
    per-record "score" (probability of malicious). None when the run has no
    scores (LLM runs store hard labels only) or only one class.
    """
    from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

    pairs = [(_true_binary(r), r.get("score")) for r in raw]
    pairs = [(t == "malicious", float(s)) for t, s in pairs if t is not None and isinstance(s, (int, float))]
    if len(pairs) < 2 or len({t for t, _ in pairs}) < 2:
        return None
    y = np.array([t for t, _ in pairs], dtype=int)
    s = np.array([v for _, v in pairs])
    fpr, tpr, _ = roc_curve(y, s)
    precision, recall, _ = precision_recall_curve(y, s)
    return {
        "n": len(pairs),
        "roc": {"fpr": fpr.round(5).tolist(), "tpr": tpr.round(5).tolist(), "auc": round(float(auc(fpr, tpr)), 4)},
        "pr": {"precision": precision.round(5).tolist(), "recall": recall.round(5).tolist(),
               "average_precision": round(float(average_precision_score(y, s)), 4)},
    }
