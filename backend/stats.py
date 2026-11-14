"""
MarkerPhish — Detection statistics (pure functions)
====================================================
Binary threat detection with malicious = positive class.

Per record: true class y in {malicious, safe}; prediction p in {malicious,
safe, unknown}; channel in {email, sms}; optional group id.

Two ways to treat "unknown" (no parsable answer):
- excluding unknown: the record is dropped from the confusion matrix
  (the platform's original metrics);
- unknown = error (primary): the record counts as a wrong prediction, i.e. a
  missed threat (FN) when y = malicious and a false alarm (FP) when y = safe.

    precision = TP / (TP + FP)     recall (TPR) = TP / (TP + FN)
    F1 = 2PR / (P + R)             FPR = FP / (FP + TN)     FNR = FN / (TP + FN)

Bootstrap CIs: B resamples with a fixed seed. Resampling is stratified by
(class, channel) cell, so every resample has the original cell sizes; when
group ids exist, whole groups are resampled within their cell (cluster
bootstrap), so rephrasings of one email never count as independent. The CI is
the 2.5th-97.5th percentile of the metric over resamples where it is defined.

McNemar: paired comparison of two runs on the same records (same
dataset_fingerprint). b = records only run A got right, c = only run B got
right; exact two-sided binomial test when b + c < 25, otherwise the
chi-square statistic with continuity correction (|b - c| - 1)^2 / (b + c).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats as sps

BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 20261114
CI_LEVEL = 0.95
MCNEMAR_EXACT_BELOW = 25
UNKNOWN_MODES = ("excluding_unknown", "unknown_as_error")
CI_METRICS = ("precision", "recall", "f1", "fpr")


def _div(a, b):
    """a / b with NaN where b == 0 (works for scalars and numpy arrays)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)
    return out if out.ndim else float(out)


def counts(is_mal: np.ndarray, pred: np.ndarray, mode: str, weights: np.ndarray | None = None) -> dict:
    """
    Confusion counts. is_mal: bool array (true class malicious); pred: array of
    'malicious' / 'safe' / 'unknown'. weights: optional (B, n) or (n,) resample weights.
    """
    if mode not in UNKNOWN_MODES:
        raise ValueError(f"unknown mode {mode}")
    pm = pred == "malicious"
    ps = pred == "safe"
    unk = ~(pm | ps)
    if mode == "unknown_as_error":
        tp_m, fn_m = is_mal & pm, is_mal & ~pm
        fp_m, tn_m = ~is_mal & (pm | unk), ~is_mal & ps
    else:
        tp_m, fn_m = is_mal & pm, is_mal & ps
        fp_m, tn_m = ~is_mal & pm, ~is_mal & ps
    w = np.ones(len(is_mal)) if weights is None else weights
    return {
        "tp": w @ tp_m if w.ndim > 1 else float(np.sum(w * tp_m)),
        "fn": w @ fn_m if w.ndim > 1 else float(np.sum(w * fn_m)),
        "fp": w @ fp_m if w.ndim > 1 else float(np.sum(w * fp_m)),
        "tn": w @ tn_m if w.ndim > 1 else float(np.sum(w * tn_m)),
        "unknown": w @ unk if w.ndim > 1 else float(np.sum(w * unk)),
    }


def rates(c: dict) -> dict:
    """Metrics from counts (NaN where undefined)."""
    tp, fn, fp, tn = c["tp"], c["fn"], c["fp"], c["tn"]
    # F1 = 2PR / (P + R) = 2TP / (2TP + FP + FN): the second form is also defined
    # (as 0) when P or R is undefined but errors exist, and undefined only when
    # there are no positives and nothing was flagged.
    return {
        "precision": _div(tp, tp + fp),
        "recall": _div(tp, tp + fn),
        "f1": _div(2 * tp, 2 * tp + fp + fn),
        "fpr": _div(fp, fp + tn),
        "fnr": _div(fn, tp + fn),
        "specificity": _div(tn, fp + tn),
        "accuracy": _div(tp + tn, tp + tn + fp + fn),
    }


def _clean(value):
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else round(value, 4)


def point_metrics(is_mal: np.ndarray, pred: np.ndarray, mode: str) -> dict:
    """Counts, rates and support of one unknown mode (JSON-ready; undefined rates are None)."""
    c = counts(is_mal, pred, mode)
    r = rates(c)
    out = {k: int(round(v)) for k, v in c.items()}
    out.update({k: _clean(v) for k, v in r.items()})
    out["support"] = int(len(is_mal))
    out["support_malicious"] = int(np.sum(is_mal))
    out["support_safe"] = int(np.sum(~is_mal))
    out["evaluated"] = out["tp"] + out["fn"] + out["fp"] + out["tn"]
    return out


# ──────────────────────────────────────────────
# Bootstrap
# ──────────────────────────────────────────────
def bootstrap_weights(
    strata: np.ndarray,
    groups: np.ndarray | None,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    """
    (B, n) matrix: how often each record appears in each resample. Within each
    stratum, the stratum's groups are drawn with replacement (records of a drawn
    group all come along); without groups every record is its own group.
    """
    n = len(strata)
    rng = np.random.default_rng(seed)
    if groups is None:
        groups = np.arange(n).astype(str)
    w = np.zeros((n_resamples, n))
    for stratum in sorted(set(strata.tolist())):
        idx = np.flatnonzero(strata == stratum)
        g_codes, g_index = np.unique(groups[idx], return_inverse=True)
        k = len(g_codes)
        draws = rng.integers(0, k, size=(n_resamples, k))
        # group multiplicities per resample -> record weights
        mult = np.zeros((n_resamples, k))
        np.add.at(mult, (np.repeat(np.arange(n_resamples), k), draws.ravel()), 1)
        w[:, idx] = mult[:, g_index]
    return w


def bootstrap_ci(
    is_mal: np.ndarray,
    pred: np.ndarray,
    weights: np.ndarray,
    mode: str,
    mask: np.ndarray | None = None,
    level: float = CI_LEVEL,
) -> dict:
    """Percentile CIs of precision, recall, F1 and FPR (optionally on a subset mask)."""
    if mask is not None:
        is_mal, pred, weights = is_mal[mask], pred[mask], weights[:, mask]
    r = rates(counts(is_mal, pred, mode, weights))
    lo_q, hi_q = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    out = {}
    for metric in CI_METRICS:
        values = np.asarray(r[metric], dtype=float)
        valid = values[~np.isnan(values)]
        if len(valid) < 0.5 * len(values) or len(valid) == 0:
            out[metric] = {"low": None, "high": None, "valid_resamples": int(len(valid))}
        else:
            out[metric] = {
                "low": round(float(np.percentile(valid, lo_q)), 4),
                "high": round(float(np.percentile(valid, hi_q)), 4),
                "valid_resamples": int(len(valid)),
            }
    return out


# ──────────────────────────────────────────────
# McNemar
# ──────────────────────────────────────────────
def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """McNemar test on paired correctness vectors (True = correct)."""
    a = np.asarray(correct_a, dtype=bool)
    b_ = np.asarray(correct_b, dtype=bool)
    if a.shape != b_.shape:
        raise ValueError("paired vectors differ in length")
    b = int(np.sum(a & ~b_))  # only A correct
    c = int(np.sum(~a & b_))  # only B correct
    n = b + c
    if n == 0:
        p, method, statistic = 1.0, "exact", None
    elif n < MCNEMAR_EXACT_BELOW:
        p = float(min(1.0, 2 * sps.binom.cdf(min(b, c), n, 0.5)))
        method, statistic = "exact", None
    else:
        statistic = (abs(b - c) - 1) ** 2 / n
        p = float(sps.chi2.sf(statistic, 1))
        method = "chi2_cc"
    return {
        "only_a_correct": b,
        "only_b_correct": c,
        "both_correct": int(np.sum(a & b_)),
        "both_wrong": int(np.sum(~a & ~b_)),
        "n": int(len(a)),
        "statistic": None if statistic is None else round(float(statistic), 4),
        "p_value": round(p, 6),
        "method": method,
    }


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted
