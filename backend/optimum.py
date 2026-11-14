"""
MarkerPhish — Optimum analysis (pure functions)
================================================
Which configuration (model + prompt variant + inference options) is the best
choice for a resource-constrained deployment?

1. Eligibility. Only runs with protocol_version 2, not dev, with at least one
   valid prediction and on the same dataset_fingerprint enter the analysis.
   The reference fingerprint is the one shared by most eligible runs (ties:
   the most recent run). Every excluded run is listed with its reason.
2. Pareto front over three criteria: F1 (unknown = error) higher is better,
   model memory (Ollama /api/ps size) lower, P95 latency lower. A run is on the
   front when no other run is at least as good in all three and better in one.
3. Per constraint profile: feasible runs = model memory <= RAM x headroom (and
   P95 <= the profile's limit, if it has one). Winner = highest F1; ties broken
   by lower model memory, then lower P95 latency.
4. Significance. McNemar's test between the winner and every other feasible run
   on the same fingerprint (records paired by source_id, unknown = error);
   p-values Holm-adjusted over the comparisons of one profile. A run with
   adjusted p >= ALPHA is "not significantly different from the winner". This
   does not mean the runs are equivalent, only that this sample cannot tell them
   apart. CI overlap is never used as a significance test.
"""

from __future__ import annotations

from collections import Counter

from .detection import compare_runs, report_fingerprint
from .manifest import report_is_dev, report_protocol_version
from .options import differing_options, options_label
from .profiles import fits_profiles
from .stats import holm

ALPHA = 0.05


def run_point(name: str, report: dict) -> dict:
    """The three criteria and labels of one run."""
    det = report.get("detection") or {}
    ue = (det.get("overall") or {}).get("unknown_as_error") or {}
    res = report.get("resources") or {}
    timing = report.get("timing") or {}
    manifest = report.get("manifest") or {}
    return {
        "file": name,
        "model": report.get("model_name"),
        "quantization": (manifest.get("model") or {}).get("quantization"),
        "prompt_variant": (manifest.get("prompt") or {}).get("prompt_variant") or report.get("prompt_variant"),
        "f1": ue.get("f1"),
        "f1_ci": (ue.get("ci") or {}).get("f1"),
        "model_memory_mb": res.get("model_size_mb"),
        "peak_rss_mb": res.get("peak_ram_mb"),
        "p95_latency_s": (timing.get("latency_s") or {}).get("p95"),
        "unknown_rate": det.get("unknown_rate"),
        "n": det.get("n_records"),
        "is_baseline": bool(report.get("baseline")),
        "started_at": ((manifest.get("timestamps") or {}).get("started_at")) or "",
    }


def _valid_predictions(report: dict) -> bool:
    det = report.get("detection") or {}
    n = det.get("n_records") or 0
    return n > 0 and (det.get("unknown_predictions") or 0) < n


def eligibility(reports: dict[str, dict]) -> tuple[dict[str, dict], list[dict], str | None]:
    """(eligible reports, excluded [{file, reason}], reference fingerprint)."""
    excluded, candidates = [], {}
    for name, rep in reports.items():
        if report_protocol_version(rep) < 2:
            excluded.append({"file": name, "reason": "legacy run (protocol 1)"})
        elif report_is_dev(rep):
            excluded.append({"file": name, "reason": "dev run (made during the refactor)"})
        elif not _valid_predictions(rep):
            excluded.append({"file": name, "reason": "no valid predictions (all unknown)"})
        elif report_fingerprint(rep) is None:
            excluded.append({"file": name, "reason": "no dataset fingerprint"})
        else:
            candidates[name] = rep
    if not candidates:
        return {}, excluded, None
    counts = Counter(report_fingerprint(r) for r in candidates.values())
    best = max(counts.values())
    tied = {fp for fp, c in counts.items() if c == best}
    latest = max(
        (r for r in candidates.values() if report_fingerprint(r) in tied),
        key=lambda r: ((r.get("manifest") or {}).get("timestamps") or {}).get("started_at") or "",
    )
    reference = report_fingerprint(latest)
    eligible = {}
    for name, rep in candidates.items():
        if report_fingerprint(rep) == reference:
            eligible[name] = rep
        else:
            excluded.append({"file": name, "reason": "different dataset fingerprint (other sample)"})
    return eligible, excluded, reference


def pareto_front(points: list[dict]) -> set[str]:
    """Files of the non-dominated points (F1 max, memory min, P95 min); points missing a value are skipped."""
    complete = [p for p in points if None not in (p["f1"], p["model_memory_mb"], p["p95_latency_s"])]

    def dominates(a, b):
        better_or_equal = a["f1"] >= b["f1"] and a["model_memory_mb"] <= b["model_memory_mb"] and \
            a["p95_latency_s"] <= b["p95_latency_s"]
        strictly = a["f1"] > b["f1"] or a["model_memory_mb"] < b["model_memory_mb"] or \
            a["p95_latency_s"] < b["p95_latency_s"]
        return better_or_equal and strictly

    return {p["file"] for p in complete if not any(dominates(q, p) for q in complete if q is not p)}


def _rank_key(p: dict):
    return (-(p["f1"] if p["f1"] is not None else -1), p["model_memory_mb"] or float("inf"),
            p["p95_latency_s"] or float("inf"))


def analyze(reports: dict[str, dict], profiles: dict, headroom: float) -> dict:
    """Full optimum analysis of a set of runs (file name -> loaded report)."""
    eligible, excluded, reference = eligibility(reports)
    points = {name: run_point(name, rep) for name, rep in eligible.items()}
    front = pareto_front(list(points.values()))
    differing = differing_options(list(eligible.values()))
    for name, p in points.items():
        p["pareto"] = name in front
        p["options"] = options_label(eligible[name], differing) if differing else ""

    per_profile = {}
    for profile_name, profile in profiles.items():
        feasible, not_feasible = [], []
        for name, p in points.items():
            fit = fits_profiles({"model_size_mb": p["model_memory_mb"], "peak_ram_mb": p["peak_rss_mb"],
                                 "model_vram_mb": (eligible[name].get("resources") or {}).get("model_vram_mb")},
                                p["p95_latency_s"], {profile_name: profile}, headroom)[profile_name]
            if fit["fits_profile"] and p["f1"] is not None:
                feasible.append(p)
            else:
                reason = ("model memory unknown" if fit["fits_memory"] is None else
                          "exceeds the memory budget" if not fit["fits_memory"] else
                          "exceeds the latency limit" if fit["fits_latency"] is False else "no F1")
                not_feasible.append({"file": name, "reason": reason})
        feasible.sort(key=_rank_key)
        winner = feasible[0] if feasible else None
        comparisons = []
        if winner:
            raw = [(p, compare_runs(eligible[winner["file"]], eligible[p["file"]])) for p in feasible[1:]]
            testable = [(p, r) for p, r in raw if r.get("comparable")]
            adjusted = holm([r["p_value"] for _, r in testable])
            for (p, r), p_adj in zip(testable, adjusted):
                comparisons.append({
                    "file": p["file"], "p_value": r["p_value"], "p_holm": round(p_adj, 6),
                    "method": r["method"], "winner_only_correct": r["only_a_correct"],
                    "other_only_correct": r["only_b_correct"],
                    "not_significantly_different": p_adj >= ALPHA,
                })
            for p, r in raw:
                if not r.get("comparable"):
                    comparisons.append({"file": p["file"], "p_value": None, "p_holm": None,
                                        "not_significantly_different": None, "reason": r["reason"]})
        budget = profile["ram_gb"] * 1024 * headroom
        per_profile[profile_name] = {
            "budget_mb": round(budget, 1),
            "max_p95_latency_s": profile.get("max_p95_latency_s"),
            "feasible": [p["file"] for p in feasible],
            "not_feasible": not_feasible,
            "winner": winner["file"] if winner else None,
            "comparisons": comparisons,
            "alternatives_not_significantly_different": [c["file"] for c in comparisons
                                                        if c.get("not_significantly_different")],
        }
    return {
        "alpha": ALPHA,
        "correction": "holm",
        "reference_fingerprint": reference,
        "headroom_factor": headroom,
        "points": points,
        "pareto": sorted(front),
        "differing_options": differing,
        "per_profile": per_profile,
        "excluded": excluded,
    }


def pairwise_pvalues(reports: dict[str, dict]) -> dict:
    """McNemar p-values for every pair of the given runs (None when not comparable)."""
    names = list(reports)
    table = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            r = compare_runs(reports[a], reports[b])
            table[(a, b)] = r.get("p_value") if r.get("comparable") else None
    return table
