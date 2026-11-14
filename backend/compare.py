"""
MarkerPhish — Comparison table data (pure functions)
=====================================================
One row per run for the Compare tab, plus the rules that decide what is shown
and what may be marked best:

- legacy runs (protocol 1) and dev runs are hidden unless the user asks for
  them; when shown they are marked and can never be "best" in a column;
- a run without a valid prediction (all answers unknown, or none at all) is
  never "best" either;
- runs on different dataset fingerprints trigger a warning (their numbers come
  from different samples);
- options that differ between the compared runs are shown per row, because
  they make the runs different configurations.
"""

from __future__ import annotations

from .detection import report_fingerprint
from .manifest import report_is_dev, report_latency_inflated, report_protocol_version
from .options import differing_options, options_label


def hidden_reason(report: dict) -> str | None:
    """Why a run is hidden by default ('legacy' / 'dev'), or None."""
    if report_protocol_version(report) < 2:
        return "legacy"
    if report_is_dev(report):
        return "dev"
    return None


def has_valid_predictions(report: dict) -> bool:
    det = report.get("detection") or {}
    n = det.get("n_records") or 0
    return n > 0 and (det.get("unknown_predictions") or 0) < n


def visible_runs(reports: dict[str, dict], show_hidden: bool) -> dict[str, dict]:
    return {n: r for n, r in reports.items() if show_hidden or hidden_reason(r) is None}


def comparison_row(name: str, report: dict, option_keys: list[str], profile_names: list[str]) -> dict:
    det = report.get("detection") or {}
    ue = (det.get("overall") or {}).get("unknown_as_error") or {}
    timing = report.get("timing") or {}
    res = report.get("resources") or {}
    manifest = report.get("manifest") or {}
    fits = res.get("fits_profile") or {}
    hidden = hidden_reason(report)
    return {
        "file": name,
        "model": report.get("model_name"),
        "quantization": (manifest.get("model") or {}).get("quantization"),
        "prompt": (manifest.get("prompt") or {}).get("prompt_variant") or report.get("prompt_variant") or "legacy",
        "options": options_label(report, option_keys) if option_keys else "",
        "start": res.get("start_mode"),
        "n": det.get("n_records"),
        "f1": ue.get("f1"),
        "f1_ci": (ue.get("ci") or {}).get("f1"),
        "recall": ue.get("recall"),
        "precision": ue.get("precision"),
        "fpr": ue.get("fpr"),
        "unknown_rate": det.get("unknown_rate"),
        "p95_latency_s": (timing.get("latency_s") or {}).get("p95"),
        "throughput": timing.get("throughput_msgs_per_min"),
        "generation_tps": timing.get("generation_tokens_per_s"),
        "prefill_tps": timing.get("prefill_tokens_per_s"),
        "model_memory_mb": res.get("model_size_mb"),
        "peak_rss_mb": res.get("peak_ram_mb"),
        "mean_cpu_pct": res.get("mean_cpu_pct"),
        "fits": {p: (fits.get(p) or {}).get("fits_profile") for p in profile_names},
        "peak_vram_mb": res.get("peak_vram_mb"),
        "model_vram_mb": res.get("model_vram_mb"),
        "hidden": hidden,
        "baseline": bool(report.get("baseline")),
        "latency_inflated": report_latency_inflated(report),
        # never "best": legacy / dev runs and runs without a valid prediction
        "can_be_best": hidden is None and has_valid_predictions(report),
        "started_at": (manifest.get("timestamps") or {}).get("started_at"),
    }


def build_comparison(reports: dict[str, dict], profile_names: list[str]) -> dict:
    """Rows of the comparison table and the facts that shape it."""
    keys = differing_options(list(reports.values()))
    rows = [comparison_row(n, r, keys, profile_names) for n, r in reports.items()]
    fingerprints = {report_fingerprint(r) for r in reports.values() if report_fingerprint(r)}
    return {
        "rows": rows,
        "differing_options": keys,
        "fingerprints": sorted(fingerprints),
        "mixed_samples": len(fingerprints) > 1 or (bool(fingerprints) and any(report_fingerprint(r) is None
                                                                            for r in reports.values())),
        "has_gpu": any((r["peak_vram_mb"] or 0) > 0 or (r["model_vram_mb"] or 0) > 0 for r in rows),
    }


def best_values(rows: list[dict], key: str, prefer: str) -> list[bool]:
    """Best flag per row for one column; rows that may not be best never win, missing values never win."""
    values = [r[key] if r["can_be_best"] else None for r in rows]
    present = [v for v in values if v is not None]
    if not present:
        return [False] * len(rows)
    target = max(present) if prefer == "max" else min(present)
    return [v is not None and abs(v - target) <= 1e-12 for v in values]
