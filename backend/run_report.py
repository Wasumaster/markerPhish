"""
MarkerPhish — Run report assembly
==================================
One function turns a finished EvaluationRun into the JSON that is saved: the
metrics report (backend.metrics) plus the reproducibility manifest
(backend.manifest). The Run tab and the batch runner both use it, so every run
file has the same shape.
"""

from __future__ import annotations

import pandas as pd

from .evaluator import EvaluationRun
from .manifest import build_manifest
from .metrics import build_full_report
from .profiles import fits_profiles


def build_run_report(
    run: EvaluationRun,
    *,
    df_sample: pd.DataFrame,
    dataset_choice: str | None,
    sample_info: dict | None,
    hardware_profile: dict | None = None,
    execution_mode: str = "CPU Safe Mode",
    **collectors,
) -> dict:
    """
    Metrics report + "manifest" block for a finished run.

    collectors (model=, code=, ollama=) are passed through to build_manifest;
    tests use them to avoid calling git / Ollama.
    """
    # The hardware block is scanned at run start; it also fills the old
    # hardware_profile key (kept for compatibility) instead of the session scan.
    hardware = run.hardware or hardware_profile or {}
    report = build_full_report(run, hardware_profile=hardware, execution_mode=execution_mode)
    report["hardware"] = hardware
    report["resources"] = build_resources_block(run, report["resource_metrics"].get("p95_latency_s"))
    report["manifest"] = build_manifest(
        model_name=run.model_name,
        prompt_variant=run.prompt_variant,
        output_schema=(run.request_fields or {}).get("format"),
        dataset_choice=dataset_choice,
        df_sample=df_sample,
        sample_info=sample_info,
        options_sent=run.options_sent,
        request_fields={
            **{k: v for k, v in (run.request_fields or {}).items() if k != "format"},
            "options_changed_requests": run.options_changed,
        },
        started_at=run.started_at,
        ended_at=run.ended_at,
        power={
            "power_at_start": run.power_at_start,
            "power_at_end": run.power_at_end,
            "power_changed": run.power_changed,
            "on_battery_during_run": run.on_battery_during_run,
        },
        **collectors,
    )
    return report


def build_resources_block(run: EvaluationRun, p95_latency_s: float | None) -> dict:
    """
    The "resources" block of a run (units in the key names):
    memory (MB), CPU (% of the whole machine), VRAM (MB), GPU utilisation (%),
    the sampler's own cost, start mode, loaded models before the run, the time
    series of samples, and the fit against each constraint profile.
    """
    summary = dict(run.resources or {})
    samples = summary.pop("samples", [])
    block = {
        "peak_ram_mb": summary.get("peak_ram_mb"),
        "mean_ram_mb": summary.get("mean_ram_mb"),
        "baseline_ram_mb": run.baseline_ram_mb,
        "pre_warmup_ram_mb": run.pre_warmup_ram_mb,
        "peak_cpu_pct": summary.get("peak_cpu_pct"),
        "mean_cpu_pct": summary.get("mean_cpu_pct"),
        "model_size_mb": summary.get("model_size_mb"),
        "model_vram_mb": summary.get("model_vram_mb"),
        "model_disk_mb": round(run.model_size_bytes / (1024 * 1024), 1) if run.model_size_bytes else None,
        "peak_vram_mb": summary.get("peak_vram_mb"),
        "mean_vram_mb": summary.get("mean_vram_mb"),
        "mean_gpu_util_pct": summary.get("mean_gpu_util_pct"),
        "sample_interval_s": summary.get("sample_interval_s"),
        "n_samples": summary.get("n_samples"),
        "ps_polls": summary.get("ps_polls"),
        "sampler_cost_ms": summary.get("sampler_cost_ms"),
        "sampler_cost_max_ms": summary.get("sampler_cost_max_ms"),
        "max_tree_processes": summary.get("max_tree_processes"),
        "start_mode": run.start_mode,
        "cold_start_s": round(float(run.cold_start_s or 0.0), 3),
        "cold_unload_confirmed": run.cold_unload_confirmed,
        "ollama_loaded_models": (run.prerun or {}).get("ollama_loaded_models"),
        "other_models_loaded": (run.prerun or {}).get("other_models_loaded"),
        "other_models_unloaded": (run.prerun or {}).get("other_models_unloaded"),
        "other_models_unload_failed": (run.prerun or {}).get("other_models_unload_failed"),
    }
    block["fits_profile"] = fits_profiles(block, p95_latency_s)
    block["samples"] = samples
    return block

