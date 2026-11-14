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
    report = build_full_report(run, hardware_profile=hardware_profile, execution_mode=execution_mode)
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
        power_at_start=run.power_at_start,
        **collectors,
    )
    return report
