"""
MarkerPhish — Live Evaluation Runner Component
===============================================
Executes automated inference runs against the local Ollama instance with
real-time telemetry: progress tracking, dynamic accuracy, mean latency, and ETA.
"""

from __future__ import annotations

import os
import pandas as pd
import streamlit as st

from backend.evaluator import run_evaluation
from backend.run_report import build_run_report
from backend.utils import logger, save_results
from config import ACTIVE_PROMPT_VARIANT, PROMPT_VARIANTS
from ui.components import (
    cell,
    data_table,
    kv_table,
    page_title,
    section_label,
)
from ui.theme import MALICIOUS, SAFE


def render_evaluation_runner(
    selected_model: str,
    df_eval: pd.DataFrame,
    num_gpu: int | None,
    mode_display_name: str,
    prompt_variant: str | None = None,
) -> None:
    """
    Render the live evaluation runner interface and handle execution lifecycle.

    Args:
        selected_model: Name of the Ollama model to evaluate.
        df_eval: DataFrame containing the dataset records to evaluate.
        num_gpu: GPU offloading setting (0 for CPU Safe Mode, None for auto/GPU).
        mode_display_name: Human-readable execution mode string.
        prompt_variant: Prompt variant key from config.PROMPT_VARIANTS (active variant if None).
    """
    prompt_variant = prompt_variant or ACTIVE_PROMPT_VARIANT

    page_title(
        "Run",
        "Classify every message in the current sample with the selected model and prompt. "
        "Model loading is measured separately as cold start and is excluded from latency.",
    )

    # ──────────────────────────────────────────────
    # 1. Configuration
    # ──────────────────────────────────────────────
    with st.container():
        section_label("configuration")
        kv_table(
            [
                ("model", selected_model),
                ("prompt", prompt_variant),
                ("prompt description", PROMPT_VARIANTS.get(prompt_variant, {}).get("description", "—")),
                ("environment", mode_display_name),
                ("messages", f"{len(df_eval):,}"),
                ("task", "binary threat classification (malicious vs. valid)"),
            ],
            headers=("setting", "value"),
        )

    col_btn, _ = st.columns([1, 2])
    with col_btn:
        start_btn = st.button(
            "Run evaluation",
            type="primary",
            width="stretch",
            disabled=st.session_state.get("is_running", False),
        )

    if start_btn:
        st.session_state["is_running"] = True
        # The flag is reset in `finally`, so an exception or a rerun that interrupts
        # the script (Streamlit raises inside the running script) never leaves the
        # button disabled for the rest of the session.
        try:
            _execute_run(selected_model, df_eval, num_gpu, mode_display_name, prompt_variant)
        except Exception as exc:  # noqa: BLE001 - shown to the user, then logged
            logger.exception("Evaluation run failed")
            st.error(f"Evaluation failed: {exc}")
        finally:
            st.session_state["is_running"] = False


def _execute_run(
    selected_model: str,
    df_eval: pd.DataFrame,
    num_gpu: int | None,
    mode_display_name: str,
    prompt_variant: str,
) -> None:
    """Run one evaluation with live telemetry, then build, store and save the report."""
    progress_bar = st.progress(0, text="Loading model into memory (cold start, excluded from latency)...")
    telemetry_area = st.empty()
    live_log_area = st.empty()

    correct_binary_count = 0
    total_latency = 0.0

    def progress_callback(current: int, total: int, result) -> None:
        nonlocal correct_binary_count, total_latency
        pct = current / total
        is_correct_bin = result.is_binary_correct
        status_text = "correct" if is_correct_bin else "mismatch"

        progress_bar.progress(
            pct,
            text=(
                f"record {current}/{total} · "
                f"prediction {result.predicted_label} ({status_text}) · "
                f"{result.latency_seconds:.2f} s"
            ),
        )

        if is_correct_bin:
            correct_binary_count += 1
        total_latency += result.latency_seconds

        # Refresh telemetry every 2 records or on completion
        if current % 2 == 0 or current == total:
            running_acc = (correct_binary_count / current) if current > 0 else 0.0
            avg_lat = (total_latency / current) if current > 0 else 0.0
            eta_seconds = avg_lat * (total - current) if avg_lat > 0 else 0.0
            mins, secs = divmod(int(eta_seconds), 60)
            with telemetry_area.container():
                section_label("telemetry")
                data_table(
                    ["processed", "running binary accuracy", "mean latency", "estimated remaining"],
                    [[f"{current}/{total}", f"{running_acc:.2%}", f"{avg_lat:.2f} s", f"{mins}m {secs}s"]],
                    numeric={0, 1, 2, 3},
                    max_width="760px",
                )

        # Refresh the most recent inference row
        with live_log_area.container():
            section_label(f"last inference · record {result.record_id}")
            data_table(
                ["record", "true class", "prediction", "outcome", "latency"],
                [
                    [
                        str(result.record_id),
                        cell(
                            f"{result.true_label} ({result.true_binary})",
                            color=MALICIOUS if result.true_binary == "malicious" else SAFE,
                        ),
                        cell(
                            f"{result.predicted_label} ({result.predicted_binary})",
                            color=MALICIOUS if result.predicted_binary == "malicious" else SAFE,
                        ),
                        cell(status_text, bold=not is_correct_bin),
                        f"{result.latency_seconds:.3f} s",
                    ]
                ],
                numeric={4},
                max_width="760px",
            )

    eval_run = run_evaluation(
        model_name=selected_model,
        df=df_eval,
        num_gpu=num_gpu,
        progress_callback=progress_callback,
        prompt_variant=prompt_variant,
    )

    st.session_state["evaluation_run"] = eval_run
    hw_dict = (
        st.session_state["hardware_profile"].to_dict()
        if hasattr(st.session_state.get("hardware_profile"), "to_dict")
        else st.session_state.get("hardware_profile", {})
    )
    report = build_run_report(
        eval_run,
        df_sample=df_eval,
        dataset_choice=st.session_state.get("dataset_choice"),
        sample_info=st.session_state.get("sample_info"),
        hardware_profile=hw_dict,
        execution_mode=mode_display_name,
    )
    st.session_state["report"] = report

    filepath = save_results(report, selected_model, prompt_variant=prompt_variant)
    new_filename = os.path.basename(filepath)
    st.session_state["selected_result_file"] = new_filename
    st.session_state["loaded_filename"] = new_filename

    progress_bar.progress(1.0, text="Evaluation completed.")
    st.success(f"Evaluation session completed. Report saved to: {filepath}")
