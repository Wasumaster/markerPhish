"""
MarkerPhish — Live Evaluation Runner Component
===============================================
Executes automated inference runs against the local Ollama instance with
real-time telemetry: progress tracking, dynamic accuracy, mean latency, and ETA.
"""

from __future__ import annotations

import os

import pandas as pd
import psutil
import streamlit as st

from backend.baseline import BASELINE_NAME, BASELINE_VARIANT, build_baseline_report
from backend.baseline import CAVEAT as BASELINE_CAVEAT
from backend.evaluator import loaded_models, other_loaded_models, run_evaluation
from backend.experiment import estimate_run_seconds, format_duration, measured_seconds_per_record
from backend.sampling import sample_size_purpose
from backend.manifest import model_details
from backend.options import CONFIGURABLE, OptionsError, resolve_options
from backend.run_report import build_run_report
from backend.utils import list_saved_result_files, logger, save_results
from config import (
    ACTIVE_PROMPT_VARIANT,
    DEFAULT_START_MODE,
    DEFAULT_UNLOAD_OTHER_MODELS,
    PROMPT_VARIANTS,
    START_MODES,
)
from ui.components import (
    cell,
    note,
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
        "Model loading is measured separately as cold start and is excluded from latency. "
        "Memory and CPU of the Ollama server are sampled every 0.5 s during the run.",
    )

    # ──────────────────────────────────────────────
    # 1. Configuration
    # ──────────────────────────────────────────────
    sample_info = st.session_state.get("sample_info") or {}
    cells = " · ".join(f"{c['channel']} {c['label']} {c['count']}" for c in sample_info.get("cells") or [])
    purpose = sample_size_purpose(len(df_eval))
    purpose_text = {"smoke": "smoke test, not for results", "pilot": "testing and pilot only",
                    "results": "for results"}[purpose]
    seconds, quality = estimate_run_seconds(selected_model, prompt_variant, len(df_eval), _measured_latencies())
    estimate = f"{format_duration(seconds)} ({'measured on earlier runs' if quality == 'measured' else 'rough: no earlier run of this model and prompt'})"
    with st.container():
        section_label("configuration")
        kv_table(
            [
                ("model", selected_model),
                ("prompt", prompt_variant),
                ("prompt description", PROMPT_VARIANTS.get(prompt_variant, {}).get("description", "—")),
                ("environment", mode_display_name),
                ("messages", f"{len(df_eval):,} · {purpose_text}"),
                ("per cell", cells or "—"),
                ("estimated time", estimate),
                ("task", "binary threat classification (malicious vs. valid)"),
            ],
            headers=("setting", "value"),
        )
        if sample_info.get("fallback"):
            st.warning(sample_info["fallback_reason"])

    start_col, _ = st.columns([1, 2])
    with start_col:
        start_mode = st.radio(
            "start",
            options=list(START_MODES),
            index=list(START_MODES).index(DEFAULT_START_MODE),
            horizontal=True,
            help=(
                "warm: one excluded warm-up request loads the model before the timed run. "
                "cold: the model is unloaded first; the first request pays the load, its latency is "
                "excluded from the latency statistics and reported as cold start."
            ),
        )

    others = other_loaded_models(loaded_models(), selected_model)
    unload_others = DEFAULT_UNLOAD_OTHER_MODELS
    if others:
        st.warning(
            "Other models are loaded in Ollama: " + ", ".join(others) + ". They change the measured "
            "memory and latency of this run."
        )
        unload_others = st.checkbox(
            "unload other models before the run (recommended)", value=DEFAULT_UNLOAD_OTHER_MODELS
        )

    user_options, options_error = _render_advanced_options(selected_model, df_eval, prompt_variant, num_gpu)

    col_btn, _ = st.columns([1, 2])
    with col_btn:
        start_btn = st.button(
            "Run evaluation",
            type="primary",
            width="stretch",
            disabled=st.session_state.get("is_running", False) or options_error,
        )

    _render_baseline_button(df_eval)

    if start_btn:
        st.session_state["is_running"] = True
        # The flag is reset in `finally`, so an exception or a rerun that interrupts
        # the script (Streamlit raises inside the running script) never leaves the
        # button disabled for the rest of the session.
        try:
            _execute_run(
                selected_model, df_eval, num_gpu, mode_display_name, prompt_variant, start_mode, unload_others,
                user_options,
            )
        except Exception as exc:  # noqa: BLE001 - shown to the user, then logged
            logger.exception("Evaluation run failed")
            st.error(f"Evaluation failed: {exc}")
        finally:
            st.session_state["is_running"] = False


def _render_baseline_button(df_eval: pd.DataFrame) -> None:
    """Optional non-LLM reference: TF-IDF + logistic regression on the same sample (seconds, no Ollama)."""
    with st.expander("non-LLM baseline", expanded=False):
        note(BASELINE_CAVEAT)
        if not st.button("Run TF-IDF baseline on this sample", disabled=st.session_state.get("is_running", False)):
            return
        df_full = st.session_state.get("dataset_full_df")
        if df_full is None:
            st.error("Dataset not loaded. Open the Dataset tab first.")
            return
        with st.spinner("Training on the records outside the sample, then predicting the sample..."):
            report = build_baseline_report(
                df_full, df_eval, st.session_state.get("sample_info"), st.session_state.get("dataset_choice")
            )
            filepath = save_results(report, BASELINE_NAME, prompt_variant=BASELINE_VARIANT)
        name = os.path.basename(filepath)
        st.session_state["report"] = report
        st.session_state["selected_result_file"] = name
        st.session_state["loaded_filename"] = name
        f1 = report["detection"]["overall"]["unknown_as_error"]["f1"]
        st.success(f"Baseline saved to {filepath} (F1 {f1}).")


def _measured_latencies() -> dict:
    """(model, prompt variant) -> seconds per record from earlier non-inflated protocol-2 runs."""
    from frontend.multi_comparison import load_reports

    return measured_seconds_per_record(list(load_reports(list_saved_result_files()).values()))


@st.cache_data(ttl=300, show_spinner=False)
def _model_context_length(model_name: str) -> int | None:
    """The model's context length from /api/show (does not load the model)."""
    return model_details(model_name).get("context_length")


_OPTION_HELP = {
    "num_ctx": "Context window in tokens (prompt + output). 0 = default 4096, never above the model's context "
               "length; raised automatically only if a calibrated tokenizer needs more room.",
    "num_thread": "CPU threads used by Ollama. 0 = Ollama's default.",
    "num_predict": "Output token cap. 0 = default for every model (or the model's override). An answer cut at "
                   "the cap is marked truncated and counted as unknown.",
    "num_batch": "Prompt-processing batch size. 0 = Ollama's default.",
}


def _render_advanced_options(model_name: str, df_eval: pd.DataFrame, prompt_variant: str, num_gpu) -> tuple:
    """The 'advanced' expander. Returns (user options, True if they are invalid)."""
    cores = psutil.cpu_count(logical=True) or 1
    with st.expander("advanced", expanded=False):
        cols = st.columns(len(CONFIGURABLE))
        values = {}
        for col, name in zip(cols, CONFIGURABLE):
            with col:
                values[name] = int(
                    st.number_input(
                        name,
                        min_value=0,
                        max_value=cores if name == "num_thread" else None,
                        value=0,
                        step=512 if name == "num_ctx" else 1,
                        help=_OPTION_HELP[name],
                        key=f"opt_{name}",
                    )
                )
        user = {k: v for k, v in values.items() if v}
        try:
            resolved = resolve_options(
                model_name=model_name,
                prompt_variant=prompt_variant,
                df_sample=df_eval,
                model_context_length=_model_context_length(model_name),
                logical_cores=cores,
                num_gpu=num_gpu,
                user=user,
            )
        except OptionsError as exc:
            st.error(str(exc))
            return user, True
        sent = ", ".join(f"{k}={v}" for k, v in resolved.options.items())
        note(f"Options sent to Ollama: {sent}. Temperature and seed are fixed.", mono=False)
        for line in resolved.notes:
            note(line)
    return user, False


def _execute_run(
    selected_model: str,
    df_eval: pd.DataFrame,
    num_gpu: int | None,
    mode_display_name: str,
    prompt_variant: str,
    start_mode: str = DEFAULT_START_MODE,
    unload_others: bool = DEFAULT_UNLOAD_OTHER_MODELS,
    user_options: dict | None = None,
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
        start_mode=start_mode,
        unload_other_models=unload_others,
        user_options=user_options,
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
    if eval_run.power_changed or eval_run.on_battery_during_run:
        st.warning("The laptop was not on AC power for the whole run; CPU latency may be affected.")
    st.success(f"Evaluation session completed. Report saved to: {filepath}")
