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
from backend.metrics import build_full_report
from backend.utils import save_results


def render_evaluation_runner(
    selected_model: str,
    df_eval: pd.DataFrame,
    num_gpu: int | None,
    mode_display_name: str,
) -> None:
    """
    Render the live evaluation runner interface and handle execution lifecycle.

    Args:
        selected_model: Name of the Ollama model to evaluate.
        df_eval: DataFrame containing the dataset records to evaluate.
        num_gpu: GPU offloading setting (0 for CPU Safe Mode, None for auto/GPU).
        mode_display_name: Human-readable execution mode string.
    """
    st.markdown('<div class="section-title">Automated Model Evaluation</div>', unsafe_allow_html=True)

    # ──────────────────────────────────────────────
    # 1. Configuration Manifest Card
    # ──────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="card-panel">
            <div class="card-panel-header">Evaluation Configuration Manifest</div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; font-size: 0.85rem;">
                <div>
                    <span style="color: #94a3b8;">Target Model:</span><br>
                    <strong style="color: #f8fafc; font-family: monospace;">{selected_model}</strong>
                </div>
                <div>
                    <span style="color: #94a3b8;">Execution Environment:</span><br>
                    <strong style="color: #38bdf8;">{mode_display_name}</strong>
                </div>
                <div>
                    <span style="color: #94a3b8;">Dataset Volume:</span><br>
                    <strong style="color: #f8fafc;">{len(df_eval):,} messages</strong>
                </div>
                <div>
                    <span style="color: #94a3b8;">Evaluation Standard:</span><br>
                    <strong style="color: #10b981;">Binary Threat Classification</strong>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_btn, _ = st.columns([1, 2])
    with col_btn:
        start_btn = st.button(
            "Execute Evaluation Run",
            type="primary",
            width="stretch",
            disabled=st.session_state.get("is_running", False),
        )

    if start_btn:
        st.session_state["is_running"] = True

        progress_bar = st.progress(0, text="Initializing evaluation session...")
        telemetry_area = st.empty()
        live_log_area = st.empty()

        correct_binary_count = 0
        total_latency = 0.0

        def progress_callback(current: int, total: int, result) -> None:
            nonlocal correct_binary_count, total_latency
            pct = current / total
            is_correct_bin = result.is_binary_correct
            status_text = "CORRECT" if is_correct_bin else "MISMATCH"

            progress_bar.progress(
                pct,
                text=(
                    f"Processing record {current}/{total} | "
                    f"Prediction: {result.predicted_label} [{status_text}] | "
                    f"Latency: {result.latency_seconds:.2f}s"
                ),
            )

            if is_correct_bin:
                correct_binary_count += 1
            total_latency += result.latency_seconds

            # Refresh telemetry metrics every 2 records or on completion
            if current % 2 == 0 or current == total:
                with telemetry_area.container():
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Processed Records", f"{current}/{total}")
                    with m2:
                        running_acc = (correct_binary_count / current) if current > 0 else 0.0
                        st.metric("Running Binary Accuracy", f"{running_acc:.1%}")
                    with m3:
                        avg_lat = (total_latency / current) if current > 0 else 0.0
                        st.metric("Mean Latency", f"{avg_lat:.2f}s")
                    with m4:
                        eta_seconds = avg_lat * (total - current) if avg_lat > 0 else 0.0
                        mins, secs = divmod(int(eta_seconds), 60)
                        st.metric("Estimated Remaining (ETA)", f"{mins}m {secs}s")

            # Refresh live activity log card
            with live_log_area.container():
                outcome_color = "#10b981" if is_correct_bin else "#f43f5e"
                st.markdown(
                    f"""
                    <div class="card-panel" style="margin-top: 10px; padding: 12px 16px;">
                        <div class="card-panel-header">Recent Inference Telemetry (Record #{result.record_id})</div>
                        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.85rem;">
                            <span>True Class: <strong>{result.true_label}</strong> ({result.true_binary})</span>
                            <span>Model Prediction: <strong>{result.predicted_label}</strong> ({result.predicted_binary})</span>
                            <span style="color: {outcome_color}; font-weight: 600;">{status_text}</span>
                            <span>Latency: <strong>{result.latency_seconds:.3f}s</strong></span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        eval_run = run_evaluation(
            model_name=selected_model,
            df=df_eval,
            num_gpu=num_gpu,
            progress_callback=progress_callback,
        )

        st.session_state["evaluation_run"] = eval_run
        hw_dict = (
            st.session_state["hardware_profile"].to_dict()
            if hasattr(st.session_state.get("hardware_profile"), "to_dict")
            else st.session_state.get("hardware_profile", {})
        )
        report = build_full_report(eval_run, hardware_profile=hw_dict, execution_mode=mode_display_name)
        st.session_state["report"] = report

        filepath = save_results(report, selected_model)
        new_filename = os.path.basename(filepath)
        st.session_state["selected_result_file"] = new_filename
        st.session_state["loaded_filename"] = new_filename

        progress_bar.progress(1.0, text="Evaluation completed successfully.")
        st.session_state["is_running"] = False
        st.success(f"Evaluation session completed. Persistent report archived to: {filepath}")
