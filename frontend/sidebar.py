"""
MarkerPhish — Sidebar Component
=================================
Renders the application sidebar with model selection, dataset configuration,
and execution environment controls.
"""

from __future__ import annotations

import streamlit as st

from backend.data_loader import DATASET_CHOICES
from backend.evaluator import check_ollama_connection, list_available_models
from backend.hardware import resolve_num_gpu
from config import SAMPLE_SIZES


def render_sidebar() -> tuple:
    """
    Render the sidebar with all configuration controls.

    Returns:
        tuple: (selected_model, dataset_choice, sample_size, num_gpu, mode_display_name)
               or (None, None, None, None, None) if configuration is incomplete.
    """
    hw = st.session_state["hardware_profile"]

    with st.sidebar:
        st.markdown(
            """
            <div style="padding-bottom: 8px;">
                <span style="font-size: 1.25rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.01em;">MarkerPhish</span><br>
                <span style="font-size: 0.78rem; color: #94a3b8;">Evaluation & Analytics Platform</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        # ── 1. MODEL SELECTION (Very Top) ──
        st.markdown("### Model Selection")

        # Inference Server Connection Status
        connected = check_ollama_connection()
        if connected:
            st.markdown(
                '<div class="badge badge-online">Ollama Service Online</div>',
                unsafe_allow_html=True,
            )
            st.caption("Endpoint: `http://localhost:11434`")
        else:
            st.markdown(
                '<div class="badge badge-offline">Ollama Service Offline</div>',
                unsafe_allow_html=True,
            )
            st.caption("Start Ollama service: `ollama serve`")
            st.error("Server connection required to evaluate models.")
            return None, None, None, None, None

        # Installed Models Dropdown
        models = list_available_models()
        if not models:
            st.warning("No locally installed models detected. Execute `ollama pull <model>` in terminal.")
            return None, None, None, None, None

        model_names = [m["name"] for m in models]
        model_sizes = {m["name"]: m["size"] for m in models}

        selected_model = st.selectbox(
            "Installed LLM",
            options=model_names,
            index=0,
            help="Select a model installed in local Ollama repository",
        )

        if selected_model:
            size_mb = model_sizes.get(selected_model, 0) / (1024 * 1024)
            st.caption(f"Storage footprint: **{size_mb:.1f} MB** ({size_mb / 1024:.2f} GB)")

        st.divider()

        # ── 2. DATASET & SCOPE (Second) ──
        st.markdown("### Dataset & Scope")
        dataset_label = st.selectbox(
            "Evaluation Dataset",
            options=list(DATASET_CHOICES.keys()),
            index=0,
            help="Select dataset combination for evaluation",
        )
        dataset_choice = DATASET_CHOICES[dataset_label]

        sample_label = st.selectbox(
            "Evaluation Sample Size",
            options=list(SAMPLE_SIZES.keys()),
            index=1,  # Small (100) default
            help="Select subsample size for testing and latency measurement",
        )
        sample_size = SAMPLE_SIZES[sample_label]

        st.divider()

        # ── 3. EXECUTION ENVIRONMENT (Very Bottom) ──
        st.markdown("### Execution Environment")

        # Hardware scan telemetry card
        gpu_label = hw.gpu_name if hw.has_gpu else "No dedicated GPU detected"
        gpu_sub = (
            f"VRAM: {hw.gpu_vram_mb} MB | Driver: {hw.gpu_driver} | Compute: {hw.compute_cap}"
            if hw.has_gpu
            else "Standard multi-threaded CPU execution"
        )

        st.markdown(
            f"""
            <div class="card-panel" style="padding: 10px 14px; margin-bottom: 10px;">
                <div class="card-panel-header">Host Hardware Profile</div>
                <div style="font-size: 0.8rem; color: #f1f5f9; font-weight: 500;">
                    CPU: {hw.cpu_name}
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 2px;">
                    Cores: {hw.cpu_cores_physical} physical, {hw.cpu_cores_logical} logical | RAM: {hw.ram_total_gb:.1f} GB ({hw.ram_available_gb:.1f} GB free)
                </div>
                <div style="font-size: 0.8rem; color: #f1f5f9; font-weight: 500; margin-top: 6px;">
                    GPU: {gpu_label}
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 2px;">
                    {gpu_sub}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Execution Mode Selection
        device_options = ["CPU Safe Mode (Recommended)", "GPU Hardware Acceleration"]
        default_index = 0 if hw.recommended_mode == "cpu" else 1

        selected_mode_str = st.radio(
            "Inference Device Selection",
            options=device_options,
            index=default_index,
            help=(
                "CPU Safe Mode executes inference on host processor (num_gpu=0), "
                "guaranteeing stability on Pascal (sm_61) GPUs and avoiding CUDA PTX buffer overruns. "
                "GPU Acceleration enables CUDA device offloading."
            ),
        )

        num_gpu, mode_display_name = resolve_num_gpu(selected_mode_str)

        # Factual feedback on selected mode
        if num_gpu == 0:
            st.markdown(
                '<div class="notice-box">'
                "<strong>CPU Safe Mode active:</strong> Ollama `num_gpu=0` set. "
                "Inference runs strictly on CPU threads, ensuring reproducibility and zero CUDA runtime crashes."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="notice-box">'
                "<strong>GPU Acceleration active:</strong> Ollama CUDA offloading enabled. "
                "If CUDA toolchain or PTX errors occur, switch back to CPU Safe Mode."
                "</div>",
                unsafe_allow_html=True,
            )

        return selected_model, dataset_choice, sample_size, num_gpu, mode_display_name
