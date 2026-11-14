"""
MarkerPhish — Sidebar Component
=================================
Renders the application sidebar with model selection, prompt variant, dataset
configuration and execution environment controls.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from backend.data_loader import DATASET_CHOICES
from backend.evaluator import check_ollama_connection, list_available_models
from backend.hardware import resolve_num_gpu
from backend.sampling import sample_size_purpose
from config import (
    ACTIVE_PROMPT_VARIANT,
    DEFAULT_SAMPLE_SIZE,
    MIN_SAMPLE_SIZE,
    PROMPT_VARIANTS,
    SAMPLE_SIZE_PRESETS,
    to_binary_label,
)
from ui.components import esc, note
from ui.theme import INK

# Session key holding the sidebar placeholder for the dataset summary line.
# It is UI-only: the placeholder is recreated on every script run.
_DATASET_SLOT_KEY = "_ui_sidebar_dataset_slot"

_SIZE_LABELS = {50: "50 · smoke test, not for results", 100: "100 · smoke test, not for results"}
_PURPOSE_NOTES = {
    "smoke": "Smoke test, not for results.",
    "pilot": "Testing and pilot only; thesis results use 500 or more records.",
}


def render_sidebar() -> tuple:
    """
    Render the sidebar with all configuration controls.

    Returns:
        tuple: (selected_model, dataset_choice, sample_size, num_gpu, mode_display_name, prompt_variant)
               or (None, None, None, None, None, None) if configuration is incomplete.
    """
    hw = st.session_state["hardware_profile"]
    st.session_state[_DATASET_SLOT_KEY] = None

    with st.sidebar:
        # ── 1. MODEL ──
        connected = check_ollama_connection()
        if not connected:
            st.error("Ollama is offline. Server connection required to evaluate models.")
            note("Start Ollama service: ollama serve", mono=True)
            return None, None, None, None, None, None

        # Installed Models Dropdown
        models = list_available_models()
        if not models:
            st.warning("No locally installed models detected. Execute `ollama pull <model>` in terminal.")
            return None, None, None, None, None, None

        model_names = [m["name"] for m in models]
        model_sizes = {m["name"]: m["size"] for m in models}

        with st.container():
            selected_model = st.selectbox(
                "model",
                options=model_names,
                index=0,
                help="Select a model installed in local Ollama repository",
            )
            if selected_model:
                size_gb = model_sizes.get(selected_model, 0) / (1024 ** 3)
                note(f"{size_gb:.2f} GB on disk", mono=True)

        # ── 2. PROMPT ──
        with st.container():
            variant_keys = list(PROMPT_VARIANTS.keys())
            prompt_variant = st.selectbox(
                "prompt",
                options=variant_keys,
                index=variant_keys.index(ACTIVE_PROMPT_VARIANT),
                help=(
                    "Prompt-engineering setup for the binary decision (malicious vs. valid). "
                    "The channel (email / SMS) is taken from the dataset source, not guessed by the model."
                ),
            )
            st.markdown(
                f'<p class="mp-sidebar-line-sans">{esc(PROMPT_VARIANTS[prompt_variant]["description"])}</p>',
                unsafe_allow_html=True,
            )

        # ── 3. DATASET ──
        with st.container():
            dataset_label = st.selectbox(
                "dataset",
                options=list(DATASET_CHOICES.keys()),
                index=0,
                help="Select dataset combination for evaluation",
            )
            dataset_choice = DATASET_CHOICES[dataset_label]

            size_options = [*SAMPLE_SIZE_PRESETS, "custom"]
            sample_choice = st.selectbox(
                "sample size",
                options=size_options,
                index=size_options.index(DEFAULT_SAMPLE_SIZE),
                format_func=lambda v: _SIZE_LABELS.get(v, str(v)),
                help=(
                    "Records per run, balanced over phishing email / valid email / "
                    "smishing SMS / valid SMS with a fixed seed. Sizes below 500 are for "
                    "testing and the pilot only."
                ),
            )
            if sample_choice == "custom":
                sample_size = int(
                    st.number_input(
                        "custom sample size",
                        min_value=MIN_SAMPLE_SIZE,
                        value=DEFAULT_SAMPLE_SIZE,
                        step=1,
                    )
                )
            else:
                sample_size = int(sample_choice)
            purpose = sample_size_purpose(sample_size)
            if purpose != "results":
                st.markdown(
                    f'<p class="mp-sidebar-line-sans">{esc(_PURPOSE_NOTES[purpose])}</p>',
                    unsafe_allow_html=True,
                )
            # Filled by render_sidebar_dataset_summary() once the sample is loaded
            st.session_state[_DATASET_SLOT_KEY] = st.empty()

        # ── 4. DEVICE ──
        with st.container():
            gpu_label = hw.gpu_name if hw.has_gpu else "no dedicated GPU"
            gpu_sub = (
                f"{hw.gpu_vram_mb} MB VRAM · driver {hw.gpu_driver} · compute {hw.compute_cap}"
                if hw.has_gpu
                else "multi-threaded CPU execution"
            )
            st.markdown(
                '<p class="mp-label">hardware</p>'
                f'<p class="mp-sidebar-line" style="color:{INK};margin-top:0;">{esc(hw.cpu_name)}</p>'
                f'<p class="mp-sidebar-line" style="margin-top:2px;">{hw.cpu_cores_physical} physical / '
                f"{hw.cpu_cores_logical} logical cores · {hw.ram_total_gb:.1f} GB RAM "
                f"({hw.ram_available_gb:.1f} GB free)</p>"
                f'<p class="mp-sidebar-line" style="color:{INK};">{esc(gpu_label)}</p>'
                f'<p class="mp-sidebar-line" style="margin-top:2px;">{esc(gpu_sub)}</p>',
                unsafe_allow_html=True,
            )

        with st.container():
            # Execution Mode Selection
            device_options = ["CPU Safe Mode (Recommended)", "GPU Hardware Acceleration"]
            default_index = 0 if hw.recommended_mode == "cpu" else 1

            selected_mode_str = st.radio(
                "device",
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
                text = (
                    "num_gpu=0: inference runs on CPU threads only, for reproducibility "
                    "and no CUDA runtime crashes."
                )
            else:
                text = (
                    "CUDA offloading enabled. If CUDA toolchain or PTX errors occur, "
                    "switch back to CPU Safe Mode."
                )
            st.markdown(f'<p class="mp-sidebar-line-sans">{esc(text)}</p>', unsafe_allow_html=True)

        return selected_model, dataset_choice, sample_size, num_gpu, mode_display_name, prompt_variant


def render_sidebar_dataset_summary(df: pd.DataFrame | None) -> None:
    """Write '100 msgs · 50 / 50' (malicious / safe) under the dataset controls."""
    slot = st.session_state.get(_DATASET_SLOT_KEY)
    if slot is None or df is None:
        return
    binary = df["true_label"].map(to_binary_label)
    malicious = int((binary == "malicious").sum())
    safe = int((binary == "safe").sum())
    slot.markdown(
        f'<p class="mp-sidebar-line" title="malicious / safe">{len(df):,} msgs · {malicious:,} / {safe:,}</p>',
        unsafe_allow_html=True,
    )
