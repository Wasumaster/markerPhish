"""
Device choice: plain CPU / GPU labels, old labels shown as the new ones, the
choice in the Run tab's "advanced" expander (default CPU) and the no-GPU note.

AppTests use the fake Ollama and a fixed hardware profile; nothing touches a
real Ollama server or the machine's GPU.
"""

import os
import re

import pytest
from streamlit.testing.v1 import AppTest

from backend.hardware import HardwareProfile, execution_mode_label, resolve_num_gpu

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app.py")
RUN, RESULTS = 1, 2


# ── labels ──

@pytest.mark.parametrize("label, expected", [
    ("CPU", (0, "CPU")),
    ("GPU", (None, "GPU")),
    ("CPU Safe Mode (Recommended)", (0, "CPU")),  # old UI options still resolve
    ("GPU Hardware Acceleration", (None, "GPU")),
])
def test_resolve_num_gpu(label, expected):
    assert resolve_num_gpu(label) == expected


@pytest.mark.parametrize("stored, shown", [
    ("CPU Safe Mode", "CPU"),
    ("CPU Safe Mode (Recommended)", "CPU"),
    ("GPU Hardware Accelerated", "GPU"),
    ("GPU Hardware Acceleration", "GPU"),
    ("CPU", "CPU"),
    ("GPU", "GPU"),
    (None, "CPU"),
    ("CPU (scikit-learn)", "CPU (scikit-learn)"),  # the baseline keeps its own label
])
def test_old_execution_mode_labels_display_as_cpu_or_gpu(stored, shown):
    assert execution_mode_label(stored) == shown


def test_ui_and_docs_no_longer_use_the_old_names():
    for name in ["frontend/sidebar.py", "frontend/runner.py", "frontend/single_run.py", "README.md", "OVERVIEW.md"]:
        with open(os.path.join(ROOT, name), encoding="utf-8") as f:
            text = f.read()
        assert "CPU Safe Mode" not in text and "Hardware Acceleration" not in text, name


# ── Run tab ──

def _profile(has_gpu: bool) -> HardwareProfile:
    extra = dict(gpu_name="NVIDIA Test", gpu_vram_mb=8192, gpu_driver="1.0", compute_cap="8.6",
                 recommended_mode="gpu") if has_gpu else {}
    return HardwareProfile(cpu_name="Test CPU", cpu_cores_physical=4, cpu_cores_logical=8, ram_total_gb=16.0,
                           ram_available_gb=8.0, has_gpu=has_gpu, **extra)


def _app(monkeypatch, has_gpu=False):
    from backend import hardware

    monkeypatch.setattr(hardware, "scan_hardware", lambda: _profile(has_gpu))
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _text(at, index):
    return " ".join(re.sub(r"<[^>]+>", " ", m.value) for m in at.tabs[index].markdown)


def _device(at):
    return next(r for r in at.tabs[RUN].radio if r.label == "device")


def test_device_choice_is_in_advanced_and_defaults_to_cpu(fake_ollama, results_dir, monkeypatch):
    from frontend.runner import DEVICE_HELP

    at = _app(monkeypatch, has_gpu=True)  # CPU even when a capable GPU is found
    assert not any(r.label == "device" for r in at.sidebar.radio)
    advanced = next(e for e in at.tabs[RUN].expander if e.label == "advanced")
    device = next(r for r in advanced.radio if r.label == "device")
    assert device.options == ["CPU", "GPU"] and device.value == "CPU"
    assert device.help == DEVICE_HELP
    assert DEVICE_HELP.startswith("CPU: the whole model runs on the processor (num_gpu = 0).")
    assert re.search(r"environment\s+CPU\b", _text(at, RUN))
    assert "No NVIDIA GPU detected" not in _text(at, RUN)


def test_no_gpu_note_when_no_nvidia_gpu(fake_ollama, results_dir, monkeypatch):
    at = _app(monkeypatch, has_gpu=False)
    assert "No NVIDIA GPU detected: GPU mode would fall back to CPU on this machine." in _text(at, RUN)


def test_choosing_gpu_reaches_the_run_and_the_summary(fake_ollama, results_dir, monkeypatch):
    import frontend.runner as runner

    seen = {}
    monkeypatch.setattr(runner, "_execute_run",
                        lambda model, df, num_gpu, mode, *a, **k: seen.update(num_gpu=num_gpu, mode=mode))
    at = _app(monkeypatch, has_gpu=False)
    _device(at).set_value("GPU").run()
    assert re.search(r"environment\s+GPU\b", _text(at, RUN))
    assert "CUDA offloading enabled" in _text(at, RUN)
    next(b for b in at.button if b.label == "Run evaluation").click().run()
    assert seen == {"num_gpu": None, "mode": "GPU"}  # GPU path kept; Ollama falls back to CPU itself

    _device(at).set_value("CPU").run()
    next(b for b in at.button if b.label == "Run evaluation").click().run()
    assert seen == {"num_gpu": 0, "mode": "CPU"}


def test_old_run_labelled_cpu_safe_mode_shows_cpu_in_results(fake_ollama, results_dir, toy_df, monkeypatch):
    from backend.evaluator import run_evaluation
    from backend.run_report import build_run_report
    from backend.sampling import balanced_sample
    from backend.utils import save_results

    sample = balanced_sample(toy_df.assign(source_id=[f"t:{i}" for i in toy_df["id"]]), 20)
    run = run_evaluation("acme-small:1b", sample.df, num_gpu=0, prompt_variant="zeroshot")
    report = build_run_report(run, df_sample=sample.df, dataset_choice="seed+sms", sample_info=sample.info,
                              execution_mode="CPU Safe Mode",
                              code={"commit": None, "dirty": None, "changed_files": None})
    save_results(report, "acme-small:1b", prompt_variant="zeroshot")

    at = _app(monkeypatch)
    header = _text(at, RESULTS)
    assert "Safe Mode" not in header
    assert re.search(r"dataset seed\+sms · CPU\b", header)
