"""
MarkerPhish — Hardware Resource Detection & Execution Environment
==================================================================
Hardware discovery and execution environment management for LLM evaluations.
Scans host CPU, memory, and NVIDIA GPU accelerators to guide optimal device
selection (CPU Safe Mode vs. GPU Acceleration).
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Literal

import psutil

from utils import logger


@dataclass
class HardwareProfile:
    """System compute specifications and execution recommendations."""

    cpu_name: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    ram_total_gb: float
    ram_available_gb: float
    has_gpu: bool
    gpu_name: str | None = None
    gpu_vram_mb: int | None = None
    gpu_driver: str | None = None
    compute_cap: str | None = None
    is_pascal_or_older: bool = False
    recommended_mode: Literal["cpu", "gpu"] = "cpu"
    recommendation_reason: str = ""

    def to_dict(self) -> dict:
        """Serialize hardware profile for persistence in evaluation reports."""
        return asdict(self)


def _detect_cpu_name() -> str:
    """Retrieve the human-readable CPU brand string."""
    # Fast native registry lookup on Windows
    if platform.system() == "Windows":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            val, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            winreg.CloseKey(key)
            if val and isinstance(val, str):
                return val.strip()
        except Exception as e:
            logger.debug("Windows registry CPU lookup failed: %s", e)

    # Fallback to platform/processor identifier
    proc = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER")
    if proc:
        return proc.strip()

    return platform.machine() or "Generic x86_64 Processor"


def _detect_gpu() -> dict:
    """
    Query NVIDIA GPU capabilities via nvidia-smi CLI.
    Returns parsed metadata dict or failure descriptor.
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return {"detected": False, "reason": "nvidia-smi utility not found in system PATH"}

    try:
        proc = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=4,
            check=True,
        )
        output = proc.stdout.strip()
        if not output:
            return {"detected": False, "reason": "nvidia-smi returned empty output"}

        line = output.splitlines()[0].strip()
        parts = [p.strip() for p in line.split(",")]

        name = parts[0] if len(parts) > 0 else "NVIDIA GPU"
        vram_mb = int(float(parts[1])) if len(parts) > 1 and parts[1] else None
        driver = parts[2] if len(parts) > 2 else None
        compute_cap = parts[3] if len(parts) > 3 else None

        return {
            "detected": True,
            "name": name,
            "vram_mb": vram_mb,
            "driver": driver,
            "compute_cap": compute_cap,
        }
    except Exception as e:
        logger.debug("nvidia-smi execution error: %s", e)
        return {"detected": False, "reason": str(e)}


def scan_hardware() -> HardwareProfile:
    """
    Scan host hardware configuration and determine optimal inference execution settings.
    """
    # CPU & RAM
    cpu_name = _detect_cpu_name()
    cores_physical = psutil.cpu_count(logical=False) or 1
    cores_logical = psutil.cpu_count(logical=True) or 1

    mem = psutil.virtual_memory()
    ram_total = round(mem.total / (1024**3), 2)
    ram_avail = round(mem.available / (1024**3), 2)

    # GPU
    gpu_info = _detect_gpu()

    if not gpu_info.get("detected"):
        return HardwareProfile(
            cpu_name=cpu_name,
            cpu_cores_physical=cores_physical,
            cpu_cores_logical=cores_logical,
            ram_total_gb=ram_total,
            ram_available_gb=ram_avail,
            has_gpu=False,
            recommended_mode="cpu",
            recommendation_reason="No dedicated NVIDIA GPU detected. Standard multi-threaded CPU execution enabled.",
        )

    # GPU detected
    gpu_name = gpu_info.get("name", "NVIDIA GPU")
    vram_mb = gpu_info.get("vram_mb")
    driver = gpu_info.get("driver")
    compute_cap = gpu_info.get("compute_cap")

    # Evaluate compute capability for Pascal (sm_60, sm_61) or earlier
    is_pascal_or_older = False
    if compute_cap:
        try:
            cap_val = float(compute_cap)
            if cap_val <= 6.1:
                is_pascal_or_older = True
        except ValueError:
            pass

    if is_pascal_or_older or (vram_mb and vram_mb <= 3500):
        recommended_mode: Literal["cpu", "gpu"] = "cpu"
        reason = (
            f"Detected {gpu_name} ({vram_mb} MB VRAM, Compute {compute_cap}). "
            "Pascal architecture (sm_61) with limited VRAM is susceptible to Ollama CUDA PTX toolchain buffer overruns. "
            "CPU Safe Mode is recommended for guaranteed numerical stability and reproducible evaluation."
        )
    else:
        recommended_mode = "gpu"
        reason = (
            f"Detected {gpu_name} ({vram_mb} MB VRAM, Compute {compute_cap}). "
            "Hardware acceleration ready for CUDA inference."
        )

    return HardwareProfile(
        cpu_name=cpu_name,
        cpu_cores_physical=cores_physical,
        cpu_cores_logical=cores_logical,
        ram_total_gb=ram_total,
        ram_available_gb=ram_avail,
        has_gpu=True,
        gpu_name=gpu_name,
        gpu_vram_mb=vram_mb,
        gpu_driver=driver,
        compute_cap=compute_cap,
        is_pascal_or_older=is_pascal_or_older,
        recommended_mode=recommended_mode,
        recommendation_reason=reason,
    )


def resolve_num_gpu(execution_mode: str) -> tuple[int | None, str]:
    """
    Map the selected UI execution mode to Ollama's num_gpu option and display label.

    Returns:
        tuple[int | None, str]: (num_gpu, display_name)
        num_gpu: 0 for CPU, None for GPU / auto
    """
    mode_lower = execution_mode.lower()
    if "cpu" in mode_lower:
        return 0, "CPU Safe Mode"
    return None, "GPU Hardware Accelerated"
