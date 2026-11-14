"""
MarkerPhish — Constraint profiles
==================================
Whether a run's measured footprint fits a deployment target such as "a laptop
with 8 GB RAM and no dedicated GPU" (config.CONSTRAINT_PROFILES).

Rule: budget = RAM of the profile x MEMORY_HEADROOM_FACTOR. A run fits the
profile when the model memory reported by Ollama (/api/ps size: weights + KV
cache + buffers) is within the budget and, if the profile sets one, the P95
latency is within its limit. Peak process-tree RSS is reported next to it, but
is not the criterion (it is noisy, see backend.profiler). A profile without a
GPU must hold the whole model in RAM, so the total model size is compared even
if the run itself offloaded layers to a GPU; such a run is flagged, because its
latency was not measured CPU-only.
"""

from __future__ import annotations

from config import CONSTRAINT_PROFILES, MEMORY_HEADROOM_FACTOR

MB_PER_GB = 1024


def profile_budget_mb(profile: dict, headroom: float = MEMORY_HEADROOM_FACTOR) -> float:
    return round(float(profile["ram_gb"]) * MB_PER_GB * headroom, 1)


def fits_profiles(
    resources: dict,
    p95_latency_s: float | None,
    profiles: dict | None = None,
    headroom: float = MEMORY_HEADROOM_FACTOR,
) -> dict:
    """Per profile: budget, model memory, fit flags. fits is None when the model memory is unknown."""
    profiles = profiles if profiles is not None else CONSTRAINT_PROFILES
    model_mb = resources.get("model_size_mb")
    rss_mb = resources.get("peak_ram_mb")
    vram_mb = resources.get("model_vram_mb") or 0
    out = {}
    for name, profile in profiles.items():
        budget = profile_budget_mb(profile, headroom)
        limit = profile.get("max_p95_latency_s")
        fits_memory = None if model_mb is None else model_mb <= budget
        fits_latency = None
        if limit is not None:
            fits_latency = None if p95_latency_s is None else p95_latency_s <= limit
        if fits_memory is None or (limit is not None and fits_latency is None):
            fits = None
        else:
            fits = bool(fits_memory and (fits_latency if limit is not None else True))
        out[name] = {
            "ram_gb": profile["ram_gb"],
            "headroom_factor": headroom,
            "budget_mb": budget,
            "model_memory_mb": model_mb,
            "fits_memory": fits_memory,
            "peak_rss_mb": rss_mb,
            "rss_within_budget": None if rss_mb is None else rss_mb <= budget,
            "max_p95_latency_s": limit,
            "p95_latency_s": p95_latency_s,
            "fits_latency": fits_latency,
            "measured_with_gpu_offload": bool(vram_mb > 0) and not profile.get("gpu", False),
            "fits_profile": fits,
        }
    return out
