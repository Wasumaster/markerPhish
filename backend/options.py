"""
MarkerPhish — Inference options: resolution and validation
===========================================================
The Ollama options of a run are part of its identity. Configurable per run:
num_ctx, num_thread, num_predict, num_batch. temperature (0) and seed are fixed
and never configurable; top_p / top_k are irrelevant at temperature 0 and are
not exposed.

Defaults
- num_ctx: config.OLLAMA_NUM_CTX, never above the model's own context_length
  (from /api/show; unknown -> no cap).
- num_thread, num_batch: not sent (Ollama decides). The value Ollama actually
  used is recorded when it can be read (runner command line), otherwise null.
- num_predict: config.DEFAULT_NUM_PREDICT for every model, unless the model has
  an override (backend.model_overrides) or the user sets one.

Context check (before a run starts)
    estimated tokens(prompt) = len(prompt) / chars_per_token
    required: estimated tokens + num_predict <= num_ctx * (1 - CONTEXT_MARGIN_FRACTION)
chars_per_token is the model's calibrated value if one exists, else the
conservative global estimate. If the longest prompt of the sample does not fit
and num_ctx was not set by the user, num_ctx is raised (in steps of 512) to the
smallest value that fits, up to the model's context_length; the raise is
recorded. If it cannot fit (user-set num_ctx or too small a model context),
the run is refused and the record is named.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import pandas as pd

from config import (
    CHARS_PER_TOKEN_ESTIMATE,
    CONTEXT_MARGIN_FRACTION,
    DEFAULT_NUM_PREDICT,
    OLLAMA_NUM_CTX,
    OLLAMA_SEED,
    OLLAMA_TEMPERATURE,
    OLLAMA_TOP_P,
    build_classification_prompt,
    channel_for_source,
)

from .model_overrides import override_chars_per_token, override_num_predict

CONFIGURABLE = ("num_ctx", "num_thread", "num_predict", "num_batch")
NUM_CTX_STEP = 512
# Options that define a run configuration (compared across runs); num_gpu included.
IDENTITY_KEYS = ("num_ctx", "num_thread", "num_predict", "num_batch", "num_gpu", "temperature", "top_p", "seed")


class OptionsError(ValueError):
    """Invalid options or a sample that cannot fit; the message is shown to the user."""


@dataclass
class ResolvedOptions:
    options: dict  # exactly what is sent to Ollama
    sources: dict  # per option: default | user | model override | capped | auto-raised | ollama default
    chars_per_token: float
    chars_per_token_source: str  # calibrated | default
    model_context_length: int | None
    longest_prompt: dict | None = None  # {source_id, chars, estimated_tokens}
    notes: list[str] = field(default_factory=list)

    def to_manifest(self) -> dict:
        return {
            "option_sources": self.sources,
            "chars_per_token": self.chars_per_token,
            "chars_per_token_source": self.chars_per_token_source,
            "model_context_length": self.model_context_length,
            "longest_prompt": self.longest_prompt,
            "notes": self.notes,
        }


def _int_or_none(value, name: str) -> int | None:
    if value is None or value == "" or value == 0:
        return None
    try:
        v = int(value)
    except (TypeError, ValueError) as e:
        raise OptionsError(f"{name} must be a whole number") from e
    if v < 1:
        raise OptionsError(f"{name} must be at least 1")
    return v


def estimate_tokens(text_chars: int, chars_per_token: float) -> int:
    return int(math.ceil(text_chars / chars_per_token))


def longest_prompt(df: pd.DataFrame, prompt_variant: str) -> dict | None:
    """The record whose full prompt (template + message) is longest."""
    if df is None or df.empty:
        return None
    has_source = "source" in df.columns
    best = None
    for _, row in df.iterrows():
        channel = channel_for_source(row["source"] if has_source else None, row["true_label"])
        chars = len(build_classification_prompt(row["text"], channel, prompt_variant))
        if best is None or chars > best["chars"]:
            best = {"source_id": str(row.get("source_id", row.get("id"))), "chars": chars}
    return best


def required_num_ctx(prompt_chars: int, num_predict: int, chars_per_token: float) -> int:
    """Smallest num_ctx (multiple of NUM_CTX_STEP) that holds the prompt and the output with margin."""
    need = (estimate_tokens(prompt_chars, chars_per_token) + num_predict) / (1 - CONTEXT_MARGIN_FRACTION)
    return int(math.ceil(need / NUM_CTX_STEP) * NUM_CTX_STEP)


def resolve_options(
    *,
    model_name: str,
    prompt_variant: str,
    df_sample: pd.DataFrame | None,
    model_context_length: int | None,
    logical_cores: int,
    num_gpu: int | None,
    user: dict | None = None,
) -> ResolvedOptions:
    """Resolve and validate the options of one run (raises OptionsError)."""
    user = dict(user or {})
    unknown = set(user) - set(CONFIGURABLE)
    if unknown:
        raise OptionsError(f"Not configurable: {', '.join(sorted(unknown))}")
    u_ctx = _int_or_none(user.get("num_ctx"), "num_ctx")
    u_thread = _int_or_none(user.get("num_thread"), "num_thread")
    u_predict = _int_or_none(user.get("num_predict"), "num_predict")
    u_batch = _int_or_none(user.get("num_batch"), "num_batch")
    ctx_limit = int(model_context_length) if model_context_length else None

    sources: dict = {}
    notes: list[str] = []

    if u_thread is not None and u_thread > logical_cores:
        raise OptionsError(f"num_thread {u_thread} is more than the {logical_cores} logical cores of this machine")

    # num_predict
    if u_predict is not None:
        num_predict, sources["num_predict"] = u_predict, "user"
    elif override_num_predict(model_name):
        num_predict, sources["num_predict"] = override_num_predict(model_name), "model override"
    else:
        num_predict, sources["num_predict"] = DEFAULT_NUM_PREDICT, "default"

    # chars per token
    calibrated = override_chars_per_token(model_name)
    cpt = calibrated or CHARS_PER_TOKEN_ESTIMATE
    cpt_source = "calibrated" if calibrated else "default"
    if not calibrated:
        notes.append(
            f"No calibration for {model_name}: the context check uses the default {CHARS_PER_TOKEN_ESTIMATE} "
            f"characters per token. Optional: python -m backend.calibrate {model_name}"
        )

    # num_ctx
    if u_ctx is not None:
        if ctx_limit and u_ctx > ctx_limit:
            raise OptionsError(f"num_ctx {u_ctx} exceeds the model's context length ({ctx_limit})")
        num_ctx, sources["num_ctx"] = u_ctx, "user"
    else:
        num_ctx, sources["num_ctx"] = OLLAMA_NUM_CTX, "default"
        if ctx_limit and num_ctx > ctx_limit:
            num_ctx, sources["num_ctx"] = ctx_limit, "capped at model context length"
            notes.append(f"num_ctx capped at the model's context length ({ctx_limit}).")

    longest = longest_prompt(df_sample, prompt_variant) if df_sample is not None else None
    if longest:
        longest["estimated_tokens"] = estimate_tokens(longest["chars"], cpt)
        needed = required_num_ctx(longest["chars"], num_predict, cpt)
        if needed > num_ctx:
            can_raise = sources["num_ctx"] != "user" and (ctx_limit is None or needed <= ctx_limit)
            if not can_raise:
                limit_text = f"num_ctx {num_ctx}" if sources["num_ctx"] == "user" else f"the model's context ({ctx_limit})"
                raise OptionsError(
                    f"Record {longest['source_id']} does not fit: about {longest['estimated_tokens']} prompt tokens "
                    f"+ {num_predict} output tokens need num_ctx >= {needed} (with a "
                    f"{int(CONTEXT_MARGIN_FRACTION * 100)}% margin), but {limit_text} is smaller."
                )
            notes.append(f"num_ctx raised from {num_ctx} to {needed} so record {longest['source_id']} fits.")
            num_ctx, sources["num_ctx"] = needed, "auto-raised"

    options = {
        "temperature": OLLAMA_TEMPERATURE,
        "top_p": OLLAMA_TOP_P,
        "seed": OLLAMA_SEED,
        "num_ctx": num_ctx,
        "num_predict": num_predict,
    }
    if u_thread is not None:
        options["num_thread"], sources["num_thread"] = u_thread, "user"
    else:
        sources["num_thread"] = "ollama default"
    if u_batch is not None:
        options["num_batch"], sources["num_batch"] = u_batch, "user"
    else:
        sources["num_batch"] = "ollama default"
    if num_gpu is not None:
        options["num_gpu"] = num_gpu
    return ResolvedOptions(options, sources, cpt, cpt_source, ctx_limit, longest, notes)


# ──────────────────────────────────────────────
# What Ollama actually used
# ──────────────────────────────────────────────
_THREADS_RE = re.compile(r"(?:--threads|-t)\s+(\d+)")
_BATCH_RE = re.compile(r"(?:--batch-size|-b)\s+(\d+)")


def resolved_from_runner(cmdlines: list[str]) -> dict:
    """num_thread / num_batch from the runner command line, or None (not shown on every platform)."""
    threads = batch = None
    for cmd in cmdlines:
        if threads is None and (m := _THREADS_RE.search(cmd)):
            threads = int(m.group(1))
        if batch is None and (m := _BATCH_RE.search(cmd)):
            batch = int(m.group(1))
    return {"num_thread": threads, "num_batch": batch}


# ──────────────────────────────────────────────
# Run identity
# ──────────────────────────────────────────────
def run_options(report: dict) -> dict:
    """The identity-relevant options a run was made with (from its manifest; {} for legacy runs)."""
    sent = (((report or {}).get("manifest") or {}).get("inference") or {}).get("options") or {}
    return {k: sent.get(k) for k in IDENTITY_KEYS if k in sent}


def differing_options(reports: list[dict]) -> list[str]:
    """Option names whose values are not the same in all given runs (a missing option counts as a value)."""
    if len(reports) < 2:
        return []
    all_opts = [run_options(r) for r in reports]
    keys = sorted({k for o in all_opts for k in o})
    return [k for k in keys if len({repr(o.get(k)) for o in all_opts}) > 1]


def options_label(report: dict, keys: list[str]) -> str:
    """'num_ctx=4096, num_thread=default' for the given option names."""
    opts = run_options(report)
    return ", ".join(f"{k}={opts.get(k, 'default')}" for k in keys)
