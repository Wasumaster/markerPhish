"""
MarkerPhish — Optional per-model overrides
===========================================
A small JSON file (config.MODEL_OVERRIDES_PATH) with optional settings per model
name, e.g. written by `python -m backend.calibrate <model>`:

    {
      "some-model:7b": {
        "chars_per_token": 2.31,        # measured, safety factor applied
        "num_predict": 384,             # output cap for this model only
        "digest": "<model digest at calibration>",
        "calibrated_at": "2026-11-14T18:00:00+01:00",
        "measurements": [...]
      }
    }

Nothing is required: a model without an entry uses the global defaults. A
missing, empty or broken file is treated as "no overrides" (logged, never
raised). Model names are never hardcoded; any model Ollama lists can have an
entry.
"""

from __future__ import annotations

import json
import os

from config import MODEL_OVERRIDES_PATH

from .utils import logger

ALLOWED_KEYS = {"chars_per_token", "num_predict", "digest", "calibrated_at", "measurements", "note"}


def _path(path: str | None) -> str:
    """The overrides file (read at call time, so tests can redirect it)."""
    import config

    return path or getattr(config, "MODEL_OVERRIDES_PATH", MODEL_OVERRIDES_PATH)


def load_overrides(path: str | None = None) -> dict:
    path = _path(path)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError) as e:
        logger.warning("Ignoring unreadable model overrides file %s: %s", path, e)
        return {}


def model_override(model_name: str, path: str | None = None) -> dict:
    """The entry for a model ("x" and "x:latest" are the same model), or {}."""
    data = load_overrides(path)
    entry = data.get(model_name)
    if entry is None and ":" not in model_name:
        entry = data.get(model_name + ":latest")
    if entry is None and model_name.endswith(":latest"):
        entry = data.get(model_name[: -len(":latest")])
    return entry if isinstance(entry, dict) else {}


def _positive(value, cast):
    try:
        v = cast(value)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def override_chars_per_token(model_name: str, path: str | None = None) -> float | None:
    return _positive(model_override(model_name, path).get("chars_per_token"), float)


def override_num_predict(model_name: str, path: str | None = None) -> int | None:
    return _positive(model_override(model_name, path).get("num_predict"), int)


def save_override(model_name: str, path: str | None = None, **fields) -> dict:
    """Merge fields into the model's entry and write the file atomically."""
    unknown = set(fields) - ALLOWED_KEYS
    if unknown:
        raise ValueError(f"Unknown override fields: {sorted(unknown)}")
    path = _path(path)
    data = load_overrides(path)
    entry = dict(data.get(model_name) or {})
    entry.update({k: v for k, v in fields.items() if v is not None})
    data[model_name] = entry
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
    os.replace(tmp, path)
    return entry
