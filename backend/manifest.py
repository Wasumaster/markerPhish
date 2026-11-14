"""
MarkerPhish — Reproducibility manifest
=======================================
Everything needed to re-run an evaluation and to decide whether two runs are
comparable: protocol, data sample, prompt, model, inference options, software
versions, power state and timestamps.

Collectors that touch the outside world (git, Ollama, Windows registry, battery)
never raise: a value that cannot be read is stored as None.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime

import pandas as pd
import requests

from config import (
    DEV_BUILD,
    OLLAMA_BASE_URL,
    OLLAMA_SHOW_ENDPOINT,
    OLLAMA_TAGS_ENDPOINT,
    PREPROCESSING_VERSION,
    PROJECT_ROOT,
    PROTOCOL_VERSION,
    RESULTS_DIR,
    build_classification_prompt,
)

from .utils import logger

# Placeholder for the message in the stored prompt template.
MESSAGE_PLACEHOLDER = "{{MESSAGE}}"
TEMPLATE_CHANNELS = ("email", "sms")


def now_iso() -> str:
    """Current local time, ISO 8601 with timezone offset."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


# ──────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────
def prompt_template(variant: str, output_schema: dict | None = None) -> dict:
    """
    The exact prompt text of a variant per channel, with MESSAGE_PLACEHOLDER where
    the message goes, plus the JSON Schema sent as Ollama's "format" parameter.
    """
    return {
        "templates": {ch: build_classification_prompt(MESSAGE_PLACEHOLDER, ch, variant) for ch in TEMPLATE_CHANNELS},
        "output_schema": output_schema,
    }


def prompt_template_hash(template: dict) -> str:
    """SHA-256 over the canonical JSON of prompt_template(): every template text and the output schema."""
    return sha256_text(canonical_json(template))


# ──────────────────────────────────────────────
# Data sample
# ──────────────────────────────────────────────
def dataset_fingerprint(df: pd.DataFrame) -> str:
    """
    SHA-256 over (source_id, text, true_label) of every sampled record, sorted by
    source_id, so it identifies the sample regardless of evaluation order. Two
    runs with the same fingerprint saw exactly the same records and texts.
    Falls back to the sequential id when source_id is absent (older data).
    """
    key = "source_id" if "source_id" in df.columns else "id"
    rows = sorted(
        (str(r[key]), str(r["text"]), str(r["true_label"])) for _, r in df[[key, "text", "true_label"]].iterrows()
    )
    return sha256_text(canonical_json(rows))


def sampled_records(df: pd.DataFrame) -> list[dict]:
    """source_id (and group_id) of every sampled record, in evaluation order."""
    cols = [c for c in ("id", "source_id", "group_id") if c in df.columns]
    return [{c: (int(r[c]) if c == "id" else str(r[c])) for c in cols} for _, r in df[cols].iterrows()]


# ──────────────────────────────────────────────
# Software versions
# ──────────────────────────────────────────────
def _git(args: list[str]) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def code_version() -> dict:
    """
    Git commit and dirty flag of the working tree. Saved runs under the results
    directory do not count as changes. Both values are None without git.
    """
    commit = _git(["rev-parse", "HEAD"])
    status = _git(["status", "--porcelain"])
    if commit is None or status is None:
        return {"commit": None, "dirty": None, "changed_files": None}
    results_rel = os.path.relpath(RESULTS_DIR, PROJECT_ROOT).replace("\\", "/").rstrip("/") + "/"
    changed = []
    for line in status.splitlines():
        path = line[3:].strip().strip('"')
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if not path.replace("\\", "/").startswith(results_rel):
            changed.append(path)
    return {"commit": commit.strip(), "dirty": bool(changed), "changed_files": changed}


def ollama_version(timeout: float = 5) -> str | None:
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/version", timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("version")
    except Exception as e:  # noqa: BLE001 - never crash a run over metadata
        logger.debug("Ollama version unavailable: %s", e)
        return None


def model_details(model_name: str, timeout: float = 10) -> dict:
    """
    Model identity from /api/tags (digest, size) and /api/show (details, context
    length). Reading these does not load the model.
    """
    out = {
        "name": model_name,
        "tag": model_name.split(":", 1)[1] if ":" in model_name else "latest",
        "digest": None,
        "size_bytes": None,
        "family": None,
        "parameter_size": None,
        "parameter_count": None,
        "quantization": None,
        "format": None,
        "context_length": None,
        "modified_at": None,
    }
    wanted = {model_name} | ({model_name + ":latest"} if ":" not in model_name else set())
    try:
        resp = requests.get(OLLAMA_TAGS_ENDPOINT, timeout=timeout)
        resp.raise_for_status()
        for m in resp.json().get("models", []):
            if m.get("name") in wanted:
                out["digest"] = m.get("digest")
                out["size_bytes"] = m.get("size")
                out["modified_at"] = m.get("modified_at")
                break
    except Exception as e:  # noqa: BLE001
        logger.debug("Model list unavailable: %s", e)
    try:
        resp = requests.post(OLLAMA_SHOW_ENDPOINT, json={"model": model_name}, timeout=timeout)
        resp.raise_for_status()
        info = resp.json()
        details = info.get("details") or {}
        out["family"] = details.get("family")
        out["parameter_size"] = details.get("parameter_size")
        out["quantization"] = details.get("quantization_level")
        out["format"] = details.get("format")
        model_info = info.get("model_info") or {}
        out["parameter_count"] = model_info.get("general.parameter_count")
        ctx = [v for k, v in model_info.items() if k.endswith(".context_length")]
        out["context_length"] = ctx[0] if ctx else None
    except Exception as e:  # noqa: BLE001
        logger.debug("Model details unavailable for %s: %s", model_name, e)
    return out


# ──────────────────────────────────────────────
# Power state (laptop CPU latency depends on it)
# ──────────────────────────────────────────────
# Well-known Windows power plan GUIDs -> English names (the registry name is localized).
_POWER_PLANS = {
    "381b4222-f694-41f0-9685-ff5bb260df2e": "Balanced",
    "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c": "High performance",
    "a1841308-3541-4fab-bc81-f71556f20b4a": "Power saver",
    "e9a42b02-d5df-448d-aa00-03f14749eb61": "Ultimate performance",
}
# Windows 11 "power mode" overlay GUIDs (Settings > System > Power).
_POWER_MODES = {
    "961cc777-2547-4f9d-8174-7d86181b8a7a": "Best power efficiency",
    "00000000-0000-0000-0000-000000000000": "Balanced",
    "3af9b8d9-7c97-431d-ad78-34a8bfea439f": "Better performance",
    "ded574b5-45a0-4f42-8737-46345c09c238": "Best performance",
}


def _windows_power_plan() -> tuple[dict | None, str | None]:
    """(active plan {guid, name}, active power mode name) from the registry, or None."""
    try:
        import winreg
    except ImportError:
        return None, None
    try:
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes"
        )
    except OSError:
        return None, None
    plan = None
    try:
        guid = str(winreg.QueryValueEx(key, "ActivePowerScheme")[0]).lower()
        name = _POWER_PLANS.get(guid)
        if name is None:
            try:
                friendly = winreg.QueryValueEx(winreg.OpenKey(key, guid), "FriendlyName")[0]
                name = str(friendly).rsplit(",", 1)[-1] or None
            except OSError:
                name = None
        plan = {"guid": guid, "name": name}
    except OSError:
        pass
    return plan, _windows_power_mode(key)


def _windows_power_mode(key) -> str | None:
    import winreg

    on_ac = _on_ac_power()
    value_names = ["ActiveOverlayAcPowerScheme", "ActiveOverlayDcPowerScheme"]
    if on_ac is False:
        value_names.reverse()
    for value_name in value_names:
        try:
            guid = str(winreg.QueryValueEx(key, value_name)[0]).lower()
            return _POWER_MODES.get(guid, guid)
        except OSError:
            continue
    return None


def _on_ac_power() -> bool | None:
    try:
        import psutil

        battery = psutil.sensors_battery()
    except Exception:  # noqa: BLE001
        return None
    if battery is None:
        return True if platform.system() != "Windows" else None  # desktop without battery
    return bool(battery.power_plugged) if battery.power_plugged is not None else None


def power_state() -> dict:
    """
    AC vs battery, battery level, Windows power plan and Windows 11 power mode.
    Any value that cannot be read is None.
    """
    battery_percent = None
    try:
        import psutil

        battery = psutil.sensors_battery()
        battery_percent = round(battery.percent, 1) if battery is not None else None
    except Exception:  # noqa: BLE001
        pass
    plan, mode = _windows_power_plan() if platform.system() == "Windows" else (None, None)
    return {
        "on_ac_power": _on_ac_power(),
        "battery_percent": battery_percent,
        "power_plan": plan,
        "power_mode": mode,
    }


# ──────────────────────────────────────────────
# Manifest
# ──────────────────────────────────────────────
def build_manifest(
    *,
    model_name: str,
    prompt_variant: str,
    output_schema: dict | None,
    dataset_choice: str | None,
    df_sample: pd.DataFrame,
    sample_info: dict | None,
    options_sent: dict,
    request_fields: dict,
    started_at: str,
    ended_at: str,
    power: dict | None = None,
    model: dict | None = None,
    code: dict | None = None,
    ollama: str | None = None,
) -> dict:
    """
    Assemble the manifest block of a run. Collectors (model, code, ollama, power)
    are called here unless passed in (tests and the batch runner pass them).
    """
    template = prompt_template(prompt_variant, output_schema)
    info = sample_info or {}
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dev": DEV_BUILD,
        "preprocessing_version": PREPROCESSING_VERSION,
        "dataset": {
            "dataset_choice": dataset_choice,
            "requested_sample_size": info.get("requested_size"),
            "sample_size": len(df_sample),
            "fallback": info.get("fallback"),
            "fallback_reason": info.get("fallback_reason"),
            "sample_seed": info.get("sample_seed"),
            "group_aware": info.get("group_aware"),
            "n_groups": info.get("n_groups"),
            "allocation_order": info.get("allocation_order"),
            "cells": info.get("cells"),
            "dataset_fingerprint": dataset_fingerprint(df_sample),
            "records": sampled_records(df_sample),
        },
        "prompt": {
            "prompt_variant": prompt_variant,
            "prompt_template_hash": prompt_template_hash(template),
            "message_placeholder": MESSAGE_PLACEHOLDER,
            **template,
        },
        "model": model if model is not None else model_details(model_name),
        "inference": {"options": options_sent, **request_fields},
        "software": {
            "ollama_version": ollama if ollama is not None else ollama_version(),
            # Recorded since the 127.0.0.1 fix; its absence marks latency-inflated runs.
            "ollama_url": OLLAMA_BASE_URL,
            "code": code if code is not None else code_version(),
            "python_version": platform.python_version(),
            "os": platform.platform(),
        },
        # power_at_start / power_at_end / power_changed / on_battery_during_run
        "power": power if power is not None else {"power_at_start": power_state()},
        "timestamps": {"started_at": started_at, "ended_at": ended_at},
    }


# ──────────────────────────────────────────────
# Reading manifests of saved runs
# ──────────────────────────────────────────────
def report_protocol_version(report: dict) -> int:
    """Protocol of a saved run: from its manifest, or 1 (legacy) when it has none."""
    manifest = (report or {}).get("manifest") or {}
    try:
        return int(manifest.get("protocol_version", 1))
    except (TypeError, ValueError):
        return 1


def report_is_dev(report: dict) -> bool:
    """True for runs produced while the refactor was in progress (excluded from Compare by default)."""
    return bool(((report or {}).get("manifest") or {}).get("dev", False))


def manifest_differences(manifest: dict, current: dict) -> list[tuple[str, object, object]]:
    """
    Differences between a run's manifest and the current environment, as
    (field, run value, current value). `current` holds the keys code_commit,
    code_dirty, model_digest, preprocessing_version, prompt_template_hash; a key
    whose current value is None (unknown) is not compared.
    """
    run_values = {
        "code_commit": ((manifest.get("software") or {}).get("code") or {}).get("commit"),
        "code_dirty": ((manifest.get("software") or {}).get("code") or {}).get("dirty"),
        "model_digest": (manifest.get("model") or {}).get("digest"),
        "preprocessing_version": manifest.get("preprocessing_version"),
        "prompt_template_hash": (manifest.get("prompt") or {}).get("prompt_template_hash"),
    }
    diffs = []
    for field, run_value in run_values.items():
        now = current.get(field)
        if now is not None and run_value != now:
            diffs.append((field, run_value, now))
    return diffs


def current_environment(model_name: str, prompt_variant: str, output_schema: dict | None) -> dict:
    """The current values compared by manifest_differences (None where unknown)."""
    code = code_version()
    try:
        template_hash = prompt_template_hash(prompt_template(prompt_variant, output_schema))
    except ValueError:  # e.g. "legacy": no template exists any more
        template_hash = None
    return {
        "code_commit": code["commit"],
        "code_dirty": code["dirty"],
        "model_digest": model_details(model_name).get("digest"),
        "preprocessing_version": PREPROCESSING_VERSION,
        "prompt_template_hash": template_hash,
    }


def _fmt(value) -> str:
    if value is None:
        return "not recorded"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def manifest_summary(manifest: dict) -> list[tuple[str, str]]:
    """The main manifest fields as (label, text) rows for display."""
    ds = manifest.get("dataset") or {}
    prompt = manifest.get("prompt") or {}
    model = manifest.get("model") or {}
    sw = manifest.get("software") or {}
    code = sw.get("code") or {}
    power_block = manifest.get("power") or {}
    power = power_block.get("power_at_start") or power_block  # (Phase-3 dev runs stored one flat dict)
    ts = manifest.get("timestamps") or {}
    plan = power.get("power_plan") or {}
    cells = ds.get("cells") or []
    cell_text = ", ".join(f"{c['channel']} {c['label']} {c['count']}" for c in cells) or None
    options = (manifest.get("inference") or {}).get("options") or {}
    return [
        ("protocol", f"{_fmt(manifest.get('protocol_version'))}" + (" (dev)" if manifest.get("dev") else "")),
        ("preprocessing version", _fmt(manifest.get("preprocessing_version"))),
        ("dataset", _fmt(ds.get("dataset_choice"))),
        ("sample", f"{_fmt(ds.get('sample_size'))} records, seed {_fmt(ds.get('sample_seed'))}"
                   + (f", requested {ds.get('requested_sample_size')}" if ds.get("fallback") else "")),
        ("cells", _fmt(cell_text)),
        ("groups", f"{_fmt(ds.get('n_groups'))} (group-aware: {_fmt(ds.get('group_aware'))})"),
        ("dataset fingerprint", _fmt(ds.get("dataset_fingerprint"))),
        ("prompt", _fmt(prompt.get("prompt_variant"))),
        ("prompt template hash", _fmt(prompt.get("prompt_template_hash"))),
        ("model", f"{_fmt(model.get('name'))} · {_fmt(model.get('parameter_size'))} · {_fmt(model.get('quantization'))}"),
        ("model digest", _fmt(model.get("digest"))),
        ("model context length", _fmt(model.get("context_length"))),
        ("options", ", ".join(f"{k}={v}" for k, v in options.items()) or "not recorded"),
        ("ollama", _fmt(sw.get("ollama_version"))),
        ("code", f"{_fmt(code.get('commit'))}" + (" (uncommitted changes)" if code.get("dirty") else "")),
        ("python / os", f"{_fmt(sw.get('python_version'))} · {_fmt(sw.get('os'))}"),
        ("power at start", f"AC: {_fmt(power.get('on_ac_power'))} · plan: {_fmt(plan.get('name'))} · mode: {_fmt(power.get('power_mode'))}"),
        ("power at end", f"AC: {_fmt((power_block.get('power_at_end') or {}).get('on_ac_power'))}"),
        ("power changed during run", _fmt(power_block.get("power_changed"))),
        ("started / ended", f"{_fmt(ts.get('started_at'))} / {_fmt(ts.get('ended_at'))}"),
    ]


def report_power_warning(report: dict) -> str | None:
    """A warning text when the run was not on AC power the whole time, else None."""
    power = ((report or {}).get("manifest") or {}).get("power") or {}
    start = (power.get("power_at_start") or {}).get("on_ac_power")
    end = (power.get("power_at_end") or {}).get("on_ac_power")
    if power.get("power_changed"):
        return (
            f"AC power changed during this run (at start: {_fmt(start)}, at end: {_fmt(end)}). "
            "On a laptop the CPU clock depends on the power source, so latency is not comparable."
        )
    if power.get("on_battery_during_run"):
        return "This run ran on battery power. On a laptop this lowers CPU clocks, so latency is not comparable."
    return None


LATENCY_INFLATION_NOTE = (
    "Measured before the Ollama URL fix: requests went to \"localhost\", which on Windows first tries "
    "IPv6 and adds about 2 s of connection delay to every request. Latency and throughput of this run are "
    "inflated (about +2 s per message); detection metrics are not affected."
)


def report_latency_inflated(report: dict) -> bool:
    """
    True for runs whose latencies include the ~2 s localhost/IPv6 connection delay:
    every run without a recorded ollama_url (legacy runs and runs made before the
    fix) or with "localhost" in it.
    """
    software = ((report or {}).get("manifest") or {}).get("software") or {}
    url = software.get("ollama_url")
    return url is None or "localhost" in str(url).lower()

