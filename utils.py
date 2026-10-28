"""
MarkerPhish — Utility Helpers
==============================
Logging, text cleaning, directory helpers, and result persistence.
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from datetime import datetime

from config import RESULTS_DIR


# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
def setup_logger(name: str = "markerphish", level: int = logging.INFO) -> logging.Logger:
    """Create a configured logger with console output."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(fmt)
    logger.addHandler(console)

    return logger


logger = setup_logger()


# ──────────────────────────────────────────────
# Text cleaning
# ──────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Normalise and clean a message body for LLM consumption.

    Steps:
        1. Unicode normalisation (NFKD)
        2. Remove unprintable chars
        3. Collapse excessive whitespace / newlines
        4. Truncate to a safe length
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^\x20-\x7E\n\t]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    if len(text) > 1500:
        text = text[:1500] + "..."
    return text


# ──────────────────────────────────────────────
# Directory helpers
# ──────────────────────────────────────────────
def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist and return the path."""
    os.makedirs(path, exist_ok=True)
    return path


# ──────────────────────────────────────────────
# Result persistence & retrieval
# ──────────────────────────────────────────────
def save_results(results: dict, model_name: str) -> str:
    """
    Save evaluation results to a timestamped JSON file.
    Returns the path to the saved file.
    """
    ensure_dir(RESULTS_DIR)
    safe_name = model_name.replace(":", "_").replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eval_{safe_name}_{timestamp}.json"
    filepath = os.path.join(RESULTS_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Results saved to %s", filepath)
    return filepath


def list_saved_result_files(model_name: str | None = None) -> list[str]:
    """
    List saved evaluation report filenames (.json) in RESULTS_DIR, sorted newest first.
    Optionally filter by model_name substring.
    """
    if not os.path.isdir(RESULTS_DIR):
        return []

    files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")]
    if model_name:
        safe = model_name.replace(":", "_").replace("/", "_")
        files = [f for f in files if safe in f]

    # Sort primarily by file modification time (most recent first)
    files.sort(
        key=lambda f: os.path.getmtime(os.path.join(RESULTS_DIR, f)),
        reverse=True,
    )
    return files


def load_result_by_filename(filename: str) -> dict | None:
    """Load a specific result JSON file by its filename in RESULTS_DIR."""
    filepath = os.path.join(RESULTS_DIR, filename)
    if not os.path.isfile(filepath):
        logger.warning("Result file does not exist: %s", filepath)
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Could not load result from %s: %s", filepath, e)
        return None


def format_result_label(filename: str) -> str:
    """
    Format a result filename into a concise, professional display string.
    Example: eval_phi3.5_latest_20261022_145434.json -> phi3.5:latest | 2026-10-22 14:54:34
    """
    stem = filename[:-5] if filename.endswith(".json") else filename
    parts = stem.split("_")
    if len(parts) >= 3 and parts[0] == "eval":
        date_str = parts[-2]
        time_str = parts[-1]
        model_part = "_".join(parts[1:-2])
        if len(date_str) == 8 and len(time_str) == 6 and date_str.isdigit() and time_str.isdigit():
            formatted_dt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
            return f"{model_part} | {formatted_dt} ({filename})"
    return filename


def load_latest_results(model_name: str | None = None) -> dict | None:
    """Load the most recent result file, optionally filtered by model name."""
    files = list_saved_result_files(model_name=model_name)
    if not files:
        return None
    return load_result_by_filename(files[0])