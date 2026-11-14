"""
MarkerPhish — Utility Helpers
==============================
Logging, text cleaning, directory helpers, and result persistence.
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
import unicodedata
from datetime import datetime

from config import LEGACY_PROMPT_VARIANT, PROMPT_VARIANTS, RESULTS_DIR


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
# A real HTML tag: "<" directly followed by a known tag name (so plain text such
# as "5 < 6 and a > b" is never treated as markup).
_HTML_TAG_DETECT_RE = re.compile(
    r"</?(?:html|head|body|title|meta|link|style|script|div|span|p|br|hr|a|img|table|thead|tbody|tr|td|th|"
    r"ul|ol|li|h[1-6]|font|center|b|i|u|em|strong|small|blockquote|pre|section|header|footer)\b[^<>]*>",
    re.IGNORECASE,
)
_HTML_DROP_BLOCK_RE = re.compile(r"<(script|style|head)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_ANCHOR_RE = re.compile(
    r"<a\b[^>]*?\bhref\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s>]+))[^>]*>(.*?)</a\s*>",
    re.IGNORECASE | re.DOTALL,
)
_HTML_BREAK_RE = re.compile(r"<br\s*/?>|</(?:p|div|tr|li|h[1-6])\s*>", re.IGNORECASE)
_HTML_ANY_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*(?:\s[^<>]*)?/?>|<![^<>]*>")


def _anchor_to_text(match: re.Match) -> str:
    """Turn <a href="URL">TEXT</a> into "TEXT (URL)" (link targets are a key phishing signal)."""
    href = (match.group(1) or match.group(2) or match.group(3) or "").strip()
    inner = _HTML_ANY_TAG_RE.sub("", match.group(4) or "").strip()
    if not href:
        return inner
    if not inner or inner == href:
        return href
    return f"{inner} ({href})"


def html_to_text(text: str) -> str:
    """Convert an HTML message body to plain text, keeping link targets visible."""
    text = _HTML_DROP_BLOCK_RE.sub(" ", text)
    text = _HTML_COMMENT_RE.sub(" ", text)
    text = _HTML_ANCHOR_RE.sub(_anchor_to_text, text)
    text = _HTML_BREAK_RE.sub("\n", text)
    text = _HTML_ANY_TAG_RE.sub("", text)
    return text


def clean_text(text: str) -> str:
    """
    Normalise and clean a message body for LLM consumption.

    Steps:
        0. HTML bodies -> plain text (links kept as "TEXT (URL)"); unescape entities
        1. Unicode normalisation (NFKD)
        2. Remove unprintable chars
        3. Collapse excessive whitespace / newlines

    It never truncates: the length limit is applied to the message body only,
    by backend.preprocessing, so headers and links are never cut.
    """
    if not isinstance(text, str):
        return ""
    if _HTML_TAG_DETECT_RE.search(text):
        text = html_to_text(text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^\x20-\x7E\n\t]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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
def save_results(results: dict, model_name: str, prompt_variant: str | None = None) -> str:
    """
    Save evaluation results to a timestamped JSON file:
    results/eval_<model>_<prompt_variant>_<timestamp>.json
    (the variant is taken from the report if not given).
    Returns the path to the saved file.
    """
    ensure_dir(RESULTS_DIR)
    safe_name = model_name.replace(":", "_").replace("/", "_")
    variant = prompt_variant or results.get("prompt_variant")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if variant:
        filename = f"eval_{safe_name}_{variant}_{timestamp}.json"
    else:
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
    Example: eval_mymodel_latest_fewshot_20261022_145434.json
             -> mymodel_latest | fewshot | 2026-10-22 14:54:34 (filename)
    Files without a prompt variant (older reports) are shown as "legacy".
    """
    stem = filename[:-5] if filename.endswith(".json") else filename
    parts = stem.split("_")
    if len(parts) >= 3 and parts[0] == "eval":
        date_str = parts[-2]
        time_str = parts[-1]
        model_parts = parts[1:-2]
        variant = LEGACY_PROMPT_VARIANT
        if len(model_parts) > 1 and model_parts[-1] in PROMPT_VARIANTS:
            variant = model_parts[-1]
            model_parts = model_parts[:-1]
        model_part = "_".join(model_parts)
        if len(date_str) == 8 and len(time_str) == 6 and date_str.isdigit() and time_str.isdigit():
            formatted_dt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
            return f"{model_part} | {variant} | {formatted_dt} ({filename})"
    return filename


# ──────────────────────────────────────────────
# Report backward compatibility
# ──────────────────────────────────────────────
def report_prompt_variant(report: dict) -> str:
    """Prompt variant of a report; "legacy" for reports created before variants existed."""
    return (report or {}).get("prompt_variant") or LEGACY_PROMPT_VARIANT


def report_cold_start(report: dict) -> float | None:
    """Cold-start (model load) time in seconds, or None if the report predates it."""
    value = (report or {}).get("resource_metrics", {}).get("cold_start_s")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


# Marker for a stored prediction whose ground-truth label is missing or unreadable.
UNKNOWN_GROUND_TRUTH = "unknown_ground_truth"


def _stored_true_label(record: dict) -> str:
    """Ground-truth label of a stored prediction, or UNKNOWN_GROUND_TRUTH if absent/invalid."""
    from config import VALID_LABELS

    value = record.get("true", record.get("true_channel"))
    if isinstance(value, str) and value.strip().lower() in VALID_LABELS:
        return value.strip().lower()
    return UNKNOWN_GROUND_TRUTH


def ensure_report_compatibility(report: dict) -> dict:
    """
    Ensure saved results from older schema versions have binary metrics populated.

    Re-derives the binary confusion matrix and distributions from raw predictions
    if missing. A record whose ground-truth label is missing or not a known label
    is never assumed to be "valid": it is marked UNKNOWN_GROUND_TRUTH in memory,
    excluded from the metrics and counted in classification_metrics
    ["unknown_ground_truth"]. The file on disk is not modified.
    """
    from config import to_binary_label

    cls = report.get("classification_metrics", {})
    if "binary_confusion_matrix" not in cls and "raw_predictions" in report:
        # Lazy import to avoid circular dependency at module level
        from .metrics import compute_classification_metrics

        raw = report["raw_predictions"]
        y_true, y_pred = [], []
        unknown_truth = 0
        for r in raw:
            true_label = _stored_true_label(r)
            if true_label == UNKNOWN_GROUND_TRUTH:
                r["true"] = UNKNOWN_GROUND_TRUTH
                unknown_truth += 1
                continue
            y_true.append(true_label)
            y_pred.append(r.get("predicted", r.get("predicted_channel", "unknown")) or "unknown")
        recomputed = compute_classification_metrics(y_true, y_pred)
        recomputed["unknown_ground_truth"] = unknown_truth
        report["classification_metrics"] = recomputed

        dist = report.get("data_distribution", {})
        if "true_binary" not in dist:
            dist["true_binary"] = {
                "malicious": sum(1 for t in y_true if to_binary_label(t) == "malicious"),
                "safe": sum(1 for t in y_true if to_binary_label(t) == "safe"),
            }
            dist["predicted_binary"] = {
                "malicious": sum(1 for p in y_pred if to_binary_label(p) == "malicious"),
                "safe": sum(1 for p in y_pred if to_binary_label(p) == "safe"),
            }
            report["data_distribution"] = dist

    _refresh_derived_metrics(report)
    from .manifest import report_latency_inflated

    report["latency_inflated"] = report_latency_inflated(report)
    return report


def _refresh_derived_metrics(report: dict) -> None:
    """
    (Re)compute the detection and timing blocks from raw_predictions when they
    are missing or were computed by an older metrics version. Only the loaded
    copy changes; the file on disk is never rewritten.
    """
    from .detection import METRICS_VERSION, compute_detection, compute_timing

    raw = report.get("raw_predictions")
    if not raw:
        return
    if report.get("metrics_version") == METRICS_VERSION and "detection" in report and "timing" in report:
        return
    report["detection"] = compute_detection(raw)
    wall = (report.get("resource_metrics") or {}).get("total_time_s")
    report["timing"] = compute_timing(raw, wall_time_s=wall)
    report["metrics_version_stored"] = report.get("metrics_version")
    report["metrics_version"] = METRICS_VERSION
