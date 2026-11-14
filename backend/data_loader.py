"""
MarkerPhish — Data Loader
==========================
Load the PhishFuzzer (email) and Mendeley (SMS) datasets, turn every record into
the exact model input (backend.preprocessing), exclude spam, deduplicate within
each source, and merge into one DataFrame. The dataset report says what each
step did.

Pipeline per source (PREPROCESSING_VERSION 2):
    1. read rows; assign source_id (and group_id for PhishFuzzer)
    2. drop spam (config.*_EXCLUDED_*) and unrecognized labels, counted separately
    3. build the model input; drop records whose input is empty
    4. deduplicate (exact, normalized, label conflicts) on the untruncated input
    5. the truncated input (header kept, body cut) becomes `text`

Unified schema:
    id               sequential integer (1..N over the merged dataset)
    source_id        stable source identifier: "seed:<No.>", "rephrased:<No.>",
                     "sms:<data row>" (1-based CSV data row, header excluded)
    group_id         PhishFuzzer: "pf:<Original_ID>" (a seed email uses its own No.),
                     so a seed and its rephrasings share one group; SMS: "sms:<row>"
    text             model input (see backend.preprocessing)
    true_label       phishing | smishing | valid
    source           phishfuzzer_seed | phishfuzzer_rephrased | mendeley_sms
    channel          email | sms (from the source)
    truncated        bool: the body was cut to fit the context budget
    original_length  characters of the untruncated input
    chars_removed    body characters cut off
    links_moved      URLs moved from the cut-off body into the Links header
"""

from __future__ import annotations

import csv
import json
from typing import Literal

import pandas as pd

from config import (
    MESSAGE_MAX_CHARS,
    PHISHFUZZER_EXCLUDED_TYPES,
    PHISHFUZZER_LABEL_MAP,
    PHISHFUZZER_REPHRASED,
    PHISHFUZZER_SEED,
    PREPROCESSING_VERSION,
    SMISHING_CSV,
    SMISHING_EXCLUDED_LABELS,
    SMISHING_LABEL_MAP,
    VALID_LABELS,
    channel_for_source,
)

from .preprocessing import (
    build_email_input,
    build_sms_input,
    deduplicate,
    label_counts,
    truncation_stats,
)
from .sampling import cell_capacity
from .utils import clean_text, logger

SOURCE_SEED = "phishfuzzer_seed"
SOURCE_REPHRASED = "phishfuzzer_rephrased"
SOURCE_SMS = "mendeley_sms"

_SOURCE_PREFIX = {SOURCE_SEED: "seed", SOURCE_REPHRASED: "rephrased", SOURCE_SMS: "sms"}

# Columns kept in the unified DataFrame (full_text is only needed for deduplication)
_COLUMNS = [
    "source_id", "group_id", "text", "true_label", "source", "channel",
    "truncated", "original_length", "chars_removed", "links_moved", "body_urls_lost",
]


def _finish_source(records: list[dict], source: str, counts: dict) -> tuple[list[dict], dict]:
    """Deduplicate one source and build its report entry."""
    kept, dedup = deduplicate(records)
    report = {
        "source": source,
        "channel": channel_for_source(source),
        **counts,
        "after_cleaning": len(records),
        **{f"dedup_{k}": v for k, v in dedup.to_dict().items()},
        "final": len(kept),
        "labels": label_counts(kept),
        "truncation": truncation_stats(kept),
    }
    for rec in kept:
        rec.pop("full_text", None)
    logger.info(
        "  -> %s: %d raw, %d spam, %d unrecognized, %d empty, %d exact + %d normalized dupes, "
        "%d label-conflict records, %d final, %d truncated",
        source, counts["raw_records"], counts["excluded_spam"], counts["unrecognized_label"],
        counts["empty"], dedup.exact_duplicates, dedup.normalized_duplicates,
        dedup.label_conflict_records, len(kept), report["truncation"]["truncated"],
    )
    return kept, report


# ──────────────────────────────────────────────
# PhishFuzzer loader
# ──────────────────────────────────────────────
def phishfuzzer_records(raw: list[dict], source: str, max_chars: int = MESSAGE_MAX_CHARS) -> tuple[list[dict], dict]:
    """Preprocess PhishFuzzer entries (already parsed JSON) into unified records + report."""
    prefix = _SOURCE_PREFIX[source]
    counts = {"raw_records": len(raw), "excluded_spam": 0, "unrecognized_label": 0, "empty": 0}
    records = []
    for entry in raw:
        raw_label = str(entry.get("Type", "") or "").strip()
        if raw_label in PHISHFUZZER_EXCLUDED_TYPES:
            counts["excluded_spam"] += 1
            continue
        label = PHISHFUZZER_LABEL_MAP.get(raw_label)
        if label is None:
            counts["unrecognized_label"] += 1
            continue

        # Empty = neither a subject nor a body survives cleaning (a header alone is no message)
        if not clean_text(entry.get("Body") or "") and not clean_text(entry.get("Subject") or ""):
            counts["empty"] += 1
            continue
        mi = build_email_input(entry.get("Sender"), entry.get("Subject"), entry.get("URL"), entry.get("Body"), max_chars)

        number = entry.get("No.")
        original = entry.get("Original_ID")
        group = original if original not in (None, "", "None") else number
        records.append(
            {
                "source_id": f"{prefix}:{number}",
                "group_id": f"pf:{group}",
                "text": mi.text,
                "full_text": mi.full_text,
                "true_label": label,
                "source": source,
                "channel": "email",
                "truncated": mi.truncated,
                "original_length": mi.original_length,
                "chars_removed": mi.chars_removed,
                "links_moved": mi.links_moved,
                "body_urls_lost": mi.body_urls_lost,
            }
        )
    return _finish_source(records, source, counts)


def _load_phishfuzzer_json(path: str, source: str) -> tuple[list[dict], dict]:
    logger.info("Loading PhishFuzzer data from %s ...", path)
    with open(path, "r", encoding="utf-8") as f:
        return phishfuzzer_records(json.load(f), source)


def load_phishfuzzer_seed() -> tuple[list[dict], dict]:
    """Seed set: 3,300 emails (1,126 phishing / 1,100 valid / 1,074 spam)."""
    return _load_phishfuzzer_json(PHISHFUZZER_SEED, SOURCE_SEED)


def load_phishfuzzer_rephrased() -> tuple[list[dict], dict]:
    """LLM-rephrased set: 19,800 emails, six rephrasings per seed email (Original_ID)."""
    return _load_phishfuzzer_json(PHISHFUZZER_REPHRASED, SOURCE_REPHRASED)


# ──────────────────────────────────────────────
# Mendeley SMS loader
# ──────────────────────────────────────────────
def sms_records(rows: list[dict], max_chars: int = MESSAGE_MAX_CHARS) -> tuple[list[dict], dict]:
    """Preprocess Mendeley CSV rows (dicts with LABEL, TEXT) into unified records + report."""
    counts = {"raw_records": len(rows), "excluded_spam": 0, "unrecognized_label": 0, "empty": 0}
    records = []
    for row_number, row in enumerate(rows, start=1):
        raw_label = str(row.get("LABEL", "") or "").strip().lower()
        if raw_label in SMISHING_EXCLUDED_LABELS:
            counts["excluded_spam"] += 1
            continue
        label = SMISHING_LABEL_MAP.get(raw_label)
        if label is None:
            counts["unrecognized_label"] += 1
            continue
        mi = build_sms_input(row.get("TEXT", ""), max_chars)
        if not mi.text:
            counts["empty"] += 1
            continue
        records.append(
            {
                "source_id": f"sms:{row_number}",
                "group_id": f"sms:{row_number}",
                "text": mi.text,
                "full_text": mi.full_text,
                "true_label": label,
                "source": SOURCE_SMS,
                "channel": "sms",
                "truncated": mi.truncated,
                "original_length": mi.original_length,
                "chars_removed": mi.chars_removed,
                "links_moved": mi.links_moved,
                "body_urls_lost": mi.body_urls_lost,
            }
        )
    return _finish_source(records, SOURCE_SMS, counts)


def load_smishing_csv() -> tuple[list[dict], dict]:
    """Mendeley SMS: 10,191 rows (3,397 ham / 3,397 smishing / 3,397 spam)."""
    logger.info("Loading Smishing dataset from %s ...", SMISHING_CSV)
    with open(SMISHING_CSV, "r", encoding="utf-8") as f:
        return sms_records(list(csv.DictReader(f)))


# ──────────────────────────────────────────────
# Unified loader
# ──────────────────────────────────────────────
DatasetChoice = Literal["seed", "rephrased", "sms", "seed+sms", "rephrased+sms", "all"]

DATASET_CHOICES: dict[str, str] = {
    "PhishFuzzer Seed + SMS": "seed+sms",
    "PhishFuzzer Seed only": "seed",
    "PhishFuzzer Rephrased only": "rephrased",
    "Mendeley SMS only": "sms",
    "PhishFuzzer Rephrased + SMS": "rephrased+sms",
    "All datasets combined": "all",
}


def _sources_for(choice: str) -> list:
    if choice not in DATASET_CHOICES.values():
        raise ValueError(f"Unknown dataset choice '{choice}'. Available: {', '.join(DATASET_CHOICES.values())}")
    loaders = []
    if choice in ("seed", "seed+sms", "all"):
        loaders.append(load_phishfuzzer_seed)
    if choice in ("rephrased", "rephrased+sms", "all"):
        loaders.append(load_phishfuzzer_rephrased)
    if choice in ("sms", "seed+sms", "rephrased+sms", "all"):
        loaders.append(load_smishing_csv)
    return loaders


def merge_sources(per_source: list[tuple[list[dict], dict]], choice: str) -> tuple[pd.DataFrame, dict]:
    """Merge preprocessed sources into the unified DataFrame and the dataset report."""
    all_records = [rec for records, _ in per_source for rec in records]
    df = pd.DataFrame(all_records, columns=_COLUMNS)
    df = df[df["true_label"].isin(VALID_LABELS)].reset_index(drop=True)
    df.insert(0, "id", range(1, len(df) + 1))
    capacity = cell_capacity(df)
    report = {
        "dataset_choice": choice,
        "preprocessing_version": PREPROCESSING_VERSION,
        "message_max_chars": MESSAGE_MAX_CHARS,
        "sources": [rep for _, rep in per_source],
        "total_records": len(df),
        **capacity,
    }
    return df, report


def load_dataset_with_report(choice: DatasetChoice = "seed+sms") -> tuple[pd.DataFrame, dict]:
    """Load, preprocess and merge a dataset choice. Returns (DataFrame, dataset report)."""
    df, report = merge_sources([loader() for loader in _sources_for(choice)], choice)
    logger.info(
        "Unified dataset '%s': %d records, cells %s, max balanced size %d",
        choice, len(df), {f"{c['channel']}|{c['label']}": c["records"] for c in report["cells"]},
        report["max_balanced_size"],
    )
    return df, report


def load_unified_dataset(choice: DatasetChoice = "seed+sms") -> pd.DataFrame:
    """Load a dataset choice as one DataFrame (see load_dataset_with_report for the report)."""
    return load_dataset_with_report(choice)[0]
