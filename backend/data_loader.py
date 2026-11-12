"""
MarkerPhish — Data Loader
==========================
Parse, filter, clean, and unify the PhishFuzzer (email) and Mendeley (SMS)
datasets into a single Pandas DataFrame ready for evaluation.

Unified schema:
    id          — unique integer
    text        — cleaned message body
    true_label  — one of: phishing, smishing, valid
    source      — dataset origin tag
"""

import csv
import json
from typing import Literal

import numpy as np
import pandas as pd

from config import (
    PHISHFUZZER_LABEL_MAP,
    PHISHFUZZER_REPHRASED,
    PHISHFUZZER_SEED,
    SAMPLE_SEED,
    SMISHING_CSV,
    SMISHING_LABEL_MAP,
    VALID_LABELS,
)
from .utils import clean_text, logger


# ──────────────────────────────────────────────
# PhishFuzzer loader
# ──────────────────────────────────────────────
def _load_phishfuzzer_json(path: str, source_tag: str) -> list[dict]:
    """
    Load a PhishFuzzer JSON file.
    Combines Subject + Body as the message text and strictly drops 'Spam'.
    """
    logger.info("Loading PhishFuzzer data from %s ...", path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    dropped_spam = 0
    dropped_empty = 0

    for entry in raw:
        raw_label = entry.get("Type", "").strip()
        unified_label = PHISHFUZZER_LABEL_MAP.get(raw_label)

        # Drop spam
        if unified_label is None:
            dropped_spam += 1
            continue

        subject = entry.get("Subject", "") or ""
        body = entry.get("Body", "") or ""
        text = f"Subject: {subject}\n\n{body}" if subject else body
        text = clean_text(text)

        if not text:
            dropped_empty += 1
            continue

        records.append(
            {
                "text": text,
                "true_label": unified_label,
                "source": source_tag,
            }
        )

    logger.info(
        "  -> Loaded %d records (dropped %d spam, %d empty)",
        len(records),
        dropped_spam,
        dropped_empty,
    )
    return records


def load_phishfuzzer_seed() -> list[dict]:
    """Load the 3,300-record seed dataset (~2,226 after spam removal)."""
    return _load_phishfuzzer_json(PHISHFUZZER_SEED, source_tag="phishfuzzer_seed")


def load_phishfuzzer_rephrased() -> list[dict]:
    """Load the 19,800-record rephrased dataset (~13,356 after spam removal)."""
    return _load_phishfuzzer_json(PHISHFUZZER_REPHRASED, source_tag="phishfuzzer_rephrased")


# ──────────────────────────────────────────────
# Mendeley SMS / Smishing loader
# ──────────────────────────────────────────────
def load_smishing_csv() -> list[dict]:
    """
    Load the Mendeley SMS dataset from CSV.
    Labels: ham -> valid, smishing -> smishing, spam -> DROPPED.
    """
    logger.info("Loading Smishing dataset from %s ...", SMISHING_CSV)
    records = []
    dropped_spam = 0
    dropped_empty = 0

    with open(SMISHING_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_label = row.get("LABEL", "").strip().lower()
            unified_label = SMISHING_LABEL_MAP.get(raw_label)

            # Drop spam
            if unified_label is None:
                dropped_spam += 1
                continue

            text = clean_text(row.get("TEXT", ""))
            if not text:
                dropped_empty += 1
                continue

            records.append(
                {
                    "text": text,
                    "true_label": unified_label,
                    "source": "mendeley_sms",
                }
            )

    logger.info(
        "  -> Loaded %d records (dropped %d spam, %d empty)",
        len(records),
        dropped_spam,
        dropped_empty,
    )
    return records


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


def load_unified_dataset(choice: DatasetChoice = "seed+sms") -> pd.DataFrame:
    """
    Load and merge selected datasets into a single DataFrame.
    """
    all_records: list[dict] = []

    if choice in ("seed", "seed+sms", "all"):
        all_records.extend(load_phishfuzzer_seed())

    if choice in ("rephrased", "rephrased+sms", "all"):
        all_records.extend(load_phishfuzzer_rephrased())

    if choice in ("sms", "seed+sms", "rephrased+sms", "all"):
        all_records.extend(load_smishing_csv())

    df = pd.DataFrame(all_records)

    # Safety check: ensure only valid labels survived
    df = df[df["true_label"].isin(VALID_LABELS)].reset_index(drop=True)
    df.insert(0, "id", range(1, len(df) + 1))

    logger.info(
        "Unified dataset: %d records — %s",
        len(df),
        df["true_label"].value_counts().to_dict(),
    )
    return df


# ──────────────────────────────────────────────
# Reproducible stratified sampling
# ──────────────────────────────────────────────
def stratified_sample(df: pd.DataFrame, n: int | None, seed: int = SAMPLE_SEED) -> pd.DataFrame:
    """
    Draw a seeded subsample of exactly n records, stratified by source x label.

    Each stratum (e.g. "mendeley_sms|smishing") gets a share proportional to its
    size; leftover slots go to the strata with the largest fractional remainder
    (largest-remainder method). The result is shuffled with the same seed and
    re-indexed with id 1..n, so every model and prompt variant sees the same records.
    Returns df unchanged if n is None or n >= len(df).
    """
    if n is None or n >= len(df):
        return df
    if n <= 0:
        return df.iloc[0:0].reset_index(drop=True)

    strata = df["source"].astype(str) + "|" + df["true_label"].astype(str)
    counts = strata.value_counts().sort_index()
    quotas = counts * n / len(df)
    alloc = np.floor(quotas).astype(int)
    remainder = int(n - alloc.sum())
    fractions = quotas - alloc
    # Largest fractional part first; ties broken by stratum name for determinism
    order = sorted(counts.index, key=lambda k: (-fractions[k], k))
    for key in order[:remainder]:
        alloc[key] += 1

    parts = [
        df[strata == key].sample(n=int(alloc[key]), random_state=seed)
        for key in counts.index
        if alloc[key] > 0
    ]
    sample = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    sample["id"] = range(1, len(sample) + 1)
    return sample
