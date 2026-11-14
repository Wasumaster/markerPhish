"""
MarkerPhish — Balanced, channel-stratified, group-aware sampling
=================================================================
A run's sample is balanced over the (channel, label) cells of config.SAMPLING_CELLS:
phishing email, valid email, smishing SMS, valid SMS. A dataset choice with one
channel only (e.g. "sms") is balanced over its two cells.

Allocation. With k cells present, each cell gets floor(n / k) records; the
n mod k extra records go to the cells in SAMPLING_CELLS order (largest-remainder
rule with equal quotas, so every remainder is equal and the fixed order decides).

Groups. Records sharing a group_id (a PhishFuzzer seed email and its rephrasings)
are either all in the sample or all out, so near-identical texts never inflate the
effective sample size. A cell's target is met exactly with whole groups: groups
are visited in a seeded random order and a group is taken only if the remaining
target can still be reached exactly by the groups after it (suffix subset-sum
reachability, kept as Python-int bitsets).

Fallback. If some cell cannot supply its target (too few unique records, or no
exact combination of whole groups), the sample falls back to the largest balanced
size that every cell can supply exactly, and SampleResult.info says so. The
sample is never silently unbalanced.

Everything is deterministic for a given (data, n, seed).
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import (
    MIN_SAMPLE_SIZE,
    RESULTS_MIN_SAMPLE_SIZE,
    SAMPLE_SEED,
    SAMPLING_CELLS,
    SMOKE_TEST_SIZES,
    channel_for_source,
)

CELL_SEPARATOR = "|"


def cell_key(channel: str, label: str) -> str:
    return f"{channel}{CELL_SEPARATOR}{label}"


def sample_size_purpose(n: int) -> str:
    """
    What a sample size is good for: "smoke" (50 / 100: smoke test, not for results),
    "pilot" (below RESULTS_MIN_SAMPLE_SIZE: testing and the pilot only) or "results".
    """
    if n in SMOKE_TEST_SIZES or n < min(SMOKE_TEST_SIZES, default=0):
        return "smoke"
    if n < RESULTS_MIN_SAMPLE_SIZE:
        return "pilot"
    return "results"


def allocate(n: int, k: int) -> list[int]:
    """Equal split of n over k cells; the n mod k extra records go to the first cells."""
    if k <= 0:
        return []
    base, extra = divmod(int(n), k)
    return [base + (1 if i < extra else 0) for i in range(k)]


def _cells_column(df: pd.DataFrame) -> pd.Series:
    channels = [channel_for_source(s, t) for s, t in zip(df["source"], df["true_label"])]
    return pd.Series([cell_key(c, t) for c, t in zip(channels, df["true_label"])], index=df.index)


def _group_column(df: pd.DataFrame, group_col: str) -> pd.Series:
    if group_col in df.columns:
        return df[group_col].astype(str)
    # No groups: every record is its own group
    return pd.Series([f"_row{i}" for i in df.index], index=df.index)


@dataclass
class _CellPlan:
    key: str
    channel: str
    label: str
    group_ids: list[str]  # seeded random order
    sizes: list[int]
    suffix: list[int] = field(default_factory=list)  # suffix[i]: bitset of sums reachable with groups i..end

    def __post_init__(self):
        bits = 1
        suffix = [0] * (len(self.sizes) + 1)
        suffix[-1] = 1
        for i in range(len(self.sizes) - 1, -1, -1):
            bits |= bits << self.sizes[i]
            suffix[i] = bits
        self.suffix = suffix

    @property
    def records(self) -> int:
        return int(sum(self.sizes))

    def can_supply(self, target: int) -> bool:
        return target >= 0 and bool((self.suffix[0] >> target) & 1)

    def choose(self, target: int) -> list[str]:
        """Group ids summing exactly to target, preferring earlier groups in the seeded order."""
        chosen, remaining = [], target
        for i, size in enumerate(self.sizes):
            if remaining == 0:
                break
            if size <= remaining and (self.suffix[i + 1] >> (remaining - size)) & 1:
                chosen.append(self.group_ids[i])
                remaining -= size
        if remaining:
            raise ValueError(f"cell {self.key}: no exact combination for {target}")  # guarded by can_supply
        return chosen


def _plan_cells(df: pd.DataFrame, seed: int, group_col: str) -> tuple[list[_CellPlan], pd.Series, pd.Series]:
    cells = _cells_column(df)
    groups = _group_column(df, group_col)
    plans = []
    for channel, label in SAMPLING_CELLS:
        key = cell_key(channel, label)
        mask = cells == key
        if not mask.any():
            continue
        sizes = groups[mask].value_counts().sort_index()  # sorted ids -> order independent of row order
        # One generator per cell, seeded by (seed, cell): a cell's sample does not
        # depend on which other cells the dataset choice contains.
        rng = np.random.default_rng([int(seed), zlib.crc32(key.encode("utf-8"))])
        order = rng.permutation(len(sizes))
        plans.append(
            _CellPlan(
                key=key,
                channel=channel,
                label=label,
                group_ids=[str(sizes.index[i]) for i in order],
                sizes=[int(sizes.iloc[i]) for i in order],
            )
        )
    return plans, cells, groups


def _feasible(plans: list[_CellPlan], n: int) -> bool:
    targets = allocate(n, len(plans))
    return all(p.can_supply(t) for p, t in zip(plans, targets))


def _largest_feasible(plans: list[_CellPlan], upper: int) -> int:
    if not plans:
        return 0
    k = len(plans)
    # A size above k * (smallest cell) + (k - 1) always overdraws some cell.
    n = min(int(upper), k * min(p.records for p in plans) + k - 1)
    while n > 0 and not _feasible(plans, n):
        n -= 1
    return n


def cell_capacity(df: pd.DataFrame, group_col: str = "group_id") -> dict:
    """
    Unique records and groups available per cell, and the largest balanced sample
    this dataset can supply (with whole groups).
    """
    plans, _, _ = _plan_cells(df, SAMPLE_SEED, group_col)
    return {
        "cells": [
            {"channel": p.channel, "label": p.label, "records": p.records, "groups": len(p.sizes)}
            for p in plans
        ],
        "group_aware": any(s > 1 for p in plans for s in p.sizes),
        "max_balanced_size": _largest_feasible(plans, len(df)),
    }


@dataclass
class SampleResult:
    df: pd.DataFrame
    info: dict


def balanced_sample(
    df: pd.DataFrame,
    n: int,
    seed: int = SAMPLE_SEED,
    group_col: str = "group_id",
) -> SampleResult:
    """
    Draw the balanced, channel-stratified, group-aware sample of n records.

    The returned frame is shuffled with the same seed and re-indexed with id 1..n;
    source_id and group_id (when present) are kept. info records the requested
    and actual size, per-cell targets / counts / capacities, the allocation order,
    the number of groups and whether (and why) the size fell back.
    """
    n = int(n)
    if n < MIN_SAMPLE_SIZE:
        raise ValueError(f"Sample size must be at least {MIN_SAMPLE_SIZE} (got {n}).")

    plans, cells, groups = _plan_cells(df, seed, group_col)
    if not plans:
        raise ValueError("The dataset has no records in any sampling cell.")

    unassigned = int((~cells.isin([p.key for p in plans])).sum())
    actual = n if _feasible(plans, n) else _largest_feasible(plans, n)
    fallback = actual != n
    reason = None
    if fallback:
        targets = allocate(n, len(plans))
        short = [
            (
                f"{p.key} needs {t}, has {p.records} records in {len(p.sizes)} groups"
                if t > p.records
                else f"{p.key} needs {t}, but no combination of whole groups "
                f"(sizes {', '.join(str(x) for x in sorted(set(p.sizes)))}) adds up to exactly {t}"
            )
            for p, t in zip(plans, targets)
            if not p.can_supply(t)
        ]
        reason = (
            f"Requested {n} records, but not every cell can supply its balanced share "
            f"({'; '.join(short)}). Fell back to the largest balanced size: {actual}."
        )
        if actual < MIN_SAMPLE_SIZE:
            raise ValueError(reason + f" That is below the minimum of {MIN_SAMPLE_SIZE}.")

    targets = allocate(actual, len(plans))
    chosen_index = []
    cell_rows = []
    for plan, target in zip(plans, targets):
        chosen_groups = set(plan.choose(target))
        mask = (cells == plan.key) & groups.isin(chosen_groups)
        chosen_index.extend(df.index[mask].tolist())
        cell_rows.append(
            {
                "channel": plan.channel,
                "label": plan.label,
                "target": target,
                "count": int(mask.sum()),
                "capacity_records": plan.records,
                "capacity_groups": len(plan.sizes),
                "groups_in_sample": len(chosen_groups),
            }
        )

    sample = df.loc[chosen_index].sample(frac=1, random_state=seed).reset_index(drop=True)
    sample["id"] = range(1, len(sample) + 1)

    info = {
        "requested_size": n,
        "actual_size": len(sample),
        "fallback": fallback,
        "fallback_reason": reason,
        "sample_seed": seed,
        "allocation_order": [p.key for p in plans],
        "cells": cell_rows,
        "group_aware": any(s > 1 for p in plans for s in p.sizes),
        "n_groups": int(sum(r["groups_in_sample"] for r in cell_rows)),
        "max_balanced_size": _largest_feasible(plans, len(df)),
        "records_outside_cells": unassigned,
    }
    return SampleResult(sample, info)
