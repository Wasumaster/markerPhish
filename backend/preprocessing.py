"""
MarkerPhish — Preprocessing (pure functions)
=============================================
Turns raw dataset rows into the exact text the model sees, and reports what
preprocessing did. No I/O, no Streamlit: everything here is unit-tested.

Email model input (fixed format, PREPROCESSING_VERSION 2):

    From: <sender>
    Subject: <subject>
    Links: <url 1> <url 2> ...
    <empty line>
    <body>

- From / Subject: cleaned with clean_text; "(unknown)" / "(none)" when empty.
- Links: the URLs of PhishFuzzer's URL field, flattened, deduplicated in
  first-seen order, separated by single spaces (URLs contain no spaces);
  "none" when the field is empty.
- body: clean_text(Body); only the body is ever truncated. URLs that appear only
  in the cut-off part of the body are appended to the Links header, as long as
  LINKS_MIN_BODY_CHARS of body stay visible (PREPROCESSING_VERSION 3).

SMS model input: clean_text(TEXT), unchanged otherwise.

Truncation: header + body is limited to config.MESSAGE_MAX_CHARS, so the full
prompt fits num_ctx with margin (see config). The header is never cut; the body
gets the remaining characters and ends with TRUNCATION_MARKER when cut.
"""

from __future__ import annotations

import ast
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass

from config import LINKS_MIN_BODY_CHARS, LINKS_MOVED_MAX_CHARS, MESSAGE_MAX_CHARS, TRUNCATION_MARKER

from .utils import clean_text

# ──────────────────────────────────────────────
# Email input format
# ──────────────────────────────────────────────
NO_SENDER = "(unknown)"
NO_SUBJECT = "(none)"
NO_LINKS = "none"

_URL_IN_TEXT_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"')\]]+", re.IGNORECASE)


def _flatten(value):
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten(item)
    elif value is not None:
        yield value


def parse_url_field(value) -> list[str]:
    """
    URLs of a PhishFuzzer URL field, flattened and deduplicated (first-seen order).

    The field is a Python list written as a string ("['https://a', ...]"),
    sometimes nested, a real list, a single URL string, None or "None".
    """
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in ("none", "null", "nan", "[]"):
            return []
        if stripped.startswith("["):
            try:
                value = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                value = [stripped]
        else:
            value = [stripped]
    urls: list[str] = []
    seen: set[str] = set()
    for item in _flatten(value):
        url = re.sub(r"\s+", "", clean_text(str(item)))
        if url and url.lower() not in ("none", "null", "nan") and url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def format_email_header(sender, subject, urls: list[str]) -> str:
    """The From / Subject / Links header (without the trailing empty line)."""
    sender_text = re.sub(r"\s+", " ", clean_text(sender or "")) or NO_SENDER
    subject_text = re.sub(r"\s+", " ", clean_text(subject or "")) or NO_SUBJECT
    links_text = " ".join(urls) if urls else NO_LINKS
    return f"From: {sender_text}\nSubject: {subject_text}\nLinks: {links_text}"


@dataclass(frozen=True)
class ModelInput:
    """The text sent to the model for one record, plus truncation bookkeeping."""

    text: str  # what the model sees (after truncation)
    full_text: str  # untruncated input, used for deduplication
    truncated: bool
    original_length: int  # len(full_text)
    chars_removed: int  # body characters cut off (0 if not truncated)
    links_moved: int  # URLs moved from the cut-off part of the body into the Links header
    body_urls_lost: int  # URLs still lost by truncation (email: only when the header limit was reached)


def _truncate(header: str, body: str, max_chars: int) -> tuple[str, int]:
    """Cut body so header + separator + body <= max_chars. Returns (body, chars_removed)."""
    separator = "\n\n" if header else ""
    room = max_chars - len(header) - len(separator)
    if len(body) <= room:
        return body, 0
    keep = max(0, room - len(TRUNCATION_MARKER))
    cut = body[:keep].rstrip()
    return cut + TRUNCATION_MARKER, len(body) - keep


def urls_in_text(text: str) -> list[str]:
    """URLs written in free text, in order of appearance, deduplicated, trailing punctuation stripped."""
    found: list[str] = []
    for match in _URL_IN_TEXT_RE.findall(text or ""):
        url = match.rstrip(".,;:!?")
        if url and url not in found:
            found.append(url)
    return found


def build_email_input(
    sender,
    subject,
    url_field,
    body,
    max_chars: int = MESSAGE_MAX_CHARS,
    min_body_chars: int = LINKS_MIN_BODY_CHARS,
    max_moved_chars: int = LINKS_MOVED_MAX_CHARS,
) -> ModelInput:
    """
    Model input of one PhishFuzzer email (see module docstring for the format).

    If the body has to be cut, every URL that appears in the cut-off part, and is
    neither in the Links header nor in the kept part of the body, is appended to
    the Links header (in order of appearance). A longer header leaves less room
    for the body, which can cut off further links, so this repeats until no link
    is lost; it terminates because the set of moved links only grows. The header
    may grow only while min_body_chars of body stay visible (or the whole body,
    if shorter), and the moved links take at most max_moved_chars; links beyond
    that stay lost and are counted in body_urls_lost.
    Emails that fit are unchanged.
    """
    urls = parse_url_field(url_field)
    header = format_email_header(sender, subject, urls)
    body_text = clean_text(body or "")
    full = f"{header}\n\n{body_text}" if body_text else header
    cut_body, removed = _truncate(header, body_text, max_chars)
    if not removed:
        return ModelInput(full, full, False, len(full), 0, 0, 0)

    # Moved links may grow the header only while min_body_chars of body remain
    # visible (or the whole body, if shorter), so a newsletter with hundreds of links
    # cannot push out the message itself or the context budget.
    header_limit = max_chars - 2 - min(min_body_chars, len(body_text))
    body_urls = urls_in_text(body_text)
    moved: list[str] = []
    while True:
        listed = set(urls) | set(moved)
        visible = set(urls_in_text(cut_body))
        lost = [u for u in body_urls if u not in listed and u not in visible]
        added = False
        for url in lost:
            candidate = format_email_header(sender, subject, urls + moved + [url])
            moved_chars = sum(len(u) + 1 for u in moved) + len(url) + 1
            if len(candidate) > header_limit or moved_chars > max_moved_chars:
                break
            moved.append(url)
            added = True
        if not added:
            break
        header = format_email_header(sender, subject, urls + moved)
        cut_body, removed = _truncate(header, body_text, max_chars)
    still_lost = len(lost)
    text = f"{header}\n\n{cut_body}" if cut_body else header
    return ModelInput(text, full, removed > 0, len(full), removed, len(moved), still_lost)


def build_sms_input(text, max_chars: int = MESSAGE_MAX_CHARS) -> ModelInput:
    """Model input of one SMS: the cleaned text (truncated only in pathological cases)."""
    full = clean_text(text or "")
    cut, removed = _truncate("", full, max_chars)
    lost = len(set(urls_in_text(full)) - set(urls_in_text(cut))) if removed else 0
    return ModelInput(cut, full, removed > 0, len(full), removed, 0, lost)


# ──────────────────────────────────────────────
# Deduplication
# ──────────────────────────────────────────────
_EDGE_PUNCT = ".,;:!?'\"`()[]{}<>-_*~ \t\n"


def normalize_for_dedup(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation at both ends."""
    return re.sub(r"\s+", " ", str(text).lower()).strip(_EDGE_PUNCT)


@dataclass
class DedupStats:
    before: int = 0
    after: int = 0
    exact_duplicates: int = 0  # removed copies identical to an earlier record
    normalized_duplicates: int = 0  # removed copies equal only after normalization
    label_conflict_records: int = 0  # records dropped because copies disagree on the label
    label_conflict_groups: int = 0  # distinct texts with conflicting labels

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def deduplicate(records: list[dict], key: str = "full_text", label_key: str = "true_label") -> tuple[list[dict], DedupStats]:
    """
    Remove duplicates within one source (call once per source).

    Records are grouped by normalize_for_dedup(record[key]). In a group whose
    records disagree on the label, every copy is dropped (counted as a label
    conflict). Otherwise the first record is kept; each later copy counts as an
    exact duplicate if its text equals an earlier kept-or-seen text exactly, else
    as a normalized duplicate. Input order is preserved.
    """
    stats = DedupStats(before=len(records))
    by_norm: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        by_norm[normalize_for_dedup(rec[key])].append(i)

    drop: set[int] = set()
    for indices in by_norm.values():
        labels = {records[i][label_key] for i in indices}
        if len(labels) > 1:
            drop.update(indices)
            stats.label_conflict_records += len(indices)
            stats.label_conflict_groups += 1
            continue
        seen_exact = {records[indices[0]][key]}
        for i in indices[1:]:
            if records[i][key] in seen_exact:
                stats.exact_duplicates += 1
            else:
                stats.normalized_duplicates += 1
                seen_exact.add(records[i][key])
            drop.add(i)

    kept = [rec for i, rec in enumerate(records) if i not in drop]
    stats.after = len(kept)
    return kept, stats


# ──────────────────────────────────────────────
# Reporting helpers
# ──────────────────────────────────────────────
def truncation_stats(records: list[dict]) -> dict:
    """Truncation summary of one source's (deduplicated) records."""
    removed = [r["chars_removed"] for r in records if r.get("truncated")]
    n = len(records)
    return {
        "records": n,
        "truncated": len(removed),
        "truncated_pct": round(len(removed) / n, 4) if n else 0.0,
        "chars_removed_median": float(statistics.median(removed)) if removed else 0.0,
        "chars_removed_max": int(max(removed)) if removed else 0,
        "records_with_links_moved": sum(1 for r in records if r.get("links_moved", 0) > 0),
        "links_moved_total": int(sum(r.get("links_moved", 0) for r in records)),
        "records_with_body_urls_lost": sum(1 for r in records if r.get("body_urls_lost", 0) > 0),
        "max_input_chars": max((len(r["text"]) for r in records), default=0),
    }


def label_counts(records: list[dict]) -> dict:
    return dict(Counter(r["true_label"] for r in records))
