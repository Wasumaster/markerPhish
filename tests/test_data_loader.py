"""backend.data_loader: per-source pipeline on synthetic rows, and invariants on the real data."""

import re

import pytest

from backend.data_loader import load_dataset_with_report, merge_sources, phishfuzzer_records, sms_records
from config import LINKS_MIN_BODY_CHARS, MESSAGE_MAX_CHARS, TRUNCATION_MARKER


def _pf(no, typ="Phishing", original=None, body="Please verify", subject="Alert", url="None", sender="a@b.c"):
    return {"No.": no, "Original_ID": original if original is not None else no, "Type": typ,
            "Subject": subject, "Body": body, "Sender": sender, "URL": url}


def test_phishfuzzer_pipeline_counts_and_ids():
    raw = [
        _pf(1),
        _pf(2, typ="Valid", body="Lunch at 12?"),
        _pf(3, typ="Spam"),
        _pf(4, typ="Weird"),
        _pf(5, body="", subject=""),
        _pf(6),  # exact duplicate of 1 (same sender, subject, body, links)
    ]
    records, rep = phishfuzzer_records(raw, "phishfuzzer_seed")
    assert [r["source_id"] for r in records] == ["seed:1", "seed:2"]
    assert [r["group_id"] for r in records] == ["pf:1", "pf:2"]
    assert rep["raw_records"] == 6
    assert rep["excluded_spam"] == 1
    assert rep["unrecognized_label"] == 1
    assert rep["empty"] == 1
    assert rep["dedup_exact_duplicates"] == 1
    assert rep["final"] == 2
    assert rep["labels"] == {"phishing": 1, "valid": 1}
    assert "full_text" not in records[0]
    assert records[0]["text"] == "From: a@b.c\nSubject: Alert\nLinks: none\n\nPlease verify"


def test_rephrased_records_are_grouped_by_original_id():
    raw = [_pf(3301, original=1, body="v1"), _pf(3302, original=1, body="v2"), _pf(3307, original=2, body="w")]
    records, _ = phishfuzzer_records(raw, "phishfuzzer_rephrased")
    assert [r["source_id"] for r in records] == ["rephrased:3301", "rephrased:3302", "rephrased:3307"]
    assert [r["group_id"] for r in records] == ["pf:1", "pf:1", "pf:2"]


def test_sms_pipeline_uses_data_row_numbers():
    rows = [
        {"LABEL": "ham", "TEXT": "See you soon"},
        {"LABEL": "spam", "TEXT": "Cheap loans"},
        {"LABEL": "smishing", "TEXT": "Your parcel fee: pay.example"},
        {"LABEL": "ham", "TEXT": "see you soon!"},  # normalized duplicate of row 1
        {"LABEL": "ham", "TEXT": "   "},
    ]
    records, rep = sms_records(rows)
    assert [r["source_id"] for r in records] == ["sms:1", "sms:3"]
    assert rep["excluded_spam"] == 1
    assert rep["empty"] == 1
    assert rep["dedup_normalized_duplicates"] == 1
    assert all(r["channel"] == "sms" for r in records)


def test_merge_assigns_sequential_ids_and_capacity():
    pf = phishfuzzer_records([_pf(1), _pf(2, typ="Valid", body="ok")], "phishfuzzer_seed")
    sms = sms_records([{"LABEL": "ham", "TEXT": "hi"}, {"LABEL": "smishing", "TEXT": "pay now"}])
    df, rep = merge_sources([pf, sms], "seed+sms")
    assert df["id"].tolist() == [1, 2, 3, 4]
    assert rep["dataset_choice"] == "seed+sms"
    assert [c["records"] for c in rep["cells"]] == [1, 1, 1, 1]
    assert rep["max_balanced_size"] == 4


@pytest.fixture(scope="module")
def seed_sms():
    return load_dataset_with_report("seed+sms")


def test_real_seed_sms_invariants(seed_sms):
    df, rep = seed_sms
    by_source = {s["source"]: s for s in rep["sources"]}
    assert by_source["phishfuzzer_seed"]["excluded_spam"] == 1074
    assert by_source["mendeley_sms"]["excluded_spam"] == 3397
    assert df["source_id"].is_unique
    assert df["text"].str.len().max() <= MESSAGE_MAX_CHARS
    emails = df[df["channel"] == "email"]["text"]
    header = re.compile(r"^From: .+\nSubject: .+\nLinks: .+(\n\n|$)")
    assert emails.map(lambda t: bool(header.match(t))).all()
    assert rep["max_balanced_size"] >= 1000  # the main study's sample size fits
    # links from a cut-off body are in the Links header; a link stays lost only when
    # the header reached its limit, and then the minimum body is still visible
    assert (df.loc[~df["truncated"], "links_moved"] == 0).all()
    assert by_source["phishfuzzer_seed"]["truncation"]["links_moved_total"] > 0
    lost = df[df["body_urls_lost"] > 0]
    assert len(lost) <= 10
    kept_body = lost["text"].str.split("\n\n", n=1).str[1].str.len()
    # either the minimum body is visible, or the URL-field header alone left no room to move links
    ok = (kept_body >= LINKS_MIN_BODY_CHARS - len(TRUNCATION_MARKER)) | (lost["links_moved"] == 0)
    assert ok.all()
    assert not rep["group_aware"]  # seed+sms has no multi-record groups


def test_unknown_dataset_choice_is_rejected():
    with pytest.raises(ValueError, match="Unknown dataset choice"):
        load_dataset_with_report("everything")
