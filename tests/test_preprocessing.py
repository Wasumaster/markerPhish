"""backend.preprocessing: email input format, truncation, deduplication."""

import pytest

from backend.preprocessing import (
    build_email_input,
    build_sms_input,
    deduplicate,
    format_email_header,
    normalize_for_dedup,
    parse_url_field,
    truncation_stats,
)
from config import MESSAGE_MAX_CHARS, TRUNCATION_MARKER


@pytest.mark.parametrize(
    "value, expected",
    [
        ("['https://a.com/x', 'https://b.org']", ["https://a.com/x", "https://b.org"]),
        ("['https://a.com', 'https://a.com']", ["https://a.com"]),
        ("[['https://a.com'], ['https://b.com', 'https://a.com']]", ["https://a.com", "https://b.com"]),
        (["https://a.com"], ["https://a.com"]),
        ("https://single.example", ["https://single.example"]),
        (None, []),
        ("None", []),
        ("", []),
        ("[]", []),
        ("[broken", ["[broken"]),
    ],
)
def test_parse_url_field(value, expected):
    assert parse_url_field(value) == expected


def test_header_format_and_placeholders():
    assert format_email_header("Bank <a@b.com>", "Hello", ["https://x.io", "http://y.io"]) == (
        "From: Bank <a@b.com>\nSubject: Hello\nLinks: https://x.io http://y.io"
    )
    assert format_email_header(None, "  ", []) == "From: (unknown)\nSubject: (none)\nLinks: none"


def test_email_input_without_truncation():
    mi = build_email_input("Bank <a@b.com>", "Hi", "['https://x.io']", "Body text.\n\nBye")
    assert mi.text == "From: Bank <a@b.com>\nSubject: Hi\nLinks: https://x.io\n\nBody text.\n\nBye"
    assert mi.text == mi.full_text
    assert not mi.truncated and mi.chars_removed == 0
    assert mi.original_length == len(mi.text)


def test_truncation_cuts_only_the_body_and_keeps_all_links():
    body = "word " * 400  # 2,000 chars
    urls = "['https://evil.example/login', 'https://other.example']"
    mi = build_email_input("x@y.z", "Subject", urls, body, max_chars=500)
    header = "From: x@y.z\nSubject: Subject\nLinks: https://evil.example/login https://other.example"
    assert mi.text.startswith(header + "\n\n")
    assert mi.text.endswith(TRUNCATION_MARKER)
    assert len(mi.text) <= 500
    assert mi.truncated
    assert mi.original_length == len(header) + 2 + len(body.strip())
    assert mi.chars_removed == len(body.strip()) - (500 - len(header) - 2 - len(TRUNCATION_MARKER))


def test_header_longer_than_budget_is_kept_whole():
    urls = str([f"https://example.com/{i}" for i in range(50)])
    mi = build_email_input("s", "t", urls, "the body", max_chars=100)
    assert mi.text.startswith("From: s\nSubject: t\nLinks: https://example.com/0 ")
    assert "https://example.com/49" in mi.text
    assert mi.truncated


def test_links_only_in_the_cut_body_move_to_the_links_header():
    body = (
        "Hello, visit https://kept.example/start today. " + "x" * 300
        + " see https://only-in-body.example/a, https://listed.example and https://only-in-body.example/a again"
    )
    mi = build_email_input("s", "t", "['https://listed.example']", body, max_chars=250, min_body_chars=40)
    links_line = mi.text.split("\n")[2]
    # listed first, then the moved link once; the link still visible in the kept body is not moved
    assert links_line == "Links: https://listed.example https://only-in-body.example/a"
    assert mi.links_moved == 1
    assert mi.body_urls_lost == 0
    assert "https://kept.example/start" in mi.text.split("\n\n", 1)[1]
    assert len(mi.text) <= 250


def test_moving_links_repeats_until_no_link_is_lost():
    # The moved links lengthen the header, which cuts the body further and
    # pushes https://second.example out of the kept part: it must move too.
    body = "a " * 20 + "https://second.example " + "b " * 200 + "https://first.example/" + "p" * 60
    mi = build_email_input("s", "t", None, body, max_chars=160, min_body_chars=20)
    links = mi.text.split("\n")[2]
    assert "https://first.example/" in links and "https://second.example" in links
    assert mi.links_moved == 2


def test_moved_links_never_push_the_body_below_the_minimum():
    links = " ".join(f"https://news.example/item/{i}" for i in range(40))
    body = "Weekly digest. " + "y" * 400 + " " + links
    mi = build_email_input("s", "t", None, body, max_chars=600, min_body_chars=300)
    header, kept_body = mi.text.split("\n\n", 1)
    assert len(mi.text) <= 600
    assert len(kept_body) >= 300 - len(TRUNCATION_MARKER)
    assert 0 < mi.links_moved < 40
    assert mi.links_moved + mi.body_urls_lost == 40
    assert len(header.split("\n")[2].split()) - 1 == mi.links_moved


def test_moved_links_respect_the_character_cap():
    links = " ".join(f"https://t.example/{'x' * 40}{i:02d}" for i in range(30))  # 63 chars each
    body = "Hi. " + "z" * 600 + " " + links
    mi = build_email_input("s", "t", None, body, max_chars=1200, min_body_chars=100, max_moved_chars=200)
    moved = mi.text.split("\n")[2].split()[1:]
    assert mi.links_moved == len(moved) == 3  # 3 * 64 = 192 <= 200 < 4 * 64
    visible = [u for u in links.split() if u in mi.text.split("\n\n", 1)[1]]
    assert visible  # some links are still in the kept part of the body
    assert mi.links_moved + len(visible) + mi.body_urls_lost == 30


def test_emails_that_fit_are_unchanged_by_the_link_rule():
    mi = build_email_input("s", "t", None, "short body with https://x.example")
    assert mi.text.split("\n")[2] == "Links: none"
    assert mi.links_moved == 0


def test_sms_input_is_only_cleaned():
    mi = build_sms_input("  Your   code is\t123  ")
    assert mi.text == "Your code is 123"
    assert not mi.truncated


def test_default_limit_is_the_config_constant():
    mi = build_email_input("s", "t", None, "y" * (MESSAGE_MAX_CHARS * 2))
    assert len(mi.text) <= MESSAGE_MAX_CHARS


def test_normalize_for_dedup():
    assert normalize_for_dedup("  Hello,   WORLD!!  ") == "hello, world"
    assert normalize_for_dedup("\"Call me.\"") == "call me"
    assert normalize_for_dedup("a\n\nb") == "a b"


def _rec(text, label="valid"):
    return {"full_text": text, "true_label": label}


def test_deduplicate_counts_exact_normalized_and_conflicts():
    records = [
        _rec("Win a prize now!", "smishing"),
        _rec("Win a prize now!", "smishing"),  # exact duplicate
        _rec("win a   prize now", "smishing"),  # normalized duplicate
        _rec("See you at 5", "valid"),
        _rec("Pay here", "smishing"),
        _rec("pay here.", "valid"),  # conflicts with "Pay here"
        _rec("Unique message", "valid"),
    ]
    kept, stats = deduplicate(records)
    assert [r["full_text"] for r in kept] == ["Win a prize now!", "See you at 5", "Unique message"]
    assert stats.before == 7 and stats.after == 3
    assert stats.exact_duplicates == 1
    assert stats.normalized_duplicates == 1
    assert stats.label_conflict_records == 2
    assert stats.label_conflict_groups == 1


def test_truncation_stats():
    records = [
        {"text": "a" * 10, "truncated": True, "chars_removed": 100, "links_moved": 2, "body_urls_lost": 1},
        {"text": "b" * 20, "truncated": True, "chars_removed": 300, "links_moved": 1, "body_urls_lost": 0},
        {"text": "c", "truncated": False, "chars_removed": 0, "links_moved": 0, "body_urls_lost": 0},
    ]
    t = truncation_stats(records)
    assert t["truncated"] == 2
    assert t["truncated_pct"] == pytest.approx(2 / 3, abs=1e-4)
    assert t["chars_removed_median"] == 200.0
    assert t["chars_removed_max"] == 300
    assert t["records_with_body_urls_lost"] == 1
    assert t["records_with_links_moved"] == 2
    assert t["links_moved_total"] == 3
    assert t["max_input_chars"] == 20
    assert truncation_stats([])["truncated"] == 0
