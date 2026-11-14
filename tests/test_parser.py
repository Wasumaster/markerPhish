"""parse_binary_label: every fallback stage of the response parser."""

import pytest

from backend.evaluator import parse_binary_label, parse_label


@pytest.mark.parametrize(
    "raw, expected",
    [
        # 1. JSON
        ('{"label": "malicious"}', "malicious"),
        ('{"label": "valid"}', "valid"),
        ('{"reasoning": "fake DHL fee link", "label": "malicious"}', "malicious"),
        ('{"LABEL": "Valid"}', "valid"),
        ('{"label": "phishing"}', "malicious"),
        ('{"label": "safe"}', "valid"),
        ('{"label": "not malicious"}', "valid"),
        ("```json\n{\"label\": \"malicious\"}\n```", "malicious"),
        ('Sure! {"label": "valid"} Hope this helps.', "valid"),
        # 2. "label" field in malformed JSON
        ("{label: malicious", "malicious"),
        ("'label' = 'valid'", "valid"),
        # 3. Free text
        ("This message is not malicious.", "valid"),
        ("It is not a scam", "valid"),
        ("MALICIOUS", "malicious"),
        ("phishing", "malicious"),
        ("Smishing attempt", "malicious"),
        ("legitimate", "valid"),
        ("This looks like fraud to me", "malicious"),
        ("ham", "valid"),
        # 4. Nothing usable
        ("", "unknown"),
        ("   ", "unknown"),
        ("I cannot decide.", "unknown"),
        ('{"label": "maybe"}', "unknown"),
    ],
)
def test_parse_binary_label(raw, expected):
    assert parse_binary_label(raw) == expected


@pytest.mark.parametrize("value", [None, 42, ["malicious"]])
def test_parse_binary_label_rejects_non_strings(value):
    assert parse_binary_label(value) == "unknown"


def test_json_label_wins_over_reasoning_words():
    # The reasoning mentions "phishing", but the label field decides.
    raw = '{"reasoning": "no phishing signs, ordinary reminder", "label": "valid"}'
    assert parse_binary_label(raw) == "valid"


@pytest.mark.parametrize(
    "raw, expected",
    [("valid", "valid"), ("**Phishing**", "phishing"), ("benign", "valid"), ("smish", "smishing"), ("???", "unknown")],
)
def test_legacy_parse_label(raw, expected):
    assert parse_label(raw) == expected
