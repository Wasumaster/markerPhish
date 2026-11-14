"""channel_for_source and the binary <-> channel label helpers in config."""

import math

import pytest

from config import binary_to_channel_label, channel_for_source, is_binary_match, to_binary_label


@pytest.mark.parametrize(
    "source, expected",
    [
        ("phishfuzzer_seed", "email"),
        ("phishfuzzer_rephrased", "email"),
        ("mendeley_sms", "sms"),
        ("  Mendeley_SMS ", "sms"),
        ("some_other_dataset", "unknown"),
    ],
)
def test_channel_from_source(source, expected):
    assert channel_for_source(source) == expected


def test_source_wins_over_label():
    # The source tag is authoritative even if the label suggests the other channel.
    assert channel_for_source("mendeley_sms", "phishing") == "sms"


@pytest.mark.parametrize("missing", [None, "", "   ", float("nan")])
def test_label_fallback_when_source_missing(missing):
    assert channel_for_source(missing, "phishing") == "email"
    assert channel_for_source(missing, "Smishing") == "sms"
    # "valid" exists in both channels, so it cannot decide
    assert channel_for_source(missing, "valid") == "unknown"
    assert channel_for_source(missing, None) == "unknown"


def test_nan_is_treated_as_missing():
    assert channel_for_source(math.nan, math.nan) == "unknown"


@pytest.mark.parametrize(
    "binary, channel, expected",
    [
        ("malicious", "email", "phishing"),
        ("malicious", "sms", "smishing"),
        ("malicious", "unknown", "phishing"),
        ("valid", "sms", "valid"),
        ("unknown", "email", "unknown"),
        (None, "email", "unknown"),
    ],
)
def test_binary_to_channel_label(binary, channel, expected):
    assert binary_to_channel_label(binary, channel) == expected


def test_binary_label_helpers():
    assert to_binary_label("Phishing") == "malicious"
    assert to_binary_label("valid") == "safe"
    assert to_binary_label("spam") == "unknown"
    assert to_binary_label(None) == "unknown"
    assert is_binary_match("phishing", "smishing")
    assert not is_binary_match("valid", "unknown")
