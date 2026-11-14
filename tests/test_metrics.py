"""compute_classification_metrics: binary threat metrics with malicious = positive."""

import pytest

from backend.metrics import compute_classification_metrics


def _metrics(pairs):
    y_true = [t for t, _ in pairs]
    y_pred = [p for _, p in pairs]
    return compute_classification_metrics(y_true, y_pred)


def test_known_confusion_matrix():
    # TP=3 (phishing/smishing detected), FN=1, FP=2, TN=4
    pairs = (
        [("phishing", "phishing"), ("smishing", "smishing"), ("phishing", "smishing")]
        + [("smishing", "valid")]
        + [("valid", "phishing"), ("valid", "smishing")]
        + [("valid", "valid")] * 4
    )
    m = _metrics(pairs)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (3, 1, 2, 4)
    assert m["binary_confusion_matrix"] == [[3, 1], [2, 4]]
    assert m["precision"] == pytest.approx(3 / 5, abs=1e-4)
    assert m["recall"] == pytest.approx(3 / 4, abs=1e-4)
    assert m["f1"] == pytest.approx(2 * 0.6 * 0.75 / (0.6 + 0.75), abs=1e-4)
    assert m["accuracy"] == pytest.approx(7 / 10, abs=1e-4)
    assert m["specificity"] == pytest.approx(4 / 6, abs=1e-4)
    assert m["false_alarm_rate"] == pytest.approx(2 / 6, abs=1e-4)
    assert m["miss_rate"] == pytest.approx(1 / 4, abs=1e-4)
    assert m["per_class"]["malicious"]["support"] == 4
    assert m["per_class"]["safe"]["support"] == 6


def test_cross_channel_threat_counts_as_detected():
    # A phishing email predicted as "smishing" is still a detected threat (binary TP),
    # but it is a channel-level mistake.
    m = _metrics([("phishing", "smishing")])
    assert m["tp"] == 1
    assert m["channel_metrics"]["accuracy"] == 0.0


def test_unknown_predictions_are_excluded_and_counted():
    m = _metrics([("phishing", "phishing"), ("valid", "valid"), ("phishing", "unknown"), ("valid", "unknown")])
    assert m["unknown_predictions"] == 2
    assert m["total_evaluated"] == 4
    assert m["valid_predictions"] == 2
    assert m["f1"] == 1.0  # current behaviour: unknown excluded (Phase 5 adds "unknown = error")


def test_all_unknown_returns_zeroed_metrics():
    m = _metrics([("phishing", "unknown"), ("valid", "unknown")])
    assert m["valid_predictions"] == 0
    assert m["unknown_predictions"] == 2
    assert m["f1"] == 0.0 and m["false_alarm_rate"] == 0.0
    assert m["binary_confusion_matrix"] == [[0, 0], [0, 0]]
    assert m["per_class"]["malicious"]["support"] == 1


def test_no_positive_predictions_gives_zero_precision_not_error():
    m = _metrics([("phishing", "valid"), ("valid", "valid")])
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0


def test_channel_confusion_matrix_order():
    m = _metrics([("phishing", "phishing"), ("smishing", "valid"), ("valid", "valid")])
    assert m["labels"] == ["phishing", "smishing", "valid"]
    assert m["confusion_matrix"] == [[1, 0, 0], [0, 0, 1], [0, 0, 1]]
