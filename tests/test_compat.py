"""ensure_report_compatibility and loading of every stored run."""

import copy
import os

from backend.utils import (
    UNKNOWN_GROUND_TRUTH,
    ensure_report_compatibility,
    list_saved_result_files,
    load_result_by_filename,
)


def _legacy_report(raw):
    return {"model_name": "m", "classification_metrics": {"accuracy": 0.5}, "raw_predictions": raw}


def test_legacy_report_gets_binary_metrics():
    rep = _legacy_report(
        [
            {"id": 1, "true": "phishing", "predicted": "phishing"},
            {"id": 2, "true": "valid", "predicted": "smishing"},
            {"id": 3, "true": "smishing", "predicted": "unknown"},
        ]
    )
    out = ensure_report_compatibility(rep)
    cls = out["classification_metrics"]
    assert cls["binary_confusion_matrix"] == [[1, 0], [1, 0]]
    assert cls["unknown_predictions"] == 1
    assert cls["unknown_ground_truth"] == 0
    assert out["data_distribution"]["true_binary"] == {"malicious": 2, "safe": 1}


def test_missing_ground_truth_is_not_assumed_valid():
    rep = _legacy_report(
        [
            {"id": 1, "true": "phishing", "predicted": "phishing"},
            {"id": 2, "predicted": "valid"},  # no label at all
            {"id": 3, "true": "", "predicted": "phishing"},  # empty
            {"id": 4, "true": "spam", "predicted": "valid"},  # not a known label
        ]
    )
    out = ensure_report_compatibility(rep)
    cls = out["classification_metrics"]
    assert cls["unknown_ground_truth"] == 3
    assert cls["total_evaluated"] == 1
    # Before the fix, record 3 would have been a false positive on an assumed "valid".
    assert cls["fp"] == 0 and cls["tn"] == 0
    assert [r["true"] for r in out["raw_predictions"]] == ["phishing"] + [UNKNOWN_GROUND_TRUTH] * 3


def test_true_channel_key_from_old_schema_is_read():
    rep = _legacy_report([{"id": 1, "true_channel": "smishing", "predicted_channel": "valid"}])
    cls = ensure_report_compatibility(rep)["classification_metrics"]
    assert cls["fn"] == 1


def test_current_reports_are_left_untouched():
    rep = {
        "classification_metrics": {"binary_confusion_matrix": [[1, 0], [0, 1]], "f1": 1.0},
        "raw_predictions": [{"id": 1, "predicted": "valid"}],
    }
    before = copy.deepcopy(rep)
    assert ensure_report_compatibility(rep) == before


def test_every_stored_run_still_loads(legacy_results_dir):
    files = list_saved_result_files()
    assert len(files) == len([f for f in os.listdir(legacy_results_dir) if f.endswith(".json")])
    for name in files:
        rep = ensure_report_compatibility(load_result_by_filename(name))
        assert "binary_confusion_matrix" in rep["classification_metrics"], name
