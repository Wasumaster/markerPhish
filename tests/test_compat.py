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


def test_current_reports_keep_their_stored_metrics():
    rep = {
        "classification_metrics": {"binary_confusion_matrix": [[1, 0], [0, 1]], "f1": 1.0},
        "raw_predictions": [{"id": 1, "true": "valid", "predicted": "valid"}],
    }
    before = copy.deepcopy(rep["classification_metrics"])
    out = ensure_report_compatibility(rep)
    assert out["classification_metrics"] == before
    # derived blocks are added to the loaded copy only
    assert out["detection"]["overall"]["unknown_as_error"]["tn"] == 1
    assert out["latency_inflated"] is True  # no manifest -> measured before the 127.0.0.1 fix


def test_latency_inflation_flag():
    from backend.manifest import report_latency_inflated

    assert report_latency_inflated({}) is True
    assert report_latency_inflated({"manifest": {"software": {"ollama_url": "http://localhost:11434"}}}) is True
    assert report_latency_inflated({"manifest": {"software": {"ollama_url": "http://127.0.0.1:11434"}}}) is False


def test_stale_metrics_version_is_recomputed_on_load():
    from backend.detection import METRICS_VERSION

    rep = {
        "classification_metrics": {"binary_confusion_matrix": [[0, 0], [0, 0]]},
        "metrics_version": METRICS_VERSION - 1,
        "detection": {"stale": True},
        "timing": {"stale": True},
        "raw_predictions": [{"id": 1, "true": "phishing", "predicted": "unknown", "channel": "email"}],
    }
    out = ensure_report_compatibility(rep)
    assert "stale" not in out["detection"]
    assert out["detection"]["overall"]["unknown_as_error"]["fn"] == 1
    assert out["metrics_version"] == METRICS_VERSION and out["metrics_version_stored"] == METRICS_VERSION - 1


def test_every_stored_run_still_loads(legacy_results_dir):
    files = list_saved_result_files()
    assert len(files) == len([f for f in os.listdir(legacy_results_dir) if f.endswith(".json")])
    for name in files:
        rep = ensure_report_compatibility(load_result_by_filename(name))
        assert "binary_confusion_matrix" in rep["classification_metrics"], name
