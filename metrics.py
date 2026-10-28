"""
MarkerPhish — Metrics & Analytics
==================================
Calculates binary threat detection effectiveness metrics (Malicious vs. Safe),
granular distribution-channel diagnostics, and hardware resource consumption.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import (
    BINARY_LABELS,
    is_binary_match,
    to_binary_label,
)

if TYPE_CHECKING:
    from evaluator import EvaluationRun


def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> dict:
    """
    Compute classification metrics with primary emphasis on binary threat detection.

    Threat Evaluation Logic (Task 1):
        - True label in {'phishing', 'smishing'} and predicted in {'phishing', 'smishing'}
          -> True Positive (Malicious correctly detected)
        - True label == 'valid' and predicted == 'valid'
          -> True Negative (Safe correctly detected)
        - True label == 'valid' and predicted in {'phishing', 'smishing'}
          -> False Positive (False Alarm: Safe flagged as malicious)
        - True label in {'phishing', 'smishing'} and predicted == 'valid'
          -> False Negative (Missed Threat: Malicious classified as safe)

    In addition, granular distribution-channel metrics (phishing, smishing, valid)
    are retained under `channel_metrics` to prevent any loss of diagnostic detail.
    """
    if labels is None:
        labels = ["phishing", "smishing", "valid"]

    total = len(y_true)
    unknown_count = sum(1 for p in y_pred if p == "unknown")

    # Filter out failed/unknown predictions
    valid_pairs = [(t, p) for t, p in zip(y_true, y_pred) if p != "unknown"]
    y_true_clean = [t for t, _ in valid_pairs]
    y_pred_clean = [p for _, p in valid_pairs]

    empty_binary_cm = [[0, 0], [0, 0]]
    empty_channel_cm = [[0] * len(labels) for _ in labels]

    if not y_true_clean:
        return {
            # Primary binary metrics
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "specificity": 0.0,
            "miss_rate": 0.0,
            "false_alarm_rate": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "binary_confusion_matrix": empty_binary_cm,
            "binary_labels": BINARY_LABELS,
            "per_class": {
                "malicious": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": sum(1 for t in y_true if to_binary_label(t) == "malicious")},
                "safe": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": sum(1 for t in y_true if to_binary_label(t) == "safe")},
            },
            # Channel breakdown (legacy / diagnostic)
            "channel_metrics": {
                "accuracy": 0.0,
                "precision_macro": 0.0,
                "recall_macro": 0.0,
                "f1_macro": 0.0,
                "per_class": {
                    lbl: {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": y_true.count(lbl)}
                    for lbl in labels
                },
                "confusion_matrix": empty_channel_cm,
                "labels": labels,
            },
            "confusion_matrix": empty_channel_cm,
            "labels": labels,
            "unknown_predictions": unknown_count,
            "total_evaluated": total,
            "valid_predictions": 0,
            "classification_report_text": "No valid predictions recorded (all model responses failed or returned 'unknown').",
        }

    # ──────────────────────────────────────────────
    # 1. Binary Threat Detection Calculation
    # ──────────────────────────────────────────────
    y_true_binary = [to_binary_label(t) for t in y_true_clean]
    y_pred_binary = [to_binary_label(p) for p in y_pred_clean]

    tp = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == "malicious" and p == "malicious")
    tn = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == "safe" and p == "safe")
    fp = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == "safe" and p == "malicious")
    fn = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == "malicious" and p == "safe")

    n_valid = len(y_true_binary)
    bin_accuracy = (tp + tn) / n_valid if n_valid > 0 else 0.0
    bin_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    bin_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    bin_f1 = (2 * bin_precision * bin_recall) / (bin_precision + bin_recall) if (bin_precision + bin_recall) > 0 else 0.0

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    miss_rate = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    false_alarm_rate = fp / (tn + fp) if (tn + fp) > 0 else 0.0

    # Macro & weighted binary metrics
    p_macro = precision_score(y_true_binary, y_pred_binary, labels=["malicious", "safe"], average="macro", zero_division=0)
    r_macro = recall_score(y_true_binary, y_pred_binary, labels=["malicious", "safe"], average="macro", zero_division=0)
    f1_macro = f1_score(y_true_binary, y_pred_binary, labels=["malicious", "safe"], average="macro", zero_division=0)

    # Binary confusion matrix: Rows = True [Malicious, Safe], Cols = Pred [Malicious, Safe]
    binary_cm = [
        [tp, fn],
        [fp, tn],
    ]

    safe_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    safe_recall = specificity
    safe_f1 = (
        (2 * safe_precision * safe_recall) / (safe_precision + safe_recall)
        if (safe_precision + safe_recall) > 0
        else 0.0
    )

    binary_per_class = {
        "malicious": {
            "precision": round(bin_precision, 4),
            "recall": round(bin_recall, 4),
            "f1-score": round(bin_f1, 4),
            "support": tp + fn,
        },
        "safe": {
            "precision": round(safe_precision, 4),
            "recall": round(safe_recall, 4),
            "f1-score": round(safe_f1, 4),
            "support": tn + fp,
        },
    }

    # ──────────────────────────────────────────────
    # 2. Channel-Level Multi-Class Breakdown
    # ──────────────────────────────────────────────
    channel_acc = accuracy_score(y_true_clean, y_pred_clean)
    ch_p_macro = precision_score(y_true_clean, y_pred_clean, labels=labels, average="macro", zero_division=0)
    ch_r_macro = recall_score(y_true_clean, y_pred_clean, labels=labels, average="macro", zero_division=0)
    ch_f1_macro = f1_score(y_true_clean, y_pred_clean, labels=labels, average="macro", zero_division=0)
    channel_cm = confusion_matrix(y_true_clean, y_pred_clean, labels=labels)

    channel_report_dict = classification_report(
        y_true_clean,
        y_pred_clean,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    channel_per_class = {}
    for label in labels:
        if label in channel_report_dict:
            channel_per_class[label] = {
                "precision": round(channel_report_dict[label]["precision"], 4),
                "recall": round(channel_report_dict[label]["recall"], 4),
                "f1-score": round(channel_report_dict[label]["f1-score"], 4),
                "support": int(channel_report_dict[label]["support"]),
            }
        else:
            channel_per_class[label] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
                "support": y_true.count(label),
            }

    # Text report formatted cleanly
    binary_report_text = classification_report(
        y_true_binary,
        y_pred_binary,
        labels=["malicious", "safe"],
        zero_division=0,
    )
    channel_report_text = classification_report(
        y_true_clean,
        y_pred_clean,
        labels=labels,
        zero_division=0,
    )
    combined_report_text = (
        "=== BINARY THREAT DETECTION REPORT (Malicious vs. Safe) ===\n\n"
        f"{binary_report_text}\n"
        "=== DISTRIBUTION CHANNEL DIAGNOSTIC REPORT (Phishing vs. Smishing vs. Valid) ===\n\n"
        f"{channel_report_text}"
    )

    return {
        # Primary binary metrics (Task 1)
        "accuracy": round(bin_accuracy, 4),
        "precision": round(bin_precision, 4),
        "recall": round(bin_recall, 4),
        "f1": round(bin_f1, 4),
        "specificity": round(specificity, 4),
        "miss_rate": round(miss_rate, 4),
        "false_alarm_rate": round(false_alarm_rate, 4),
        "precision_macro": round(p_macro, 4),
        "recall_macro": round(r_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "binary_confusion_matrix": binary_cm,
        "binary_labels": BINARY_LABELS,
        "per_class": binary_per_class,
        # Channel breakdown for diagnostics & backwards compatibility (Task 3)
        "channel_metrics": {
            "accuracy": round(channel_acc, 4),
            "precision_macro": round(ch_p_macro, 4),
            "recall_macro": round(ch_r_macro, 4),
            "f1_macro": round(ch_f1_macro, 4),
            "per_class": channel_per_class,
            "confusion_matrix": channel_cm.tolist(),
            "labels": labels,
        },
        "confusion_matrix": channel_cm.tolist(),
        "labels": labels,
        "unknown_predictions": unknown_count,
        "total_evaluated": total,
        "valid_predictions": len(y_true_clean),
        "classification_report_text": combined_report_text,
    }


def compute_resource_metrics(run: "EvaluationRun") -> dict:
    """
    Compute resource-related metrics from an evaluation run.
    """
    latencies = [r.latency_seconds for r in run.results]

    if latencies:
        lat_array = np.array(latencies)
        avg_lat = float(np.mean(lat_array))
        med_lat = float(np.median(lat_array))
        min_lat = float(np.min(lat_array))
        max_lat = float(np.max(lat_array))
        p95_lat = float(np.percentile(lat_array, 95))
    else:
        avg_lat = med_lat = min_lat = max_lat = p95_lat = 0.0

    throughput = (run.completed / run.elapsed_time * 60) if run.elapsed_time > 0 else 0.0

    return {
        "model_name": run.model_name,
        "model_size_mb": round(run.model_size_bytes / (1024 * 1024), 1) if run.model_size_bytes else 0,
        "total_records": run.total_records,
        "completed": run.completed,
        "failed": run.failed,
        "total_time_s": round(run.elapsed_time, 2),
        "avg_latency_s": round(avg_lat, 4),
        "median_latency_s": round(med_lat, 4),
        "min_latency_s": round(min_lat, 4),
        "max_latency_s": round(max_lat, 4),
        "p95_latency_s": round(p95_lat, 4),
        "throughput_msgs_per_min": round(throughput, 2),
    }


def build_full_report(
    run: "EvaluationRun",
    hardware_profile: dict | None = None,
    execution_mode: str = "CPU Safe Mode",
) -> dict:
    """
    Build a comprehensive evaluation report combining binary classification metrics,
    granular channel metrics, computational resource telemetry, and hardware profile.
    """
    y_true = [r.true_label for r in run.results]
    y_pred = [r.predicted_label for r in run.results]

    classification = compute_classification_metrics(y_true, y_pred)
    resources = compute_resource_metrics(run)

    # Distributions
    true_channel_dist = dict(Counter(y_true))
    pred_channel_dist = dict(Counter(y_pred))

    true_binary_dist = dict(Counter(to_binary_label(t) for t in y_true))
    pred_binary_dist = dict(Counter(to_binary_label(p) for p in y_pred))

    return {
        "model_name": run.model_name,
        "execution_mode": execution_mode,
        "hardware_profile": hardware_profile or {},
        "classification_metrics": classification,
        "resource_metrics": resources,
        "data_distribution": {
            "true_labels": true_channel_dist,
            "predicted_labels": pred_channel_dist,
            "true_binary": true_binary_dist,
            "predicted_binary": pred_binary_dist,
        },
        "raw_predictions": [
            {
                "id": r.record_id,
                "true": r.true_label,
                "predicted": r.predicted_label,
                "true_binary": to_binary_label(r.true_label),
                "predicted_binary": to_binary_label(r.predicted_label),
                "is_correct_binary": is_binary_match(r.true_label, r.predicted_label),
                "is_correct_channel": r.true_label == r.predicted_label,
                "raw_response": r.raw_response,
                "latency_s": round(r.latency_seconds, 4),
            }
            for r in run.results
        ],
    }
