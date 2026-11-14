"""
MarkerPhish — Non-LLM baseline: TF-IDF + logistic regression
=============================================================
    python -m backend.baseline [--dataset seed+sms] [--size 1000] [--seed 42]

A classical reference point, evaluated on exactly the same sample as the LLM
runs and saved as a normal run JSON (model name "baseline-tfidf-lr").

Training data: every record of the dataset choice that is NOT in the
evaluation sample, minus every record sharing a group_id with a sampled record
(a PhishFuzzer seed email and its rephrasings), so no sampled text or close
paraphrase is seen in training.

Caveat (stated in the report, Compare and OVERVIEW.md): the baseline is trained
in-distribution, on the same datasets the sample comes from, whereas the LLMs
never saw these datasets. The comparison is a reference point, not a fair
contest.

Resources: only prediction is profiled, with the same ResourceSampler pointed
at this Python process (one record at a time, so latency per message is
comparable). Training time and training memory are reported separately.
"model memory" (the optimum criterion) is the size of the serialized fitted
vectorizer + classifier.
"""

from __future__ import annotations

import argparse
import os
import pickle
import platform
import sys
import time
import tracemalloc

import numpy as np
import pandas as pd
import psutil
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from config import SAMPLE_SEED, binary_to_channel_label, channel_for_source, is_binary_match, to_binary_label

from .evaluator import EvaluationRun, InferenceResult
from .hardware import hardware_block
from .manifest import now_iso, power_state
from .profiler import ResourceSampler, TreeReader

BASELINE_NAME = "baseline-tfidf-lr"
BASELINE_VARIANT = "baseline"
CAVEAT = (
    "Trained in-distribution: the baseline learns from the same datasets the evaluation sample is drawn from "
    "(the sampled records and their rephrasings are excluded), while the language models were never trained "
    "on them. It is a reference point, not a fair contest."
)
VECTORIZER_PARAMS = {"ngram_range": (1, 2), "min_df": 2, "max_features": 100_000, "sublinear_tf": True,
                     "strip_accents": "unicode", "lowercase": True}
CLASSIFIER_PARAMS = {"C": 4.0, "max_iter": 2000, "class_weight": "balanced", "solver": "liblinear"}
MB = 1024 * 1024


def training_split(df_full: pd.DataFrame, df_sample: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Training records: not sampled, and no group shared with the sample."""
    sampled_ids = set(df_sample["source_id"].astype(str))
    sampled_groups = set(df_sample["group_id"].astype(str)) if "group_id" in df_sample else set()
    is_sampled = df_full["source_id"].astype(str).isin(sampled_ids)
    shares_group = df_full["group_id"].astype(str).isin(sampled_groups) & ~is_sampled
    train = df_full[~is_sampled & ~shares_group]
    return train, {
        "dataset_records": int(len(df_full)),
        "excluded_sampled": int(is_sampled.sum()),
        "excluded_same_group": int(shares_group.sum()),
        "training_records": int(len(train)),
        "training_malicious": int(train["true_label"].map(to_binary_label).eq("malicious").sum()),
        "training_safe": int(train["true_label"].map(to_binary_label).eq("safe").sum()),
    }


def train(train_df: pd.DataFrame, seed: int = SAMPLE_SEED) -> tuple:
    """Fit vectorizer + classifier; returns (vectorizer, classifier, training stats)."""
    y = (train_df["true_label"].map(to_binary_label) == "malicious").astype(int).to_numpy()
    rss_before = psutil.Process().memory_info().rss
    tracemalloc.start()
    start = time.perf_counter()
    vectorizer = TfidfVectorizer(**VECTORIZER_PARAMS)
    X = vectorizer.fit_transform(train_df["text"].tolist())
    clf = LogisticRegression(random_state=seed, **CLASSIFIER_PARAMS)
    clf.fit(X, y)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    stats = {
        "training_time_s": round(elapsed, 3),
        "training_peak_alloc_mb": round(peak / MB, 1),  # Python allocations during fit (tracemalloc)
        "training_rss_delta_mb": round((psutil.Process().memory_info().rss - rss_before) / MB, 1),
        "vocabulary_size": len(vectorizer.vocabulary_),
    }
    return vectorizer, clf, stats


def model_size_mb(vectorizer, clf) -> float:
    return round(len(pickle.dumps((vectorizer, clf))) / MB, 2)


def evaluate(
    df_full: pd.DataFrame,
    df_sample: pd.DataFrame,
    seed: int = SAMPLE_SEED,
    sampler_factory=None,
) -> tuple[EvaluationRun, dict]:
    """Train on the non-sampled records, predict the sample. Returns (run, baseline block)."""
    train_df, split = training_split(df_full, df_sample)
    if train_df["true_label"].map(to_binary_label).nunique() < 2:
        raise ValueError("The training data must contain both malicious and safe records.")
    vectorizer, clf, train_stats = train(train_df, seed)

    run = EvaluationRun(model_name=BASELINE_NAME, total_records=len(df_sample), prompt_variant=BASELINE_VARIANT)
    run.start_mode = None  # no model server, no warm / cold start
    run.started_at = now_iso()
    run.power_at_start = power_state()
    run.hardware = hardware_block(None)
    run.options_sent = {"vectorizer": {k: list(v) if isinstance(v, tuple) else v for k, v in VECTORIZER_PARAMS.items()},
                        "classifier": CLASSIFIER_PARAMS, "random_state": seed, "decision_threshold": 0.5}
    run.request_fields = {"endpoint": None, "format": None, "think": None}
    run.model_size_bytes = int(model_size_mb(vectorizer, clf) * MB)
    sampler = (sampler_factory or (lambda name: ResourceSampler(
        name, tree=TreeReader(lambda: [os.getpid()]), ps_fetcher=lambda: None)))(BASELINE_NAME)
    run.baseline_ram_mb = sampler.start()
    run.start_time = time.time()
    has_source = "source" in df_sample.columns
    try:
        for _, row in df_sample.iterrows():
            true_label = row["true_label"]
            channel = channel_for_source(row["source"] if has_source else None, true_label)
            start = time.perf_counter()
            p = float(clf.predict_proba(vectorizer.transform([row["text"]]))[0, 1])
            latency = time.perf_counter() - start
            binary = "malicious" if p >= 0.5 else "valid"
            predicted = binary_to_channel_label(binary, channel)
            run.results.append(InferenceResult(
                record_id=row["id"], true_label=true_label, predicted_label=predicted,
                raw_response=f'{{"p_malicious": {p:.6f}}}', latency_seconds=latency,
                is_binary_correct=is_binary_match(true_label, predicted), true_binary=to_binary_label(true_label),
                predicted_binary=to_binary_label(predicted), channel=channel,
                source_id=str(row.get("source_id", "")), group_id=str(row.get("group_id", "")),
                score=round(p, 6),
            ))
            run.completed += 1
    finally:
        run.end_time = time.time()
        run.resources = sampler.stop()
        run.ended_at = now_iso()
        run.power_at_end = power_state()
    run.resources["model_size_mb"] = model_size_mb(vectorizer, clf)
    run.resources["model_vram_mb"] = 0.0
    block = {
        "name": BASELINE_NAME,
        "trained_in_distribution": True,
        "caveat": CAVEAT,
        "split": split,
        "training": train_stats,
        "model_memory_definition": "size of the pickled vectorizer + classifier (MB)",
        "software": {"scikit_learn": sklearn.__version__, "numpy": np.__version__, "python": platform.python_version()},
    }
    return run, block


def build_baseline_report(df_full, df_sample, sample_info, dataset_choice, seed=SAMPLE_SEED, sampler_factory=None,
                          **collectors) -> dict:
    """Evaluate the baseline and return a run report in the normal format."""
    from .run_report import build_run_report

    run, block = evaluate(df_full, df_sample, seed, sampler_factory)
    report = build_run_report(
        run, df_sample=df_sample, dataset_choice=dataset_choice, sample_info=sample_info,
        execution_mode="CPU (scikit-learn)",
        model={"name": BASELINE_NAME, "tag": None, "digest": None, "size_bytes": run.model_size_bytes,
               "family": "tfidf-logistic-regression", "parameter_size": None,
               "parameter_count": None, "quantization": None, "format": "pickle",
               "context_length": None, "modified_at": None},
        ollama=None, **collectors,
    )
    report["baseline"] = block
    report["prompt_variant"] = BASELINE_VARIANT
    return report


def main(argv: list[str] | None = None) -> int:
    from .data_loader import load_dataset_with_report
    from .sampling import balanced_sample
    from .utils import save_results

    parser = argparse.ArgumentParser(description="TF-IDF + logistic regression baseline on the evaluation sample.")
    parser.add_argument("--dataset", default="seed+sms")
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    args = parser.parse_args(argv)
    df, _ = load_dataset_with_report(args.dataset)
    sample = balanced_sample(df, args.size, seed=args.seed)
    report = build_baseline_report(df, sample.df, sample.info, args.dataset, seed=args.seed)
    path = save_results(report, BASELINE_NAME, prompt_variant=BASELINE_VARIANT)
    ue = report["detection"]["overall"]["unknown_as_error"]
    print(f"baseline F1 {ue['f1']} (n={report['detection']['n_records']}), "
          f"training {report['baseline']['training']['training_time_s']} s -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
