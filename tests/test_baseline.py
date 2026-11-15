"""Phase 8: TF-IDF + logistic regression baseline."""

import os

import pandas as pd
import pytest

from backend import baseline
from backend.optimum import analyze
from backend.profiler import ResourceSampler, TreeReader
from backend.sampling import balanced_sample
from backend.utils import ensure_report_compatibility, list_saved_result_files, load_result_by_filename, save_results


def _corpus(n_per_class=60, with_groups=False):
    rows, rid = [], 1
    vocab = {
        "phishing": "verify your account password urgent link suspended login now",
        "valid": "meeting agenda lunch tomorrow thanks report attached team",
        "smishing": "parcel fee pay now customs link prize claim",
    }
    for source, label in [("phishfuzzer_seed", "phishing"), ("phishfuzzer_seed", "valid"),
                          ("mendeley_sms", "smishing"), ("mendeley_sms", "valid")]:
        words = vocab[label].split()
        for i in range(n_per_class):
            text = " ".join(words[(i + k) % len(words)] for k in range(6)) + f" ref{i}"
            group = f"g{source}{label}{i // 3}" if with_groups else f"r{rid}"
            rows.append({"id": rid, "source_id": f"{source}:{rid}", "group_id": group, "text": text,
                         "true_label": label, "source": source})
            rid += 1
    return pd.DataFrame(rows)


def _sampler(name):
    return ResourceSampler(name, interval_s=0.01, tree=TreeReader(lambda: [os.getpid()]), ps_fetcher=lambda: None,
                           gpu_reader=lambda: (None, None), power_reader=lambda: True)


def test_training_split_excludes_sample_and_groups():
    df = _corpus(with_groups=True)
    sample = balanced_sample(df, 48, group_col="group_id").df  # groups of 3: 12 per cell
    train, split = baseline.training_split(df, sample)
    assert not set(train["source_id"]) & set(sample["source_id"])
    assert not set(train["group_id"]) & set(sample["group_id"])  # no rephrasing of a sampled text
    assert split["excluded_sampled"] == 48 and split["excluded_same_group"] == 0  # whole groups sampled
    assert split["training_records"] == len(df) - 48


def test_baseline_report_has_the_normal_run_format(results_dir):
    df = _corpus()
    sample = balanced_sample(df, 40)
    report = baseline.build_baseline_report(df, sample.df, sample.info, "seed+sms", sampler_factory=_sampler,
                                            code={"commit": None, "dirty": None, "changed_files": None})
    assert report["model_name"] == "baseline-tfidf-lr" and report["prompt_variant"] == "baseline"
    assert report["baseline"]["trained_in_distribution"] is True and "not a fair contest" in report["baseline"]["caveat"]
    assert report["manifest"]["prompt"]["prompt_template_hash"] is None
    assert report["manifest"]["dataset"]["dataset_fingerprint"]
    assert report["detection"]["overall"]["unknown_as_error"]["f1"] >= 0.9  # separable toy corpus
    assert all(0.0 <= r["score"] <= 1.0 for r in report["raw_predictions"])
    assert report["resources"]["model_size_mb"] > 0 and report["resources"]["start_mode"] is None
    assert report["baseline"]["training"]["training_time_s"] >= 0
    assert report["timing"]["latency_s"]["mean"] is not None
    name = os.path.basename(save_results(report, baseline.BASELINE_NAME, prompt_variant=baseline.BASELINE_VARIANT))
    loaded = ensure_report_compatibility(load_result_by_filename(name))
    assert name in list_saved_result_files() and loaded["latency_inflated"] is False


def test_baseline_is_excluded_from_the_optimum_analysis():
    df = _corpus()
    sample = balanced_sample(df, 40)
    report = baseline.build_baseline_report(df, sample.df, sample.info, "seed+sms", sampler_factory=_sampler,
                                            code={"commit": None, "dirty": None, "changed_files": None})
    out = analyze({"b": report}, {"p": {"ram_gb": 8}}, 0.5)
    assert out["points"] == {} and out["pareto"] == [] and out["reference_fingerprint"] is None
    assert out["per_profile"]["p"]["winner"] is None
    assert out["excluded"] == [{"file": "b", "reason": "non-LLM baseline (reference only, not recommended)"}]


def test_one_class_training_data_is_refused():
    df = _corpus().query("true_label == 'valid'")
    with pytest.raises(ValueError, match="both malicious and safe"):
        baseline.evaluate(df, df.head(20), sampler_factory=_sampler)


def test_real_data_no_sampled_text_in_training():
    from backend.data_loader import load_dataset_with_report

    df, _ = load_dataset_with_report("seed+sms")
    sample = balanced_sample(df, 100).df
    train, split = baseline.training_split(df, sample)
    assert split["training_records"] == len(df) - 100
    assert not set(train["text"]) & set(sample["text"])


def test_group_members_outside_the_sample_are_excluded():
    df = _corpus(with_groups=True)
    sample = df.head(1)  # one member of a 3-record group
    train, split = baseline.training_split(df, sample)
    assert split["excluded_same_group"] == 2
    assert not set(train["group_id"]) & set(sample["group_id"])
