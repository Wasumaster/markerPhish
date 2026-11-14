"""backend.sampling: balanced, channel-stratified, group-aware sampling."""

import pandas as pd
import pytest

from backend.sampling import allocate, balanced_sample, cell_capacity, sample_size_purpose


def _cells(sample: pd.DataFrame) -> dict:
    channel = sample["source"].map({"phishfuzzer_seed": "email", "mendeley_sms": "sms", "phishfuzzer_rephrased": "email"})
    return (channel + "|" + sample["true_label"]).value_counts().to_dict()


@pytest.mark.parametrize(
    "n, k, expected",
    [
        (1000, 4, [250, 250, 250, 250]),
        (250, 4, [63, 63, 62, 62]),
        (50, 4, [13, 13, 12, 12]),
        (51, 4, [13, 13, 13, 12]),
        (101, 2, [51, 50]),
        (7, 0, []),
    ],
)
def test_allocate_largest_remainder_in_fixed_order(n, k, expected):
    assert allocate(n, k) == expected


def test_cells_get_their_exact_share_and_classes_stay_balanced(toy_df):
    result = balanced_sample(toy_df, 50)
    assert len(result.df) == 50
    assert _cells(result.df) == {"email|phishing": 13, "email|valid": 13, "sms|smishing": 12, "sms|valid": 12}
    binary = result.df["true_label"].map({"phishing": "m", "smishing": "m", "valid": "s"}).value_counts()
    assert binary["m"] == binary["s"] == 25
    assert [c["count"] for c in result.info["cells"]] == [13, 13, 12, 12]
    assert result.info["fallback"] is False and result.info["fallback_reason"] is None


def test_is_deterministic_and_seed_dependent(toy_df):
    a = balanced_sample(toy_df, 40, seed=42).df
    b = balanced_sample(toy_df, 40, seed=42).df
    pd.testing.assert_frame_equal(a, b)
    c = balanced_sample(toy_df, 40, seed=7).df
    assert set(a["text"]) != set(c["text"])


def test_a_cell_sample_does_not_depend_on_other_cells(toy_df):
    sms_only = toy_df[toy_df["source"] == "mendeley_sms"]
    with_email = balanced_sample(toy_df, 40).df
    alone = balanced_sample(sms_only, 20).df  # 10 + 10, as in the 4-cell sample of 40
    pick = lambda d: set(d.loc[d["source"] == "mendeley_sms", "text"])  # noqa: E731
    assert pick(with_email) == pick(alone)


def test_ids_are_reindexed_and_source_columns_kept(toy_df):
    df = toy_df.assign(source_id=[f"src:{i}" for i in toy_df["id"]])
    s = balanced_sample(df, 24).df
    assert s["id"].tolist() == list(range(1, 25))
    assert s["source_id"].str.startswith("src:").all()
    assert s["source_id"].is_unique


def test_fallback_to_largest_balanced_size_is_reported(toy_df):
    # capacities 30 / 25 / 40 / 60 -> 101 is the largest size every cell can supply
    result = balanced_sample(toy_df, 120)
    info = result.info
    assert info["fallback"] is True
    assert info["requested_size"] == 120 and info["actual_size"] == 101 == len(result.df)
    assert "email|valid needs 30, has 25" in info["fallback_reason"]
    assert info["max_balanced_size"] == 101
    assert [c["count"] for c in info["cells"]] == [26, 25, 25, 25]


def test_single_channel_dataset_uses_two_cells(toy_df):
    sms = toy_df[toy_df["source"] == "mendeley_sms"]
    info = balanced_sample(sms, 31).info
    assert info["allocation_order"] == ["sms|smishing", "sms|valid"]
    assert [c["count"] for c in info["cells"]] == [16, 15]


def test_records_outside_known_cells_are_counted(toy_df):
    odd = pd.DataFrame([{"id": 999, "text": "x", "true_label": "phishing", "source": "unknown_src"}])
    info = balanced_sample(pd.concat([toy_df, odd], ignore_index=True), 40).info
    assert info["records_outside_cells"] == 1


def test_sizes_below_minimum_and_empty_data_are_rejected(toy_df):
    with pytest.raises(ValueError, match="at least 20"):
        balanced_sample(toy_df, 19)
    with pytest.raises(ValueError, match="no records"):
        balanced_sample(toy_df.iloc[0:0], 20)


# ── group-aware sampling ──
def _grouped_df(group_sizes_per_cell: dict) -> pd.DataFrame:
    rows, rid = [], 1
    for (source, label), sizes in group_sizes_per_cell.items():
        for g, size in enumerate(sizes):
            for j in range(size):
                rows.append(
                    {"id": rid, "text": f"{source}-{label}-g{g}-{j}", "true_label": label,
                     "source": source, "group_id": f"{source}-{label}-g{g}"}
                )
                rid += 1
    return pd.DataFrame(rows)


def test_groups_are_never_split():
    df = _grouped_df(
        {
            ("phishfuzzer_rephrased", "phishing"): [3] * 10 + [2] * 5,
            ("phishfuzzer_rephrased", "valid"): [3] * 10 + [2] * 5,
        }
    )
    result = balanced_sample(df, 40)
    assert result.info["group_aware"] is True
    assert [c["count"] for c in result.info["cells"]] == [20, 20]
    full_sizes = df.groupby("group_id").size()
    sampled_sizes = result.df.groupby("group_id").size()
    assert (sampled_sizes == full_sizes[sampled_sizes.index]).all()
    assert result.info["n_groups"] == result.df["group_id"].nunique()


def test_unreachable_exact_target_falls_back():
    # only groups of 6: a cell can hold 0, 6, 12, ... records -> 50 requested -> 48
    df = _grouped_df(
        {
            ("phishfuzzer_rephrased", "phishing"): [6] * 20,
            ("phishfuzzer_rephrased", "valid"): [6] * 20,
        }
    )
    result = balanced_sample(df, 50)
    assert result.info["fallback"] is True
    assert result.info["actual_size"] == 48
    assert "email|phishing needs 25, but no combination of whole groups (sizes 6)" in result.info["fallback_reason"]


def test_cell_capacity_reports_records_groups_and_max_size(toy_df):
    cap = cell_capacity(toy_df)
    assert [(c["channel"], c["label"], c["records"]) for c in cap["cells"]] == [
        ("email", "phishing", 30), ("email", "valid", 25), ("sms", "smishing", 40), ("sms", "valid", 60)
    ]
    assert cap["group_aware"] is False
    assert cap["max_balanced_size"] == 101


@pytest.mark.parametrize("n, purpose", [(50, "smoke"), (100, "smoke"), (20, "smoke"), (250, "pilot"), (499, "pilot"), (500, "results"), (1000, "results")])
def test_sample_size_purpose(n, purpose):
    assert sample_size_purpose(n) == purpose
