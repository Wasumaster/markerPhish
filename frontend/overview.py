"""
MarkerPhish — Dataset Overview Component
=========================================
Dataset composition (one sentence, a stacked composition bar, a by-source table)
and a message browser for phishing, smishing, and legitimate samples.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import SPAM_EXCLUSION_RATIONALE, channel_for_source, to_binary_label
from ui.components import (
    cell,
    esc,
    colored,
    data_table,
    html_block,
    kv_table,
    note,
    page_title,
    section_label,
    section_title,
    stacked_bar,
    strong,
)
from ui.theme import CLASS_COLORS, MALICIOUS, SAFE

# Above this many rows the message list falls back to an interactive dataframe,
# so very large selections stay fully browsable without a huge HTML payload.
_HTML_TABLE_ROW_LIMIT = 500

_CLASS_FILTER_LABELS = {
    "All Classes": "all",
    "Phishing (Email Threat)": "phishing",
    "Smishing (SMS Threat)": "smishing",
    "Valid (Safe Communication)": "valid",
}


def _label_color(label: str) -> str:
    return MALICIOUS if to_binary_label(label) == "malicious" else SAFE


def _n(value) -> str:
    return f"{int(value):,}"


def _render_sample_section(sample_info: dict) -> None:
    """Per-cell allocation of the current sample (balanced, channel-stratified)."""
    cells = sample_info["cells"]
    rows = [
        [c["channel"], c["label"], _n(c["target"]), _n(c["count"]), _n(c["groups_in_sample"]), _n(c["capacity_records"])]
        for c in cells
    ]
    total = [
        "all",
        "",
        _n(sum(c["target"] for c in cells)),
        _n(sample_info["actual_size"]),
        _n(sample_info["n_groups"]),
        _n(sum(c["capacity_records"] for c in cells)),
    ]
    with st.container():
        section_label("sample cells")
        data_table(
            ["channel", "label", "target", "in sample", "groups", "available"],
            rows,
            numeric={2, 3, 4, 5},
            total=total,
            max_width="640px",
        )
        order = ", ".join(k.replace("|", " ") for k in sample_info["allocation_order"])
        grouping = (
            "Group-aware: a PhishFuzzer seed email and its rephrasings are all in the sample or all out."
            if sample_info["group_aware"]
            else "Every group has one record here, so group-aware sampling changes nothing."
        )
        note(
            f"Seed {sample_info['sample_seed']}. When the size is not divisible by the number of cells, "
            f"the extra records go to the cells in this order: {order}. {grouping}"
        )


def _render_preprocessing_report(report: dict) -> None:
    """What preprocessing did per source: spam removal, deduplication, truncation, capacity."""
    section_title(
        "Preprocessing",
        f"Preprocessing version {report['preprocessing_version']}. Counts cover the whole dataset choice, "
        "before sampling.",
    )
    sources = report["sources"]

    with st.container():
        section_label("records by step")
        data_table(
            ["source", "raw", "spam removed", "unrecognized", "empty", "exact dupes",
             "normalized dupes", "label conflicts", "final"],
            [
                [
                    s["source"],
                    _n(s["raw_records"]),
                    _n(s["excluded_spam"]),
                    _n(s["unrecognized_label"]),
                    _n(s["empty"]),
                    _n(s["dedup_exact_duplicates"]),
                    _n(s["dedup_normalized_duplicates"]),
                    _n(s["dedup_label_conflict_records"]),
                    _n(s["final"]),
                ]
                for s in sources
            ],
            numeric=set(range(1, 9)),
        )
        note(
            "Duplicates are removed within each source, on the full model input before truncation: exact "
            "copies, then copies that are equal after lowercasing, collapsing whitespace and stripping "
            "punctuation at the ends. If copies of one text carry different labels, all of them are dropped "
            "(label conflicts)."
        )

    with st.container():
        section_label("truncation")
        data_table(
            ["source", "records", "truncated", "share", "median chars removed", "max chars removed",
             "links moved", "records with moved links", "longest input"],
            [
                [
                    s["source"],
                    _n(t["records"]),
                    _n(t["truncated"]),
                    f"{t['truncated_pct']:.1%}",
                    _n(t["chars_removed_median"]),
                    _n(t["chars_removed_max"]),
                    _n(t["links_moved_total"]),
                    _n(t["records_with_links_moved"]),
                    _n(t["max_input_chars"]),
                ]
                for s in sources
                for t in [s["truncation"]]
            ],
            numeric=set(range(1, 9)),
        )
        note(
            "Email input: From, Subject and Links header lines, then the body. Header plus body is limited to "
            f"{report['message_max_chars']:,} characters, so the longest prompt plus the output fits the context "
            "window. The header is never cut, only the body. Links that appear only in the cut-off part of the "
            "body are moved into the Links header, so truncation removes no link."
        )

    with st.container():
        section_label("available per cell")
        data_table(
            ["channel", "label", "unique records", "groups"],
            [[c["channel"], c["label"], _n(c["records"]), _n(c["groups"])] for c in report["cells"]],
            numeric={2, 3},
            max_width="640px",
        )
        note(f"Largest balanced sample for this dataset choice: {report['max_balanced_size']:,} records.")

    with st.expander("why spam is excluded", expanded=False):
        st.markdown(f'<p class="mp-desc">{esc(SPAM_EXCLUSION_RATIONALE)}</p>', unsafe_allow_html=True)


def render_data_overview(
    df: pd.DataFrame | None,
    dataset_report: dict | None = None,
    sample_info: dict | None = None,
) -> None:
    """
    Render the sample overview, the preprocessing report and the message browser.

    Args:
        df: the current sample (unified schema: id, source_id, text, true_label, source, ...)
        dataset_report: report of backend.data_loader.load_dataset_with_report
        sample_info: info of backend.sampling.balanced_sample
    """
    page_title("Dataset")

    if sample_info and sample_info.get("fallback"):
        st.warning(sample_info["fallback_reason"])

    if df is None or df.empty:
        st.warning("No records loaded in dataset.")
        if dataset_report:
            _render_preprocessing_report(dataset_report)
        return

    # ──────────────────────────────────────────────
    # 1. Summary sentence
    # ──────────────────────────────────────────────
    binary = df["true_label"].apply(to_binary_label)
    malicious_count = int((binary == "malicious").sum())
    valid_count = int((binary == "safe").sum())
    split = ", split evenly:" if malicious_count == valid_count else ":"
    html_block(
        '<p class="mp-sentence">'
        f"{strong(f'{len(df):,}')} messages{split} "
        f"{colored(f'{malicious_count:,} malicious', MALICIOUS)} and "
        f"{colored(f'{valid_count:,} safe', SAFE)}.</p>"
        '<p class="mp-desc">Emails from PhishFuzzer and SMS from the Mendeley corpus are mapped onto one binary label.</p>'
    )

    # ──────────────────────────────────────────────
    # 2. Composition bar (smishing, phishing, valid sms, valid email)
    # ──────────────────────────────────────────────
    channels = [channel_for_source(s, t) for s, t in zip(df["source"], df["true_label"])]
    labels = df["true_label"].tolist()
    smishing = sum(1 for lbl in labels if lbl == "smishing")
    phishing = sum(1 for lbl in labels if lbl == "phishing")
    valid_sms = sum(1 for lbl, ch in zip(labels, channels) if lbl == "valid" and ch == "sms")
    valid_email = sum(1 for lbl, ch in zip(labels, channels) if lbl == "valid" and ch != "sms")

    with st.container():
        section_label("composition")
        stacked_bar(
            [
                ("smishing", smishing, CLASS_COLORS["smishing"]),
                ("phishing", phishing, CLASS_COLORS["phishing"]),
                ("valid sms", valid_sms, CLASS_COLORS["valid sms"]),
                ("valid email", valid_email, CLASS_COLORS["valid email"]),
            ]
        )

    # ──────────────────────────────────────────────
    # 3. By-source table
    # ──────────────────────────────────────────────
    source_counts = df.groupby(["source", "true_label"]).size().unstack(fill_value=0)
    rows = []
    totals = {"phishing": 0, "smishing": 0, "valid": 0}
    for source, counts in source_counts.iterrows():
        values = {k: int(counts.get(k, 0)) for k in totals}
        for k, v in values.items():
            totals[k] += v
        rows.append(
            [
                source,
                channel_for_source(source),
                f"{values['phishing']:,}",
                f"{values['smishing']:,}",
                f"{values['valid']:,}",
                f"{sum(values.values()):,}",
            ]
        )
    total_row = [
        "all",
        "",
        f"{totals['phishing']:,}",
        f"{totals['smishing']:,}",
        f"{totals['valid']:,}",
        f"{sum(totals.values()):,}",
    ]
    with st.container():
        section_label("by source")
        data_table(
            ["source", "channel", "phishing", "smishing", "valid", "total"],
            rows,
            numeric={2, 3, 4, 5},
            total=total_row,
            max_width="640px",
        )

    if sample_info:
        _render_sample_section(sample_info)
    if dataset_report:
        _render_preprocessing_report(dataset_report)

    # ──────────────────────────────────────────────
    # 4. Message browser & inspector
    # ──────────────────────────────────────────────
    section_title("Messages")

    filt_col1, filt_col2, filt_col3 = st.columns([2, 6, 1.2], vertical_alignment="bottom")
    with filt_col1:
        class_filter = st.selectbox(
            "class",
            options=list(_CLASS_FILTER_LABELS.keys()),
            index=0,
            format_func=lambda opt: _CLASS_FILTER_LABELS[opt],
        )
    with filt_col2:
        search_query = st.text_input(
            "search text, domains, URLs, headers",
            value="",
            placeholder="",
        )

    # Apply filters
    filtered_df = df.copy()
    if class_filter == "Phishing (Email Threat)":
        filtered_df = filtered_df[filtered_df["true_label"] == "phishing"]
    elif class_filter == "Smishing (SMS Threat)":
        filtered_df = filtered_df[filtered_df["true_label"] == "smishing"]
    elif class_filter == "Valid (Safe Communication)":
        filtered_df = filtered_df[filtered_df["true_label"] == "valid"]

    if search_query.strip():
        filtered_df = filtered_df[
            filtered_df["text"].str.contains(search_query.strip(), case=False, na=False)
        ]

    with filt_col3:
        html_block(
            f'<p class="mp-sidebar-line" style="text-align:right;margin:0 0 12px 0;">'
            f"{len(filtered_df):,} / {len(df):,}</p>"
        )

    # Message table
    preview_df = filtered_df.copy()
    if "source_id" not in preview_df:
        preview_df["source_id"] = preview_df["source"]
    preview_df = preview_df[["id", "source_id", "true_label", "text"]]
    preview_df["text_preview"] = preview_df["text"].str.slice(0, 160).str.replace("\n", " ") + "..."

    if len(preview_df) <= _HTML_TABLE_ROW_LIMIT:
        data_table(
            ["id", "source id", "label", "text"],
            [
                [
                    str(r.id),
                    r.source_id,
                    cell(r.true_label, color=_label_color(r.true_label), bold=True),
                    r.text_preview,
                ]
                for r in preview_df.itertuples(index=False)
            ],
            numeric={0},
            sans={3},
            max_height="420px",
        )
    else:
        table = preview_df[["id", "source_id", "true_label", "text_preview"]].rename(
            columns={"source_id": "source id", "true_label": "label", "text_preview": "text"}
        )
        st.dataframe(
            table.style.map(lambda v: f"color: {_label_color(v)}; font-weight: 600;", subset=["label"]),
            width="stretch",
            height=420,
            hide_index=True,
        )
        note(f"{len(preview_df):,} rows: interactive table (sortable, scrollable).")

    # Inspect full record by ID
    with st.expander("inspect message", expanded=False):
        if not filtered_df.empty:
            available_ids = filtered_df["id"].tolist()
            selected_id = st.selectbox("record id", options=available_ids, index=0)
            selected_row = filtered_df[filtered_df["id"] == selected_id].iloc[0]

            kv_table(
                [
                    ("channel label", cell(selected_row["true_label"], color=_label_color(selected_row["true_label"]), bold=True)),
                    ("binary category", to_binary_label(selected_row["true_label"])),
                    ("data source", selected_row["source"]),
                    ("source id", selected_row.get("source_id", "")),
                    ("group id", selected_row.get("group_id", "")),
                    ("truncated", "yes" if selected_row.get("truncated") else "no"),
                ],
                headers=("field", "value"),
            )

            st.text_area("full message text", value=selected_row["text"], height=200, disabled=True)
        else:
            st.info("No records match the selected filter criteria.")

