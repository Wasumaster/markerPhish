"""
MarkerPhish — Dataset Overview Component
=========================================
Dataset composition (one sentence, a stacked composition bar, a by-source table)
and a message browser for phishing, smishing, and legitimate samples.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import channel_for_source, to_binary_label
from ui.components import (
    cell,
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


def render_data_overview(df: pd.DataFrame) -> None:
    """
    Render the dataset overview and message browser component.

    Args:
        df: Unified DataFrame containing records with columns:
            id, text, true_label, source
    """
    page_title("Dataset")

    if df is None or df.empty:
        st.warning("No records loaded in dataset.")
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
    preview_df = filtered_df[["id", "source", "true_label", "text"]].copy()
    preview_df["text_preview"] = preview_df["text"].str.slice(0, 160).str.replace("\n", " ") + "..."

    if len(preview_df) <= _HTML_TABLE_ROW_LIMIT:
        data_table(
            ["id", "source", "label", "text"],
            [
                [
                    str(r.id),
                    r.source,
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
        table = preview_df[["id", "source", "true_label", "text_preview"]].rename(
            columns={"true_label": "label", "text_preview": "text"}
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
                ],
                headers=("field", "value"),
            )

            st.text_area("full message text", value=selected_row["text"], height=200, disabled=True)
        else:
            st.info("No records match the selected filter criteria.")

