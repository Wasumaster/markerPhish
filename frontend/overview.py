"""
MarkerPhish — Dataset Overview Component
=========================================
Visualises dataset composition, class distributions, and provides an interactive
message exploration browser for phishing, smishing, and legitimate samples.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from config import to_binary_label


def render_data_overview(df: pd.DataFrame) -> None:
    """
    Render the dataset overview and message browser component.

    Args:
        df: Unified DataFrame containing records with columns:
            id, text, true_label, source
    """
    st.markdown('<div class="section-title">Dataset Characteristics & Composition</div>', unsafe_allow_html=True)

    if df is None or df.empty:
        st.warning("No records loaded in dataset.")
        return

    # ──────────────────────────────────────────────
    # 1. Top KPI Summary Metrics
    # ──────────────────────────────────────────────
    c1, c2, c3 = st.columns([1, 1, 1.2])
    with c1:
        st.metric("Total Evaluated Records", f"{len(df):,}")
    with c2:
        malicious_count = len(df[df["true_label"].isin(["phishing", "smishing"])])
        malicious_pct = (malicious_count / len(df) * 100) if len(df) > 0 else 0
        st.metric("Malicious Threat Messages", f"{malicious_count:,}", f"{malicious_pct:.1f}% of total")
    with c3:
        valid_count = len(df[df["true_label"] == "valid"])
        valid_pct = (valid_count / len(df) * 100) if len(df) > 0 else 0
        st.metric("Safe Legitimate Messages", f"{valid_count:,}", f"{valid_pct:.1f}% of total")

    # ──────────────────────────────────────────────
    # 2. Asymmetric Visualisation & Provenance Table
    # ──────────────────────────────────────────────
    col_chart, col_meta = st.columns([1.4, 1.0])

    with col_chart:
        df_binary = df.copy()
        df_binary["binary_class"] = df_binary["true_label"].apply(to_binary_label)
        binary_counts = df_binary["binary_class"].value_counts().reset_index()
        binary_counts.columns = ["Classification", "Count"]

        color_map = {
            "malicious": "#ef4444",
            "safe": "#10b981",
        }

        fig = px.bar(
            binary_counts,
            x="Classification",
            y="Count",
            color="Classification",
            color_discrete_map=color_map,
            title="Binary Threat Distribution (Safe vs. Malicious)",
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#94a3b8"),
            showlegend=False,
            height=300,
            title_font_size=14,
            margin=dict(t=40, b=20, l=20, r=20),
            xaxis=dict(gridcolor="#1e293b", title=""),
            yaxis=dict(gridcolor="#1e293b", title="Record Count"),
        )
        st.plotly_chart(fig, width="stretch")

    with col_meta:
        st.markdown(
            '<div class="card-panel-header">Distribution Channel Breakdown</div>',
            unsafe_allow_html=True,
        )
        source_counts = df.groupby(["source", "true_label"]).size().unstack(fill_value=0)
        st.dataframe(source_counts, width="stretch")

        st.caption(
            "Evaluation mapping: Emails (PhishFuzzer) and SMS messages (Mendeley) "
            "are unified into binary classes: Malicious threats vs. Safe communications."
        )

    # ──────────────────────────────────────────────
    # 3. Interactive Message Browser & Inspector
    # ──────────────────────────────────────────────
    st.markdown('<div class="section-title">Message Browser & Class Inspector</div>', unsafe_allow_html=True)

    filt_col1, filt_col2 = st.columns([1, 2])
    with filt_col1:
        class_filter = st.selectbox(
            "Filter by Ground Truth Class",
            options=["All Classes", "Phishing (Email Threat)", "Smishing (SMS Threat)", "Valid (Safe Communication)"],
            index=0,
        )
    with filt_col2:
        search_query = st.text_input(
            "Search Message Text",
            value="",
            placeholder="Search keywords, domains, URLs, sender headers...",
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

    st.caption(f"Displaying {len(filtered_df):,} of {len(df):,} messages matching filter criteria.")

    # Show preview table
    preview_df = filtered_df[["id", "source", "true_label", "text"]].copy()
    preview_df["binary_threat"] = preview_df["true_label"].apply(to_binary_label)
    preview_df["text_preview"] = preview_df["text"].str.slice(0, 160).str.replace("\n", " ") + "..."

    st.dataframe(
        preview_df[["id", "source", "true_label", "binary_threat", "text_preview"]],
        width="stretch",
        height=260,
        hide_index=True,
    )

    # Inspect full record by ID
    with st.expander("Inspect Full Message Body", expanded=False):
        if not filtered_df.empty:
            available_ids = filtered_df["id"].tolist()
            selected_id = st.selectbox("Select Record ID to Inspect", options=available_ids, index=0)
            selected_row = filtered_df[filtered_df["id"] == selected_id].iloc[0]

            inf_col1, inf_col2, inf_col3 = st.columns(3)
            with inf_col1:
                st.markdown(f"**Channel Label:** `{selected_row['true_label']}`")
            with inf_col2:
                st.markdown(f"**Binary Category:** `{to_binary_label(selected_row['true_label'])}`")
            with inf_col3:
                st.markdown(f"**Data Source:** `{selected_row['source']}`")

            st.text_area("Full Message Text", value=selected_row["text"], height=200, disabled=True)
        else:
            st.info("No records match the selected filter criteria.")
