"""Rendering helpers for analysis tabs."""

from __future__ import annotations

from typing import Dict, List

import plotly.express as px
import streamlit as st

from parrotlm.analysis_utils import process_custom_lexicon, process_logs


def render_basic_analysis_tab() -> None:
    """Render raw dataframe and aggregate latency/token charts."""
    st.header("Basic Data Analysis")
    if st.button("Refresh Data", key="refresh_basic"):
        st.rerun()

    all_logs = st.session_state["all_logs"]
    if all_logs.empty:
        st.info("No data found.")
        return

    st.dataframe(all_logs)

    st.subheader("Metrics Overview")
    col1, col2 = st.columns(2)
    with col1:
        avg_latency = all_logs.groupby("speaker_model")["latency_ms"].mean().reset_index()
        fig_latency = px.bar(avg_latency, x="speaker_model", y="latency_ms", title="Average Latency (ms)")
        st.plotly_chart(fig_latency, use_container_width=True)
    with col2:
        avg_tokens = all_logs.groupby("speaker_model")["output_tokens"].mean().reset_index()
        fig_tokens = px.bar(avg_tokens, x="speaker_model", y="output_tokens", title="Average Output Tokens")
        st.plotly_chart(fig_tokens, use_container_width=True)


def render_stylometric_analysis_tab() -> None:
    """Render custom lexicon editor and stylometric analysis output."""
    st.header("🧠 Stylometric Analysis (NLTK)")
    st.subheader("🏷️ Custom Lexicon Configuration")
    st.markdown("Define specific word categories to track during the conversation.")

    _initialize_lexicon_state()
    _render_lexicon_editor()
    category_dict = _build_category_dict(st.session_state["custom_lexicon"])

    st.markdown("---")
    if not st.button("🚀 Run Analysis", type="primary", width="stretch"):
        return

    all_logs = st.session_state["all_logs"]
    if all_logs.empty:
        st.warning("No data found.")
        return

    with st.spinner("Processing text..."):
        analyzed_df = process_logs(all_logs)
        if category_dict:
            analyzed_df = process_custom_lexicon(analyzed_df, category_dict)

    st.success("Analysis Complete!")
    st.dataframe(analyzed_df)
    csv_data = analyzed_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Analysis as CSV",
        data=csv_data,
        file_name="stylometric_analysis.csv",
        mime="text/csv",
    )

    _render_pos_chart(analyzed_df)
    if category_dict:
        _render_custom_lexicon_chart(analyzed_df, category_dict)


def _initialize_lexicon_state() -> None:
    if "custom_lexicon" in st.session_state:
        return

    st.session_state["custom_lexicon"] = [
        {"category": "Positive", "words": "love, great, happy, good"},
        {"category": "Negative", "words": "hate, bad, sad, terrible"},
        {"category": "Hesitation", "words": "um, uh, er, maybe, perhaps"},
    ]


def _render_lexicon_editor() -> None:
    for index, item in enumerate(st.session_state["custom_lexicon"]):
        col1, col2, col3 = st.columns([1, 2, 0.2])
        with col1:
            item["category"] = st.text_input(
                f"Category Name {index}",
                item["category"],
                key=f"lex_cat_{index}",
                placeholder="Category",
                label_visibility="collapsed",
            )
        with col2:
            item["words"] = st.text_input(
                f"Words {index}",
                item["words"],
                key=f"lex_words_{index}",
                placeholder="words, separated, by, commas",
                label_visibility="collapsed",
            )
        with col3:
            if st.button("🗑️", key=f"lex_del_{index}", help="Remove category"):
                st.session_state["custom_lexicon"].pop(index)
                st.rerun()

    if st.button("➕ Add New Category"):
        st.session_state["custom_lexicon"].append({"category": "", "words": ""})
        st.rerun()


def _build_category_dict(custom_lexicon: List[Dict[str, str]]) -> Dict[str, List[str]]:
    return {
        item["category"].strip(): [word.strip() for word in item["words"].split(",") if word.strip()]
        for item in custom_lexicon
        if item["category"].strip()
    }


def _render_pos_chart(analyzed_df) -> None:
    st.subheader("Linguistic Patterns")
    pos_columns = ["noun_ratio", "verb_ratio", "adj_ratio", "adv_ratio"]
    avg_pos = analyzed_df.groupby("speaker_model")[pos_columns].mean().reset_index()
    melted_pos = avg_pos.melt(id_vars="speaker_model", var_name="POS Type", value_name="Ratio")
    fig_pos = px.bar(
        melted_pos,
        x="POS Type",
        y="Ratio",
        color="speaker_model",
        barmode="group",
        title="POS Distribution (Grouped by Category)",
    )
    st.plotly_chart(fig_pos, use_container_width=True)


def _render_custom_lexicon_chart(analyzed_df, category_dict: Dict[str, List[str]]) -> None:
    st.subheader("Custom Category Frequencies")
    lexicon_columns = list(category_dict.keys())
    avg_lexicon = analyzed_df.groupby("speaker_model")[lexicon_columns].mean().reset_index()
    melted_lexicon = avg_lexicon.melt(id_vars="speaker_model", var_name="Category", value_name="Avg Count")
    fig_lexicon = px.bar(
        melted_lexicon,
        x="Category",
        y="Avg Count",
        color="speaker_model",
        barmode="group",
        title="Custom Word Usage (Grouped by Category)",
    )
    st.plotly_chart(fig_lexicon, use_container_width=True)

