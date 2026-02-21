"""Rendering helpers for analysis tabs."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd
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
    avg_latency_by_model, avg_tokens_by_model = _compute_basic_metrics(all_logs)
    _render_basic_metric_charts(avg_latency_by_model, avg_tokens_by_model)


def _compute_basic_metrics(all_logs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute average latency and token usage per speaker model."""
    avg_latency_by_model = all_logs.groupby("speaker_model")["latency_ms"].mean().reset_index()
    avg_tokens_by_model = all_logs.groupby("speaker_model")["output_tokens"].mean().reset_index()
    return avg_latency_by_model, avg_tokens_by_model


def _render_basic_metric_charts(avg_latency_by_model: pd.DataFrame, avg_tokens_by_model: pd.DataFrame) -> None:
    """Render basic aggregate metric charts from precomputed data."""
    latency_column, tokens_column = st.columns(2)

    with latency_column:
        latency_chart = px.bar(
            avg_latency_by_model,
            x="speaker_model",
            y="latency_ms",
            title="Average Latency (ms)",
        )
        st.plotly_chart(latency_chart, use_container_width=True)

    with tokens_column:
        tokens_chart = px.bar(
            avg_tokens_by_model,
            x="speaker_model",
            y="output_tokens",
            title="Average Output Tokens",
        )
        st.plotly_chart(tokens_chart, use_container_width=True)


def render_stylometric_analysis_tab() -> None:
    """Render custom lexicon editor and stylometric analysis output."""
    st.header("Linguistic Analysis")
    st.subheader("Custom Lexicon Configuration")
    st.markdown("Define specific word categories to track during the conversation.")

    _initialize_lexicon_state()
    _render_lexicon_editor()
    category_dict = _build_category_dict(st.session_state["custom_lexicon"])

    st.markdown("---")
    if not st.button("Run Analysis", type="primary", width="stretch"):
        return

    all_logs = st.session_state["all_logs"]
    if all_logs.empty:
        st.warning("No data found.")
        return

    analyzed_df = _run_stylometric_analysis(all_logs, category_dict)

    st.success("Analysis complete.")
    st.dataframe(analyzed_df)

    _render_analysis_download_button(analyzed_df)

    _render_pos_chart(analyzed_df)
    if category_dict:
        _render_custom_lexicon_chart(analyzed_df, category_dict)


def _run_stylometric_analysis(
    all_logs: pd.DataFrame,
    category_dict: Dict[str, List[str]],
) -> pd.DataFrame:
    """Process log text into stylometric metrics and optional custom lexicon counts."""
    with st.spinner("Processing text..."):
        analyzed_df = process_logs(all_logs)
        if category_dict:
            analyzed_df = process_custom_lexicon(analyzed_df, category_dict)

    return analyzed_df


def _render_analysis_download_button(analyzed_df: pd.DataFrame) -> None:
    """Render CSV download action for analyzed dataframe output."""
    csv_data = analyzed_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Analysis as CSV",
        data=csv_data,
        file_name="stylometric_analysis.csv",
        mime="text/csv",
    )


def _initialize_lexicon_state() -> None:
    """Populate the default custom lexicon the first time the tab is opened."""
    if "custom_lexicon" in st.session_state:
        return

    st.session_state["custom_lexicon"] = [
        {"category": "Positive", "words": "love, great, happy, good"},
        {"category": "Negative", "words": "hate, bad, sad, terrible"},
        {"category": "Hesitation", "words": "um, uh, er, maybe, perhaps"},
    ]


def _render_lexicon_editor() -> None:
    """Render editable category rows and controls for the custom lexicon."""
    for index, item in enumerate(st.session_state["custom_lexicon"]):
        category_column, words_column, delete_column = st.columns([1, 2, 0.2])

        with category_column:
            item["category"] = st.text_input(
                f"Category Name {index}",
                item["category"],
                key=f"lex_cat_{index}",
                placeholder="Category",
                label_visibility="collapsed",
            )

        with words_column:
            item["words"] = st.text_input(
                f"Words {index}",
                item["words"],
                key=f"lex_words_{index}",
                placeholder="words, separated, by, commas",
                label_visibility="collapsed",
            )

        with delete_column:
            if st.button("Delete", key=f"lex_del_{index}", help="Remove category"):
                st.session_state["custom_lexicon"].pop(index)
                st.rerun()

    if st.button("Add New Category"):
        st.session_state["custom_lexicon"].append({"category": "", "words": ""})
        st.rerun()


def _build_category_dict(custom_lexicon: List[Dict[str, str]]) -> Dict[str, List[str]]:
    """Convert lexicon rows into a normalized dictionary of category to words."""
    category_dict: Dict[str, List[str]] = {}

    for item in custom_lexicon:
        category_name = item["category"].strip()
        if not category_name:
            continue

        words: List[str] = []
        for raw_word in item["words"].split(","):
            normalized_word = raw_word.strip()
            if normalized_word:
                words.append(normalized_word)

        category_dict[category_name] = words

    return category_dict


def _render_pos_chart(analyzed_df: pd.DataFrame) -> None:
    """Render grouped POS-ratio bars per speaker model."""
    st.subheader("Linguistic Patterns")
    pos_columns = ["noun_ratio", "verb_ratio", "adj_ratio", "adv_ratio"]
    avg_pos_by_model = analyzed_df.groupby("speaker_model")[pos_columns].mean().reset_index()
    melted_pos = avg_pos_by_model.melt(id_vars="speaker_model", var_name="POS Type", value_name="Ratio")
    pos_chart = px.bar(
        melted_pos,
        x="POS Type",
        y="Ratio",
        color="speaker_model",
        barmode="group",
        title="POS Distribution (Grouped by Category)",
    )
    st.plotly_chart(pos_chart, use_container_width=True)


def _render_custom_lexicon_chart(analyzed_df: pd.DataFrame, category_dict: Dict[str, List[str]]) -> None:
    """Render grouped custom-lexicon bars per speaker model."""
    st.subheader("Custom Category Frequencies")
    lexicon_columns = list(category_dict.keys())
    avg_lexicon_by_model = analyzed_df.groupby("speaker_model")[lexicon_columns].mean().reset_index()
    melted_lexicon = avg_lexicon_by_model.melt(
        id_vars="speaker_model",
        var_name="Category",
        value_name="Avg Count",
    )
    lexicon_chart = px.bar(
        melted_lexicon,
        x="Category",
        y="Avg Count",
        color="speaker_model",
        barmode="group",
        title="Custom Word Usage (Grouped by Category)",
    )
    st.plotly_chart(lexicon_chart, use_container_width=True)
