import streamlit as st
import os
import pandas as pd
import plotly.express as px
from analysis_utils import process_logs, process_custom_lexicon

def render_basic_analysis(DATA_DIR):
    st.header("Basic Data Analysis")
    if st.button("Refresh Data", key="refresh_basic"):
        st.rerun()

    jsonl_path = os.path.join(DATA_DIR, "experiment_log.jsonl")
    if os.path.exists(jsonl_path):
        df = pd.read_json(jsonl_path, lines=True)
        st.dataframe(df)

        st.subheader("Metrics Overview")
        col1, col2 = st.columns(2)
        with col1:
            avg_latency = df.groupby("speaker_model")["latency_ms"].mean().reset_index()
            fig_latency = px.bar(avg_latency, x="speaker_model", y="latency_ms", title="Average Latency (ms)")
            st.plotly_chart(fig_latency, use_container_width=True)
        with col2:
            avg_tokens = df.groupby("speaker_model")["output_tokens"].mean().reset_index()
            fig_tokens = px.bar(avg_tokens, x="speaker_model", y="output_tokens", title="Average Output Tokens")
            st.plotly_chart(fig_tokens, use_container_width=True)
    else:
        st.info("No data found.")

def render_stylometric_analysis(DATA_DIR):
    st.header("Stylometric Analysis (NLTK)")
    st.subheader("Custom Word Frequency (LIWC-style)")
    st.markdown("Define categories and words to count. Format: `Category: word1, word2`")

    default_lexicon = """Positive: love, great, happy, good
Negative: hate, bad, sad, terrible
Hesitation: um, uh, er, maybe, perhaps"""

    lexicon_input = st.text_area("Define Custom Lexicon", default_lexicon, height=100)

    category_dict = {}
    if lexicon_input:
        for line in lexicon_input.split("\n"):
            if ":" in line:
                cat, words = line.split(":", 1)
                category_dict[cat.strip()] = [w.strip() for w in words.split(",")]

    if st.button("Run Analysis", type="primary"):
        jsonl_path = os.path.join(DATA_DIR, "experiment_log.jsonl")
        if os.path.exists(jsonl_path):
            with st.spinner("Processing text..."):
                df = pd.read_json(jsonl_path, lines=True)
                analyzed_df = process_logs(df)
                if category_dict:
                    analyzed_df = process_custom_lexicon(analyzed_df, category_dict)

                st.success("Analysis Complete!")
                st.dataframe(analyzed_df)

                csv = analyzed_df.to_csv(index=False).encode("utf-8")
                st.download_button(label="Download Analysis as CSV", data=csv, file_name="stylometric_analysis.csv", mime="text/csv")

                st.subheader("Linguistic Patterns")
                pos_cols = ["noun_ratio", "verb_ratio", "adj_ratio", "adv_ratio"]
                avg_pos = analyzed_df.groupby("speaker_model")[pos_cols].mean().reset_index()
                melted_pos = avg_pos.melt(id_vars="speaker_model", var_name="POS Type", value_name="Ratio")
                fig_pos = px.bar(melted_pos, x="POS Type", y="Ratio", color="speaker_model", barmode="group", title="POS Distribution (Grouped by Category)")
                st.plotly_chart(fig_pos, use_container_width=True)

                if category_dict:
                    st.subheader("Custom Category Frequencies")
                    lex_cols = list(category_dict.keys())
                    avg_lex = analyzed_df.groupby("speaker_model")[lex_cols].mean().reset_index()
                    melted_lex = avg_lex.melt(id_vars="speaker_model", var_name="Category", value_name="Avg Count")
                    fig_lex = px.bar(melted_lex, x="Category", y="Avg Count", color="speaker_model", barmode="group", title="Custom Word Usage (Grouped by Category)")
                    st.plotly_chart(fig_lex, use_container_width=True)
        else:
            st.warning("No data found.")
