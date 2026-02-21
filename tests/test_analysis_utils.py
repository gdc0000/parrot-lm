from unittest.mock import patch

import pandas as pd
import pytest

from parrotlm import analysis_utils


def test_count_custom_words_counts_per_category():
    text = "Love is good and love stays good"
    category_dict = {"Positive": ["love", "good"], "Negative": ["bad"]}

    counts = analysis_utils.count_custom_words(text, category_dict)

    assert counts["Positive"] == 4
    assert counts["Negative"] == 0


def test_count_custom_words_ignores_basic_punctuation():
    text = "Love, love! good..."
    category_dict = {"Positive": ["love", "good"]}

    counts = analysis_utils.count_custom_words(text, category_dict)

    assert counts["Positive"] == 3


def test_process_logs_adds_metric_columns():
    df = pd.DataFrame({"content": ["hello world"]})
    fake_metrics = {
        "token_count": 2,
        "sentence_count": 1,
        "avg_sentence_length": 2.0,
        "noun_ratio": 0.5,
        "verb_ratio": 0.0,
        "adj_ratio": 0.0,
        "adv_ratio": 0.0,
        "pron_ratio": 0.0,
    }

    with patch.object(analysis_utils, "analyze_text", return_value=fake_metrics):
        result = analysis_utils.process_logs(df)

    assert "token_count" in result.columns
    assert "noun_ratio" in result.columns
    assert result.loc[0, "token_count"] == 2


def test_process_logs_falls_back_to_zero_metrics_on_error():
    df = pd.DataFrame({"content": ["hello world"]})

    with patch.object(analysis_utils, "analyze_text", side_effect=RuntimeError("boom")):
        result = analysis_utils.process_logs(df)

    assert result.loc[0, "token_count"] == 0
    assert result.loc[0, "sentence_count"] == 0


def test_process_logs_requires_dataframe_input():
    with pytest.raises(TypeError):
        analysis_utils.process_logs(["not-a-dataframe"])


def test_process_custom_lexicon_requires_dataframe_input():
    with pytest.raises(TypeError):
        analysis_utils.process_custom_lexicon(["not-a-dataframe"], {"x": ["y"]})


def test_process_logs_returns_copy_when_content_column_missing():
    df = pd.DataFrame({"speaker_model": ["a"]})
    result = analysis_utils.process_logs(df)

    assert id(result) != id(df)
    assert list(result.columns) == ["speaker_model"]


def test_process_custom_lexicon_adds_category_columns():
    df = pd.DataFrame({"content": ["love bad love"]})
    category_dict = {"Positive": ["love"], "Negative": ["bad"]}

    result = analysis_utils.process_custom_lexicon(df, category_dict)

    assert result.loc[0, "Positive"] == 2
    assert result.loc[0, "Negative"] == 1


def test_process_custom_lexicon_returns_copy_when_category_dict_is_empty():
    df = pd.DataFrame({"content": ["hello"]})
    result = analysis_utils.process_custom_lexicon(df, {})

    assert id(result) != id(df)
    assert list(result.columns) == ["content"]


def test_ensure_nltk_resources_raises_with_missing_packages():
    with patch.object(analysis_utils.nltk.data, "find", side_effect=LookupError):
        with pytest.raises(RuntimeError) as raised:
            analysis_utils.ensure_nltk_resources(download_missing=False)

    assert "Missing NLTK resources" in str(raised.value)

