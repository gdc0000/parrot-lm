import pandas as pd
import pytest

from parrotlm.ui import analysis_tabs


def test_build_category_dict_ignores_blank_rows_and_words():
    custom_lexicon = [
        {"category": "Positive", "words": "love, , great"},
        {"category": "  ", "words": "ignored"},
        {"category": "Negative", "words": "bad, sad"},
    ]

    category_dict = analysis_tabs._build_category_dict(custom_lexicon)

    assert category_dict["Positive"] == ["love", "great"]
    assert category_dict["Negative"] == ["bad", "sad"]
    assert "  " not in category_dict


def test_compute_basic_metrics_returns_expected_averages():
    all_logs = pd.DataFrame(
        [
            {"speaker_model": "A", "latency_ms": 100, "output_tokens": 10},
            {"speaker_model": "A", "latency_ms": 300, "output_tokens": 20},
            {"speaker_model": "B", "latency_ms": 200, "output_tokens": 30},
        ]
    )

    avg_latency, avg_tokens = analysis_tabs._compute_basic_metrics(all_logs)

    assert avg_latency.loc[avg_latency["speaker_model"] == "A", "latency_ms"].iloc[0] == 200
    assert avg_tokens.loc[avg_tokens["speaker_model"] == "A", "output_tokens"].iloc[0] == 15
    assert avg_tokens.loc[avg_tokens["speaker_model"] == "B", "output_tokens"].iloc[0] == 30


def test_compute_basic_metrics_raises_when_required_columns_are_missing():
    all_logs = pd.DataFrame([{"speaker_model": "A"}])

    with pytest.raises(KeyError):
        analysis_tabs._compute_basic_metrics(all_logs)
