import importlib
import unittest
from unittest.mock import patch

import pandas as pd


class TestAnalysisUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch("nltk.data.find", return_value=True), patch("nltk.download", return_value=True):
            cls.analysis_utils = importlib.import_module("parrotlm.analysis_utils")

    def test_count_custom_words_counts_per_category(self):
        text = "Love is good and love stays good"
        category_dict = {"Positive": ["love", "good"], "Negative": ["bad"]}

        counts = self.analysis_utils.count_custom_words(text, category_dict)

        self.assertEqual(counts["Positive"], 4)
        self.assertEqual(counts["Negative"], 0)

    def test_process_logs_adds_metric_columns(self):
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

        with patch.object(self.analysis_utils, "analyze_text", return_value=fake_metrics):
            result = self.analysis_utils.process_logs(df)

        self.assertIn("token_count", result.columns)
        self.assertIn("noun_ratio", result.columns)
        self.assertEqual(result.loc[0, "token_count"], 2)

