"""Text analysis and custom lexicon utilities used by the app."""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Dict, Iterable

import nltk
import pandas as pd

logger = logging.getLogger(__name__)

_RESOURCE_PATHS = (
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ("taggers/universal_tagset", "universal_tagset"),
)
_NLTK_READY = False


def _empty_metrics() -> Dict[str, float]:
    return {
        "token_count": 0,
        "sentence_count": 0,
        "avg_sentence_length": 0.0,
        "noun_ratio": 0.0,
        "verb_ratio": 0.0,
        "adj_ratio": 0.0,
        "adv_ratio": 0.0,
        "pron_ratio": 0.0,
    }


def ensure_nltk_resources(download_missing: bool = False) -> None:
    """Validate required NLTK resources and optionally download missing ones."""
    missing: list[str] = []
    for resource_path, download_name in _RESOURCE_PATHS:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            missing.append(download_name)

    if not missing:
        return

    if download_missing:
        for download_name in missing:
            nltk.download(download_name, quiet=True)
        return ensure_nltk_resources(download_missing=False)

    missing_csv = ", ".join(sorted(set(missing)))
    raise RuntimeError(
        "Missing NLTK resources. Run `python -c \"import nltk; "
        f"[nltk.download(r) for r in {sorted(set(missing))}]\"` to install: {missing_csv}."
    )


def _ensure_nltk_ready() -> None:
    global _NLTK_READY
    if _NLTK_READY:
        return
    ensure_nltk_resources(download_missing=False)
    _NLTK_READY = True


def analyze_text(text: Any) -> Dict[str, float]:
    """Analyze stylometric metrics for one text value."""
    if text is None:
        return _empty_metrics()

    text_value = text if isinstance(text, str) else str(text)
    text_value = text_value.strip()
    if not text_value:
        return _empty_metrics()

    _ensure_nltk_ready()

    tokens = nltk.word_tokenize(text_value)
    sentences = nltk.sent_tokenize(text_value)

    if not tokens:
        return _empty_metrics()

    tagged_tokens = nltk.pos_tag(tokens, tagset="universal")
    pos_counts = Counter(tag for _, tag in tagged_tokens)
    total_tokens = len(tokens)
    total_sentences = len(sentences)

    return {
        "token_count": float(total_tokens),
        "sentence_count": float(total_sentences),
        "avg_sentence_length": total_tokens / total_sentences if total_sentences else 0.0,
        "noun_ratio": pos_counts.get("NOUN", 0) / total_tokens,
        "verb_ratio": pos_counts.get("VERB", 0) / total_tokens,
        "adj_ratio": pos_counts.get("ADJ", 0) / total_tokens,
        "adv_ratio": pos_counts.get("ADV", 0) / total_tokens,
        "pron_ratio": pos_counts.get("PRON", 0) / total_tokens,
    }


def _safe_analyze_text(value: Any) -> Dict[str, float]:
    try:
        return analyze_text(value)
    except Exception:
        logger.exception("Failed to analyze a text row; using zeroed metrics.")
        return _empty_metrics()


def process_logs(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text analysis to a conversation log dataframe."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")
    if df.empty or "content" not in df.columns:
        return df.copy()

    metrics_by_row = []
    for content in df["content"]:
        metrics_by_row.append(_safe_analyze_text(content))

    metrics_df = pd.DataFrame(metrics_by_row, index=df.index)
    return pd.concat([df.copy(), metrics_df], axis=1)


def _normalize_tokens(text: Any) -> Iterable[str]:
    text_value = text if isinstance(text, str) else str(text or "")
    return re.findall(r"\b[\w']+\b", text_value.lower())


def count_custom_words(text: Any, category_dict: Dict[str, list[str]]) -> Dict[str, int]:
    """Count matches for custom word categories."""
    if not category_dict:
        return {}

    token_counter = Counter(_normalize_tokens(text))
    counts: Dict[str, int] = {}

    for category, words in category_dict.items():
        words_list = words or []
        category_total = 0
        for word in words_list:
            if not word:
                continue
            normalized_word = str(word).lower()
            category_total += token_counter.get(normalized_word, 0)

        counts[category] = category_total

    return counts


def process_custom_lexicon(df: pd.DataFrame, category_dict: Dict[str, list[str]]) -> pd.DataFrame:
    """Add custom lexicon counts to a dataframe that contains a `content` column."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")
    if df.empty or "content" not in df.columns or not category_dict:
        return df.copy()

    lexicon_counts_by_row = []
    for content in df["content"]:
        lexicon_counts_by_row.append(count_custom_words(content, category_dict))

    lexicon_df = pd.DataFrame(lexicon_counts_by_row, index=df.index)
    return pd.concat([df.copy(), lexicon_df], axis=1)
