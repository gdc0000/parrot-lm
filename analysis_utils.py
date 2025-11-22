import spacy
import pandas as pd
from collections import Counter

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Fallback if model not found, though it should be installed
    print("Downloading en_core_web_sm...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def analyze_text(text):
    """
    Analyzes a single text string and returns stylometric metrics.
    """
    doc = nlp(text)
    
    # POS Counts
    pos_counts = Counter([token.pos_ for token in doc])
    total_tokens = len(doc)
    
    # Calculate ratios
    metrics = {
        "token_count": total_tokens,
        "sentence_count": len(list(doc.sents)),
        "avg_sentence_length": total_tokens / len(list(doc.sents)) if len(list(doc.sents)) > 0 else 0,
        "noun_ratio": pos_counts.get("NOUN", 0) / total_tokens if total_tokens > 0 else 0,
        "verb_ratio": pos_counts.get("VERB", 0) / total_tokens if total_tokens > 0 else 0,
        "adj_ratio": pos_counts.get("ADJ", 0) / total_tokens if total_tokens > 0 else 0,
        "adv_ratio": pos_counts.get("ADV", 0) / total_tokens if total_tokens > 0 else 0,
        "pron_ratio": pos_counts.get("PRON", 0) / total_tokens if total_tokens > 0 else 0,
    }
    
    return metrics

def process_logs(df):
    """
    Applies text analysis to a DataFrame of conversation logs.
    Expects a 'content' column.
    Returns the DataFrame with added metric columns.
    """
    if df.empty or "content" not in df.columns:
        return df
        
    # Apply analysis to each message
    # We use apply with a lambda to expand the dictionary into columns
    metrics_df = df["content"].apply(lambda x: pd.Series(analyze_text(x)))
    
    # Concatenate with original dataframe
    result_df = pd.concat([df, metrics_df], axis=1)
    
    return result_df

def count_custom_words(text, category_dict):
    """
    Counts occurrences of words from user-defined categories.
    category_dict: { "CategoryName": ["word1", "word2"], ... }
    Returns a dictionary with counts for each category.
    """
    text_lower = text.lower()
    tokens = text_lower.split() # Simple tokenization for word matching
    
    counts = {cat: 0 for cat in category_dict}
    
    for cat, words in category_dict.items():
        for word in words:
            # Simple matching: count occurrences of the word in the token list
            # Note: This is a basic implementation. For more robust matching (regex, stems), 
            # more complex logic would be needed.
            counts[cat] += tokens.count(word.lower())
            
    return counts

def process_custom_lexicon(df, category_dict):
    """
    Applies custom lexicon counting to the dataframe.
    """
    if df.empty or "content" not in df.columns or not category_dict:
        return df
        
    lexicon_df = df["content"].apply(lambda x: pd.Series(count_custom_words(x, category_dict)))
    return pd.concat([df, lexicon_df], axis=1)
