import nltk
import pandas as pd
from collections import Counter

# Ensure necessary NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    nltk.data.find('taggers/universal_tagset')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('universal_tagset')

def analyze_text(text):
    """
    Analyzes a single text string and returns stylometric metrics using NLTK.
    """
    if not text:
        return {
            "token_count": 0,
            "sentence_count": 0,
            "avg_sentence_length": 0,
            "noun_ratio": 0,
            "verb_ratio": 0,
            "adj_ratio": 0,
            "adv_ratio": 0,
            "pron_ratio": 0,
        }

    # Tokenize
    tokens = nltk.word_tokenize(text)
    sentences = nltk.sent_tokenize(text)
    
    # POS Tagging (Universal Tagset for simpler tags: NOUN, VERB, ADJ, ADV, etc.)
    tagged_tokens = nltk.pos_tag(tokens, tagset='universal')
    
    # POS Counts
    pos_counts = Counter([tag for word, tag in tagged_tokens])
    total_tokens = len(tokens)
    
    # Calculate ratios
    metrics = {
        "token_count": total_tokens,
        "sentence_count": len(sentences),
        "avg_sentence_length": total_tokens / len(sentences) if len(sentences) > 0 else 0,
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
