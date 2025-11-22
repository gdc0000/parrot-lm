"""
Configuration settings for the Multi-Agent Simulation.
"""

# --- Model Definitions ---
# Organized by size category for fair comparison.
# Keys are categories, values are dicts of {Friendly Name: OpenRouter Slug}
MODELS_BY_SIZE = {
    "Small (~7B-13B)": {
        "MythoMax 13B (Specialized)": "gryphe/mythomax-l2-13b",
        "Mistral 7B (Generalist)": "mistralai/mistral-7b-instruct",
        "Llama 3 8B (Generalist)": "meta-llama/llama-3-8b-instruct",
        "Toppy M 7B (Specialized)": "undi95/toppy-m-7b",
    },
    "Medium (~30B-70B)": {
        "Llama 3 70B (Generalist)": "meta-llama/llama-3-70b-instruct",
        "Mixtral 8x7B (Generalist)": "mistralai/mixtral-8x7b-instruct",
        "Midnight Rose 70B (Specialized)": "sophosympatheia/midnight-rose-70b",
    },
    "Large (Frontier)": {
        "Claude 3 Opus": "anthropic/claude-3-opus",
        "GPT-4 Turbo": "openai/gpt-4-turbo",
        "Gemini Pro 1.5": "google/gemini-pro-1.5",
    }
}

# Flattened list for backward compatibility if needed, though we will update GUI to use categories
MODELS = {name: slug for category in MODELS_BY_SIZE.values() for name, slug in category.items()}

# --- Scenarios ---
# Define the system prompts for different experimental conditions.
SCENARIOS = {
    "Strangers": (
        "You do not know the other person. You are meeting for the first time. "
        "Engage in conversation naturally, getting to know them."
    ),
    "Early Dating": (
        "You have just started dating. There is mutual attraction. "
        "Flirt and deepen your connection."
    ),
    "Committed": (
        "You are in a long-term, committed relationship. "
        "Express your deep bond and familiarity with each other."
    ),
}

# --- Simulation Parameters ---
# Number of turns per agent in a single conversation.
# Total messages = NUM_TURNS * 2
NUM_TURNS = 10

# Number of times to repeat each experiment configuration.
ITERATIONS = 3

# Output directory for logs
DATA_DIR = "data"
