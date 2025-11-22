"""
Configuration settings for the Multi-Agent Simulation.
"""

# --- Model Definitions ---
# Define the models to be used in the simulation.
# Keys are friendly names, values are the OpenRouter model slugs.
MODELS = {
    "Generalist": "meta-llama/llama-3-70b-instruct",
    "Specialized": "gryphe/mythomax-l2-13b",
    # Add more models here as needed
    # "DeepSeek": "deepseek/deepseek-chat",
}

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
