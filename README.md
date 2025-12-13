# 🦜ParrotLM
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A Python-based framework for simulating and analyzing social interactions between Large Language Models (LLMs). This tool allows researchers to configure agents with specific personas, interaction settings, and models to study social dynamics, linguistic patterns, and emergent behaviors.

## 🚀 Features

*   **Dynamic Agent Configuration**: Assign specific **Personas** (e.g., "Julius Caesar", "Data Scientist") and **Interaction Settings** (e.g., "Intimate", "Professional", "Debate") to control the tone and content of the conversation.
*   **Real-Time Simulation**: Watch the conversation unfold turn-by-turn in an interactive Streamlit GUI.
*   **Multi-Model Support**: Integrate any model from OpenRouter (Llama 3, Claude 3, Grok, etc.) by simply entering its slug.
*   **Stylometric Analysis**: Built-in NLP tools (powered by **NLTK**) to analyze:
    *   Part-of-Speech (POS) distribution.
    *   Sentence length and complexity.
    *   **Custom Word Frequency (LIWC-style)**: Define your own categories (e.g., "Aggression", "Politeness") and track their usage.
*   **Data Logging**: All experiments are automatically logged to JSONL and CSV for easy export and external analysis.

## 🛠️ Technical Stack

*   **Core Logic**: Python 3.10+
*   **GUI**: [Streamlit](https://streamlit.io/) for the interactive dashboard.
*   **LLM Integration**: [OpenAI Python Client](https://github.com/openai/openai-python) (configured for OpenRouter API).
*   **Data Visualization**: [Plotly](https://plotly.com/python/) for interactive charts.
*   **NLP & Analysis**: [NLTK](https://www.nltk.org/) for linguistic processing and POS tagging.
*   **Resilience**: `tenacity` for robust API retry logic.

## ⚙️ Orchestration Pipeline

The framework follows a modular pipeline designed for flexibility and real-time feedback:

1.  **Configuration (GUI)**:
    *   User defines Model IDs, Personas, Interaction Settings, and Temperature in `gui_app.py`.
    *   User selects or types a "Conversation Starter".

2.  **Prompt Generation**:
    *   `prompt_utils.py` dynamically constructs the System Prompt by combining:
        *   **Scenario Base**: The interaction setting (e.g., "You are in a professional environment...").
        *   **Persona Context**: The character description (e.g., "You are a pirate...").
        *   **Custom Instructions**: Any additional constraints.

3.  **Orchestration (Streaming)**:
    *   `orchestrator.py` initializes two `Agent` instances with the generated prompts.
    *   The `run_simulation` method executes the conversation loop.
    *   It **yields** `log_entry` objects turn-by-turn, allowing the GUI to display messages and metrics in real-time without waiting for the entire simulation to finish.

4.  **Data Logging**:
    *   Each turn (content, latency, token usage, model ID) is appended to an in-memory list and saved incrementally to `data/experiment_log.jsonl`.

5.  **Analysis**:
    *   `analysis_utils.py` processes the logs using `NLTK` to extract linguistic features.
    *   The GUI visualizes these metrics, grouping them by model or category for comparative analysis.

## 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

> **Note:** NLTK does not require a separate model download; required data will be downloaded automatically on first run.

4.  **Set up API Key**:
    *   Create a `.env` file (copy from `.env.example`) and add your `OPENROUTER_API_KEY`.
    *   Alternatively, enter the key directly in the GUI sidebar.

## 🖥️ Usage

Run the Streamlit application:

```bash
python -m streamlit run gui_app.py
```

1.  **Configure Agents**: On the main page, enter the Model Slugs (e.g., `x-ai/grok-beta`) and describe their Personas.
2.  **Set Context**: Choose an Interaction Setting (e.g., "Intimate") and a Conversation Starter.
3.  **Run**: Click "Start Simulation" to watch the agents converse.
4.  **Analyze**: Switch to the "Stylometric Analysis" tab to view linguistic insights.

## License

This project is licensed under the Apache License, Version 2.0. See `LICENSE` file for details.
