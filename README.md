# Multi-Agent Simulation Framework

A Python-based framework to analyze social dynamics between LLMs using the OpenRouter API. This tool allows you to simulate interactions between two agents under various experimental conditions and analyze the results.

## Features

- **Modular Architecture**: Separate configuration, orchestration, and execution logic.
- **Configurable Experiments**: Easily define models, scenarios, and simulation parameters.
- **Rich Data Logging**: Captures detailed metadata for each turn (latency, tokens, finish reason, etc.) in JSONL format.
- **CSV Conversion**: Automatically converts logs to Pandas-ready CSV files.
- **Resilience**: Built-in retry logic for API failures.
- **Interactive GUI**: Streamlit-based interface for easy configuration, execution, and analysis.

## Setup

1.  **Clone the repository** (if you haven't already).
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configure Environment**:
    - Copy `.env.example` to `.env`:
        ```bash
        cp .env.example .env
        ```
        *(On Windows, you can just copy and rename the file manually or use `copy .env.example .env`)*
    - Edit `.env` and add your **OpenRouter API Key**.

## Configuration

Modify `simulation_config.py` to customize your experiments:

-   **MODELS**: Add or remove models (keys are friendly names, values are OpenRouter slugs).
-   **SCENARIOS**: Define the system prompts for different social contexts.
-   **NUM_TURNS**: Set the length of each conversation.
-   **ITERATIONS**: Set how many times to repeat each experiment.

## Running Experiments

To start the simulation batch process:

```bash
python run_experiment.py
```

The script will:
1.  Iterate through all configured Model pairs and Scenarios.
2.  Run the simulations.
3.  Save raw logs to `data/experiment_log.jsonl`.
4.  Convert the logs to `data/experiment_log.csv` upon completion.

### Using the GUI

For an interactive experience, use the Streamlit app:

```bash
streamlit run gui_app.py
```

The GUI allows you to:
-   **Configure**: Select models, scenarios, and adjust parameters on the fly.
-   **Run**: Execute simulations and view live conversation logs.
-   **Analyze**: Visualize performance metrics (latency, tokens) and compare scenarios with interactive charts.

## Data Output

The output CSV contains the following columns:

-   `experiment_id`: Unique UUID for the conversation.
-   `turn_id`: Sequence number of the turn.
-   `scenario`: The experimental condition.
-   `speaker_model`: The model generating the text.
-   `responder_model`: The model receiving the text.
-   `content`: The generated response.
-   `latency_ms`: Response generation time.
-   `input_tokens` / `output_tokens`: Token usage.
-   `finish_reason`: Why the generation stopped.
-   `is_refusal`: Boolean flag for safety refusals.
