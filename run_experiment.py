import os
import pandas as pd
import logging
from simulation_config import MODELS, SCENARIOS, NUM_TURNS, ITERATIONS, DATA_DIR
from orchestrator import Orchestrator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_experiments(models=None, scenarios=None, num_turns=NUM_TURNS, iterations=ITERATIONS, 
                    agent_a_params=None, agent_b_params=None):
    """
    Main function to run all configured experiments.
    Can be called with custom arguments for GUI integration.
    Yields log entries for real-time updates.
    """
    # Use defaults from config if not provided
    models = models or MODELS
    scenarios = scenarios or SCENARIOS
    agent_a_params = agent_a_params or {}
    agent_b_params = agent_b_params or {}
    
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    jsonl_path = os.path.join(DATA_DIR, "experiment_log.jsonl")
    
    logger.info("Starting Batch Experiments...")
    
    total_experiments = len(models) * len(models) * len(scenarios) * iterations
    current_experiment = 0

    # Loop through all combinations
    model_names = list(models.keys())
    
    for model_a_name in model_names:
        for model_b_name in model_names:
            for scenario_name, system_prompt_base in scenarios.items():
                for i in range(iterations):
                    current_experiment += 1
                    logger.info(f"--- Experiment {current_experiment}/{total_experiments} ---")
                    
                    agent_a_config = {
                        "model": models[model_a_name],
                        "system_prompt": system_prompt_base,
                        "params": agent_a_params
                    }
                    agent_b_config = {
                        "model": models[model_b_name],
                        "system_prompt": system_prompt_base,
                        "params": agent_b_params
                    }
                    
                    # Initialize Orchestrator
                    orchestrator = Orchestrator(
                        agent_a_config=agent_a_config,
                        agent_b_config=agent_b_config,
                        scenario_name=scenario_name
                    )
                    
                    # Run Simulation (Streaming)
                    for log_entry in orchestrator.run_simulation(num_turns=num_turns):
                        # Save logs incrementally
                        orchestrator.save_logs(jsonl_path)
                        yield log_entry

    logger.info("All experiments completed.")
    
    # Convert to CSV
    convert_to_csv(jsonl_path)

def convert_to_csv(jsonl_path: str):
    """
    Converts the JSONL log file to a CSV file for easier analysis.
    """
    csv_path = jsonl_path.replace(".jsonl", ".csv")
    try:
        logger.info(f"Converting {jsonl_path} to {csv_path}...")
        df = pd.read_json(jsonl_path, lines=True)
        df.to_csv(csv_path, index=False)
        logger.info("Conversion successful.")
    except Exception as e:
        logger.error(f"Error converting to CSV: {e}")

if __name__ == "__main__":
    run_experiments()
