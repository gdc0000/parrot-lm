"""Default configuration values used by the simulation runtime."""

# Number of turns for each agent in a simulation run.
# Total messages = NUM_TURNS * 2
NUM_TURNS: int = 10

# Output directory for persisted logs.
DATA_DIR: str = "data"


def validate_simulation_config(num_turns: int = NUM_TURNS, data_dir: str = DATA_DIR) -> None:
    """Validate core configuration values and raise a clear error if invalid."""
    if not isinstance(num_turns, int) or num_turns <= 0:
        raise ValueError("`num_turns` must be a positive integer.")
    if not isinstance(data_dir, str) or not data_dir.strip():
        raise ValueError("`data_dir` must be a non-empty string.")
