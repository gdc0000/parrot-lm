import pytest
from unittest.mock import patch

from main import (
    initialize_infrastructure,
    configure_simulation_agents,
    execute_simulation,
    process_simulation_results,
    main,
)
from parrotlm.simulation_config import SimulationConfig


def test_main_happy_path():
    with patch("main.initialize_infrastructure") as mock_init:
        with patch("main.configure_simulation_agents") as mock_config:
            with patch("main.execute_simulation") as mock_exec:
                with patch("main.process_simulation_results") as mock_process:
                    mock_config.return_value = ("agent_a", "agent_b")
                    mock_exec.return_value = [{"log": "entry"}]

                    main()

                    mock_init.assert_called_once()
                    mock_config.assert_called_once()
                    mock_exec.assert_called_once()
                    mock_process.assert_called_once()


def test_main_failure_logs_phase():
    with patch("main.initialize_infrastructure") as mock_init:
        with patch("main.log_structured") as mock_log:
            mock_init.side_effect = ValueError("Config error")

            with pytest.raises(ValueError, match="Config error"):
                main()

            mock_log.assert_called_with(
                50,  # logging.CRITICAL
                "unhandled_exception",
                failed_phase="initialization",
                error="Config error",
                exception_type="ValueError",
            )
