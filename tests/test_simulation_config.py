import pytest

from parrotlm import simulation_config


def test_expected_constants_exist():
    assert hasattr(simulation_config, "NUM_TURNS")
    assert hasattr(simulation_config, "DATA_DIR")
    assert isinstance(simulation_config.NUM_TURNS, int)
    assert simulation_config.NUM_TURNS > 0
    assert isinstance(simulation_config.DATA_DIR, str)
    assert simulation_config.DATA_DIR


def test_validate_simulation_config_accepts_valid_values():
    simulation_config.validate_simulation_config(num_turns=3, data_dir="data")


def test_validate_simulation_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        simulation_config.validate_simulation_config(num_turns=0, data_dir="data")
    with pytest.raises(ValueError):
        simulation_config.validate_simulation_config(num_turns=2, data_dir="")
    with pytest.raises(ValueError):
        simulation_config.validate_simulation_config(num_turns=-1, data_dir="data")
    with pytest.raises(ValueError):
        simulation_config.validate_simulation_config(num_turns="3", data_dir="data")
    with pytest.raises(ValueError):
        simulation_config.validate_simulation_config(num_turns=3, data_dir="   ")
    with pytest.raises(ValueError):
        simulation_config.validate_simulation_config(num_turns=3, data_dir=123)

