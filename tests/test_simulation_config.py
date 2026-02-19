import unittest

from parrotlm import simulation_config


class TestSimulationConfig(unittest.TestCase):
    def test_expected_constants_exist(self):
        self.assertTrue(hasattr(simulation_config, "NUM_TURNS"))
        self.assertTrue(hasattr(simulation_config, "DATA_DIR"))
        self.assertIsInstance(simulation_config.NUM_TURNS, int)
        self.assertGreater(simulation_config.NUM_TURNS, 0)
        self.assertIsInstance(simulation_config.DATA_DIR, str)
        self.assertTrue(simulation_config.DATA_DIR)

    def test_validate_simulation_config_accepts_valid_values(self):
        simulation_config.validate_simulation_config(num_turns=3, data_dir="data")

    def test_validate_simulation_config_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            simulation_config.validate_simulation_config(num_turns=0, data_dir="data")
        with self.assertRaises(ValueError):
            simulation_config.validate_simulation_config(num_turns=2, data_dir="")

