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

