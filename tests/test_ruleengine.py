import unittest
import numpy as np

from src.fuzzylogic import FuzzyRuleEngine
from src.fuzzylogic.functions import trimf
from src.fuzzylogic.structs import Universe, LinguisticVariable, Rule


class TestFuzzyRuleEngine(unittest.TestCase):
    """Test suite for FuzzyRuleEngine module."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Define universes
        self.quality = Universe('quality', np.arange(0, 11, 1))
        self.service = Universe('service', np.arange(0, 11, 1))
        self.tip = Universe('tip', np.arange(0, 26, 1))

        # Define membership functions
        self.quality.low = trimf((0, 0, 5))
        self.quality.medium = trimf((0, 5, 10))
        self.quality.high = trimf((5, 10, 10))
        self.service.low = trimf((0, 0, 5))
        self.service.medium = trimf((0, 5, 10))
        self.service.high = trimf((5, 10, 10))
        self.tip.low = trimf((0, 0, 13))
        self.tip.medium = trimf((0, 13, 25))
        self.tip.high = trimf((13, 25, 25))

        # Create rule engine
        self.fis = FuzzyRuleEngine()

        # Define rules
        self.rule_1 = Rule(self.quality.low & self.service.low, self.tip.low)
        self.rule_2 = Rule(self.service.medium, self.tip.medium)
        self.rule_3 = Rule(self.quality.high | self.service.high, self.tip.high)


    def test_original_demo_scenario(self):
        """Test the original tip calculation scenario from demo."""
        # Add rules to engine
        self.fis.add_rule(self.rule_1)
        self.fis.add_rule(self.rule_2)
        self.fis.add_rule(self.rule_3)

        # Test inference with the original demo values
        lvar = LinguisticVariable(quality=6.5, service=9.8)
        result = self.fis.infer(lvar, self.tip)

        # Validate the result
        self.assertIsInstance(result, (float, np.floating))
        self.assertAlmostEqual(result, 20.24, places=2)


def create_test_suite():
    """Create a test suite with all test cases."""
    suite = unittest.TestSuite()

    # Add all test methods from TestFuzzyRuleEngine
    suite.addTest(unittest.makeSuite(TestFuzzyRuleEngine))

    return suite


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
