import unittest
import numpy as np

from src.fuzzylogic import FuzzyRuleEngine
from src.fuzzylogic.functions import trimf
from src.fuzzylogic.structs import Universe, LinguisticVariable, Rule
from src.fuzzylogic.config import Config
from src.fuzzylogic import operators


class TestFuzzyRuleEngine(unittest.TestCase):
    """Test suite for FuzzyRuleEngine module."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        pass


    def test_tipping_problem(self):
        """Test the original tip calculation scenario from demo."""

        # Set configuration for t-norm and t-conorm
        Config.tnorm = operators.minimum
        Config.tconorm = operators.maximum

        # Define universes
        quality = Universe('quality', np.arange(0, 11, 1))
        service = Universe('service', np.arange(0, 11, 1))
        tip = Universe('tip', np.arange(0, 26, 1))

        # Define membership functions
        quality.low = trimf((0, 0, 5))
        quality.medium = trimf((0, 5, 10))
        quality.high = trimf((5, 10, 10))
        service.low = trimf((0, 0, 5))
        service.medium = trimf((0, 5, 10))
        service.high = trimf((5, 10, 10))
        tip.low = trimf((0, 0, 13))
        tip.medium = trimf((0, 13, 25))
        tip.high = trimf((13, 25, 25))

        # Define rules
        rule_1 = Rule(quality.low & service.low, tip.low)
        rule_2 = Rule(service.medium, tip.medium)
        rule_3 = Rule(quality.high | service.high, tip.high)

        # Create rule engine
        fis = FuzzyRuleEngine()

        # Add rules to engine
        fis.add_rule(rule_1)
        fis.add_rule(rule_2)
        fis.add_rule(rule_3)

        # Test inference with the original demo values
        lvar = LinguisticVariable(quality=6.5, service=9.8)
        result = fis.infer(lvar)

        quality.plot()
        service.plot()
        tip.plot()
        fis.plot()

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
