from typing import Callable
import numpy as np

from .structs import LinguisticVariable, Rule


class FuzzyRuleEngine:

    def __init__(self):
        self.rules: list[Rule] = []
        self.activations = {}


    def defuzz(self, agg, domain):
        # Centroid defuzzification
        return np.sum(agg * domain)/np.sum(agg)


    def add_rule(self, rule: Callable):
        self.rules.append(rule)


    def infer(self, x: LinguisticVariable, target):
        # Run rules
        self.activations = {str(rule): rule(x) for rule in self.rules}

        # Aggregate rules
        agg = np.max([activation for activation in self.activations.values()], axis=0)

        # Defuzzify
        crisp = self.defuzz(agg, target._domain)

        return crisp
