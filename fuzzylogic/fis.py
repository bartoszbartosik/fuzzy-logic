import numpy as np
from matplotlib import pyplot as plt

from typing import Callable


class FuzzyInferenceSystem:

    def __init__(self, **kwargs):
        self.universes = kwargs
        self.memfuncs = {key: {} for key in self.universes.keys()}
        self.rules = []


    def aggregate(self, x, universe):

        # Fetch universe domain
        u = self.universes[universe]

        # Compute activations for each rule
        activations = self.__apply_rules(x, u)

        return u, np.max(activations, axis=0)


    def __apply_rules(self, x, u):
        return [rule(x, u) for rule in self.rules]


    def defuzz(self, u, agg):
        # Centroid defuzzification
        return np.sum(agg*u)/np.sum(agg)


    def add_rule(self, rule: Callable):
        self.rules.append(rule)


    def add_membership_function(self, universe, name: str, memfunc: Callable):
        assert universe in self.universes.keys(), 'Universe not found'
        self.memfuncs[universe].update({name: {'f': memfunc}})


    def infer(self, x: np.ndarray, universe: str):
        assert universe in self.universes.keys(), 'Universe not found'

        # Aggregate rules
        u, agg = self.aggregate(x, universe)

        # Defuzzify
        crisp = self.defuzz(u, agg)

        return crisp
