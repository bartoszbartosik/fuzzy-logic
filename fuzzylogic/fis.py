import numpy as np
from .structs import Universe, LinguisticVariable

from typing import Callable, Iterable


class FuzzyInferenceSystem:

    def __init__(self, *args):
        self.universes = args
        self.rules = []


    def aggregate(self, x, domain):

        # Compute activations for each rule
        activations = self.__apply_rules(x, domain)

        return np.max(activations, axis=0)


    def __apply_rules(self, x, u):
        return [rule(x, u) for rule in self.rules]


    def defuzz(self, agg, domain):
        # Centroid defuzzification
        return np.sum(agg * domain)/np.sum(agg)


    def add_rule(self, rule: callable):
        self.rules.append(rule)


    def infer(self, x: list[LinguisticVariable] | list[float], universe: Universe):
        assert universe in self.universes

        # Compute membership values
        mu_x = {xx.universe: self.__compute_mfvals(xx) for xx in x}

        # Apply rules
        for rule in self.rules:
            rule(x, universe.domain)

        # Aggregate rules
        agg = self.aggregate([xx.value for xx in x], universe.domain)

        # Defuzzify
        crisp = self.defuzz(agg, universe.domain)

        return crisp


    def __compute_mfvals(self, x: LinguisticVariable):
        return {term: mu(x.value) for term, mu in x.universe.terms}
