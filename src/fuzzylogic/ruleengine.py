import numpy as np

from .structs import LinguisticVariable, Rule, Universe


class FuzzyRuleEngine:

    def __init__(self):
        self.rules: list[Rule] = []
        self.activations: dict[str, np.ndarray] = {}
        self._target: Universe | None = None


    def defuzz(self, agg: np.ndarray, domain: np.ndarray) -> float:
        # Centroid defuzzification
        return np.sum(agg * domain)/np.sum(agg)


    def add_rule(self, rule: Rule):
        # Validate rule
        target = rule.q.universe
        if self._target is None:
            self._target = target
        elif self._target != target:
            raise ValueError(f'Rule target universe {target.name} does not match existing target universe {self._target.name}. All rules must have the same target universe.')

        self.rules.append(rule)


    def infer(self, x: LinguisticVariable) -> float:
        # Run rules
        self.activations = {str(rule): rule(x) for rule in self.rules}

        # Aggregate rules
        agg = np.max([activation for activation in self.activations.values()], axis=0)

        # Defuzzify
        target = self.rules[0].q.universe
        crisp = self.defuzz(agg, target._domain)

        return crisp
