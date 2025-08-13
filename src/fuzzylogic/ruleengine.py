import numpy as np
from matplotlib import pyplot as plt

from .structs import LinguisticVariable, Rule, Universe


class FuzzyRuleEngine:

    def __init__(self):
        self.rules: list[Rule] = []
        self.activations: dict[str, np.ndarray] = {}
        self.agg: np.ndarray | None = None
        self.crisp: float | None = None
        self._target: Universe | None = None

    def __defuzz(self):
        domain = self.rules[0].q.universe.domain
        self.crisp = np.sum(self.agg * domain)/np.sum(self.agg)

    def __aggregate(self):
        self.agg = np.max([activation for activation in self.activations.values()], axis=0)

    def __compute_activations(self, x: LinguisticVariable):
        self.activations = {str(rule): rule(x) for rule in self.rules}

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
        self.__compute_activations(x)

        # Aggregate rules
        self.__aggregate()

        # Defuzzify
        self.__defuzz()

        return self.crisp

    def plot(self):
        if self._target is None:
            raise ValueError('No target universe defined. Add rules before plotting.')

        plt.figure()
        plt.fill_between(self._target._domain, self.agg, alpha=0.5, label='aggregation')
        plt.plot(self._target._domain, self.agg)
        plt.axvline(self.crisp, linestyle='--', label=f'crisp output')
        plt.text(self.crisp+self.crisp*0.01, self._target._domain[0], f'{self.crisp:.2f}', horizontalalignment='left', verticalalignment='bottom', color='tab:blue', fontweight='bold')
        plt.xlabel(self._target.name)
        plt.ylabel(f'{r"$\mu$"}({self._target.name})')
        plt.grid()
        plt.legend()
        plt.show()


