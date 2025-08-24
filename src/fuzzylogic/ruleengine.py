from abc import ABC, abstractmethod

import numpy as np
from matplotlib import pyplot as plt

from .structs import LinguisticVariable, Rule, _MamdaniRule, _TSKRule, Universe



class RuleEngine(ABC):
    def __init__(self):
        self.rules: list[Rule] = []
        self.activations: dict = {}
        self.crisp: float | None = None

    def add_rule(self, rule: Rule):
        self.rules.append(rule)

    @abstractmethod
    def infer(self, x: LinguisticVariable) -> float:
        pass

    @abstractmethod
    def _defuzz(self, *args):
        pass

    @abstractmethod
    def _compute_activations(self, x: LinguisticVariable):
        pass


class MamdaniRuleEngine(RuleEngine):
    def __init__(self):
        super().__init__()
        self.agg: np.ndarray | None = None
        self._target: Universe | None = None

    def _defuzz(self):
        domain = self.rules[0].q.universe.domain
        self.crisp = np.sum(self.agg * domain)/np.sum(self.agg)

    def __aggregate(self):
        self.agg = np.max([activation for activation in self.activations.values()], axis=0)

    def _compute_activations(self, x: LinguisticVariable):
        self.activations = {rule: rule(x) for rule in self.rules}

    def add_rule(self, rule):
        # Validate rule
        target = rule.target
        if self._target is None:
            self._target = target
        elif self._target != target:
            raise ValueError(f'Rule target universe {target.name} does not match existing target universe {self._target.name}. All rules must have the same target universe.')

        super().add_rule(rule)

    def infer(self, x: LinguisticVariable) -> float:
        # Run rules
        self._compute_activations(x)

        # Aggregate rules
        self.__aggregate()

        # Defuzzify
        self._defuzz()

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
        plt.ylim(-0.1, 1.1)
        plt.grid()
        plt.legend()
        plt.show()


class TSKRuleEngine(RuleEngine):
    def __init__(self):
        super().__init__()

    def infer(self, x: LinguisticVariable) -> float:
        self._compute_activations(x)
        self._defuzz(x)
        return self.crisp

    def _defuzz(self, x: LinguisticVariable):
        num = sum(activation * rule.q(x) for rule, activation in self.activations.items())
        den = sum(self.activations.values())
        self.crisp = num / den

    def _compute_activations(self, x: LinguisticVariable):
        self.activations = {rule: rule.p(x) for rule in self.rules}
