from typing import Callable

import numpy as np

from fuzzylogic.structs import Universe, LinguisticVariable


def conjunction(x: list):
    return np.min(list)


def disjunction(x: list):
    return np.max(list)


def implication(p: Callable, q: Callable):
    def activation(x, domain):
        __run(__imply(x, p), q, domain)
    return activation


def __and(x: np.ndarray, mus: list[Callable]):
    return np.min([mu(xx) for xx, mu in zip(x, mus)], axis=0)


def __or(x, mus):
    return np.max([mu(xx) for xx, mu in zip(x, mus)], axis=0)


def __imply(x, mu):
    return np.min(mu(x))


def __run(p, q, domain):
    return np.min([q(domain), np.ones_like(q(domain))*p], axis=0)


def _terms2dict(k: list[tuple[Universe, str]]):
    mu_dict = {}
    for universe, term in k:
        if universe not in mu_dict:
            mu_dict[universe] = {}
        mu_dict[universe][term] = universe.terms[term]
    return mu_dict


class Rule:
    def __init__(self, p: list[tuple[Universe, str]], rule: Callable, q: tuple[Universe, str]):
        self.p = _terms2dict(p)
        self.rule = rule

        self.target_universe = q[0]
        self.target_term = q[1]
        self.target_muvals = self.target_universe.terms[self.target_term](self.target_universe.domain)

    def __call__(self, x: list[LinguisticVariable]):
        # TODO: come up with more concise way to identify mu for each x
        # FIXME: does LinguisticVariable really have to contain the universe?
        # Compute activations for each rule (concise)
        muvals = [self.p[linvar.universe][term](linvar.value) for linvar in x for term in self.p[linvar.universe]]

        # Compute activations for each rule (full)
        muvals = {}
        for linvar in x:
            for term in self.p[linvar.universe]:
                muvals[linvar.universe][term] = self.p[linvar.universe][term](linvar.value)


        y = np.min(muvals)
        a = np.fmin(muvals, self.target_muvals)

    def __str__(self):
        operator = None
        match str(self.rule):
            case str(conjunction):
                operator = 'AND'
            case str(disjunction):
                operator = 'OR'
            case str(implication):
                operator = 'IMPLIES'

        if operator is None:
            raise ValueError(f'Unknown rule type: {self.rule}')

        return f'IF [{f" {operator.upper()} ".join([f"{universe} IS {term}" for universe in self.p for term in self.p[universe]])}] THEN {self.target_universe} IS {self.target_term}'

    def __repr__(self):
        return self.__str__()


