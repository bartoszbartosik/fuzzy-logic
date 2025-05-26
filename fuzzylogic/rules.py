from typing import Callable

import numpy as np

from fuzzylogic.structs import Universe, LinguisticVariable


def negation(p: Callable):
    def activation(x):
        return np.ones_like(p(x)) - p(x)
    return activation


def conjunction(p: list[Callable], q: Callable):
    def activation(x, domain):
        return __run(__and(x, p), q, domain)
    return activation


def disj(x: list[float], domain: np.ndarray, p: list[tuple[Universe, str]], q: tuple[Universe, str]):
    return __run(__or(x, p), q, domain)


def disjunction(p: list[Callable], q: Callable):
    def activation(x, domain):
        return __run(__or(x, p), q, domain)
    return activation


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

        self.target_muvals = q[0].terms[q[1]](q[0].domain)

    def __call__(self, x: list[LinguisticVariable]):
        # TODO: come up with more concise way to identify mu for each x
        # FIXME: does LinguisticVariable really have to contain the universe?
        # Compute activations for each rule (concise)
        muvals = [self.p[linvar.universe][term](linvar.value) for linvar in x for term in self.p[linvar.universe]]

        y = np.min(muvals)
        a = np.fmin(muvals, self.target_muvals)




