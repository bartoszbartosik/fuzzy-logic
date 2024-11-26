from typing import Callable

import numpy as np


def conjunction(p: list[Callable], q: Callable):
    p_and = lambda x: np.min([pp(xx) for xx, pp in zip(x, p)], axis=0)
    activation = lambda x, u: np.min([q(u), np.ones_like(q(u))*p_and(x)], axis=0)
    return activation


def disjunction(p: list[Callable], q: Callable):
    p_or = lambda x: np.max([pp(xx) for xx, pp in zip(x, p)], axis=0)
    activation = lambda x, u: np.min([q(u), np.ones_like(q(u))*p_or(x)], axis=0)
    return activation


def implication(p: Callable, q: Callable):
    activation = lambda x, u: np.min([q(u), np.ones_like(q(u))*np.min(p(x))], axis=0)
    return activation
