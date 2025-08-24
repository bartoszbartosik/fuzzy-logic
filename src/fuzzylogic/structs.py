from typing import Callable

import numpy as np
from matplotlib import pyplot as plt

from ._expressionengine import _ExpressionEngine, _Expression


class LinguisticVariable:
    __slots__ = ['_data']

    def __init__(self, **kwargs):
        self._data = {}
        for key, value in kwargs.items():
            if not isinstance(value, (int, float, np.ndarray)):
                raise ValueError(f"Value for '{key}' must be a number or numpy array")
            self._data[key] = value

    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name in self.__slots__:
            super().__setattr__(name, value)
        else:
            if isinstance(value, (int, float, np.ndarray)):
                self._data[name] = float(value)
            else:
                raise ValueError(f"Value for '{name}' must be a number or numpy array")

    def __getitem__(self, item):
        if item in self._data:
            return self._data[item]
        raise KeyError(f"'{item}' not found in LinguisticVariable")

    def __repr__(self):
        items = [f"{k}={v}" for k, v in self._data.items()]
        return f"[{', '.join(items)}]"


class Universe(_ExpressionEngine):
    __slots__ = ['_name', '_domain', '_sets']

    def __init__(self, name: str, domain: np.ndarray):
        self._name: str = name
        self._domain: np.ndarray = domain
        self._sets: dict = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def domain(self) -> np.ndarray:
        return self._domain

    def plot(self):
        plt.figure()
        for fuzzy_set in self._sets.values():
            fuzzy_set._plot()
        plt.xlabel(self._name)
        plt.ylabel(f'{r'$\mu$'}({self._name})')
        plt.ylim(-0.1, 1.1)
        plt.grid()
        plt.legend()
        plt.show()

    def __call__(self, x: LinguisticVariable):
        return x[self._name]

    def __setattr__(self, name, func: Callable):
        if name in self.__slots__:
            super().__setattr__(name, func)
        else:
            assert str.isidentifier(name), f'Fuzzy Set name be an identifier, got {name}'
            self._sets[name] = _FuzzySet(self, name, func)

    def __getattr__(self, item):
        if item in self._sets:
            return self._sets[item]
        raise KeyError(f'{item} not found in universe {self._name}')

    def __eq__(self, other):
        if not isinstance(other, Universe):
            return False
        return all([
            self._name == other._name,
            np.array_equal(self._domain, other._domain)
        ])

    def __repr__(self):
        return self._name


class _FuzzySet(_ExpressionEngine):
    def __init__(self, universe: Universe, name: str, func: Callable):
        self.universe = universe
        self.name = name
        self.func = func

    def __call__(self, x: LinguisticVariable) -> float | np.ndarray:
        return self.func(x[self.universe.name])

    def __str__(self):
        return f'{self.universe}.{self.name}'

    def __repr__(self):
        return str(self)

    def _plot(self):
        x = self.universe.domain
        y = self(LinguisticVariable(**{self.universe.name: x}))
        plt.plot(x, y, label=str(self))


class Rule:
    def __new__(cls, p: _FuzzySet | _Expression, q: _FuzzySet | Universe | _Expression | float):
        if isinstance(q, _FuzzySet):
            return super().__new__(_MamdaniRule)
        else:
            return super().__new__(_TSKRule)

    def __call__(self, x: LinguisticVariable) -> float | np.ndarray: pass

    def __init__(self, p: _FuzzySet | _Expression, q: _FuzzySet | Universe | _Expression | float):
        self.p = p
        self.q = q


class _MamdaniRule(Rule):
    def __init__(self, p: _FuzzySet | _Expression, q: _FuzzySet):
        super().__init__(p, q)
        self.target = self.q.universe

    def __call__(self, x: LinguisticVariable) -> np.ndarray:
        x_q = LinguisticVariable(**{self.q.universe.name: self.q.universe.domain})
        return np.fmin(self.p(x), self.q(x_q))

    def __str__(self):
        return f'IF {self.p} THEN {self.q}'

    def __repr__(self):
        return str(self)


class _TSKRule(Rule):
    def __init__(self, p: _FuzzySet | _Expression, q: Universe | _Expression | float):
        super().__init__(p, q)

    def __call__(self, x: LinguisticVariable) -> float:
        return self.p(x) * self.q(x)

    def __str__(self):
        return f'IF {self.p} THEN {self.q}'

    def __repr__(self):
        return str(self)

