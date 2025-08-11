from typing import Callable

import numpy as np

from . import operators
from .config import Config



class Universe:

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

    def __setattr__(self, name, func: Callable):
        if name in self.__slots__:
            super().__setattr__(name, func)
        else:
            assert str.isidentifier(name), f'Fuzzy Set name be an identifier, got {name}'
            self._sets[name] = FuzzySet(self, name, func)

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

    def to_dict(self):
        return self._data.copy()

    @classmethod
    def from_dict(cls, data):
        for key, value in data.items():
            setattr(cls, key, value)
        return LinguisticVariable(**data)

    @classmethod
    def from_json_file(cls, file_path):
        import json
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class FuzzySet:
    def __init__(self, universe: Universe, name: str, func: Callable):
        self.universe = universe
        self.name = name
        self.func = func

    def __call__(self, x: LinguisticVariable) -> float | np.ndarray:
        return self.func(x[self.universe.name])

    def __or__(self, other):
        return self.__compare(other, Config.tconorm, operators._TCONORM_STR, operators._TCONORM_SYMBOL_STR)

    def __and__(self, other):
        return self.__compare(other, Config.tnorm, operators._TNORM_STR, operators._TNORM_SYMBOL_STR)

    def __compare(self, other, operator, operator_str, operator_symbol_str):
        if isinstance(other, FuzzyExpressionTree):
            return other | self
        else:
            if self.universe == other.universe:
                universe = self.universe
                name = self.name + operator_symbol_str + other.name
                func = lambda x: operator(self.func(x), other.func(x))
                return FuzzySet(universe, name, func)
            else:
                return FuzzyExpressionTree(self, other, operator, operator_str)


    def __str__(self):
        return f'{self.universe}.{self.name}'

    def __repr__(self):
        return str(self)


class FuzzyExpressionTree:
    def __init__(self, A: FuzzySet, B: FuzzySet, operator, operator_str: str):
        self._sets = {
            operator: {
                A.universe.name: A,
                B.universe.name: B
            }
        }
        self._str = f'({A} {operator_str} {B})'

    def __call__(self, x: LinguisticVariable) -> float:
        return self.__evaluate_tree(self._sets, x)

    def __or__(self, other):
        return self.__build_tree(other, Config.tconorm, operators._TCONORM_STR)

    def __and__(self, other):
        return self.__build_tree(other, Config.tnorm, operators._TNORM_STR)

    def __str__(self):
        return self._str

    def __repr__(self):
        return self._str

    def __build_tree(self, other, operator: Callable, operator_str: str):
        if isinstance(other, FuzzyExpressionTree):
            self._sets = {operator: {**self._sets, **other._sets}}
        elif isinstance(other, FuzzySet):
            self._sets = {operator: {**self._sets, other.universe.name: other}}
        self._str = f'({self._str} {operator_str} {other})'
        return self

    @staticmethod
    def __evaluate_tree(sets: dict, lvar: LinguisticVariable) -> float:

        operator = next(iter(sets))
        children = [{child: sets[operator][child]} for child in sets[operator]]

        values = []
        for child in children:
            node = next(iter(child.values()))
            if isinstance(node, FuzzySet):
                value = node(lvar)
                values.append(value)
            else:
                value = FuzzyExpressionTree.__evaluate_tree(child, lvar)
                values.append(value)

        return operator(*values)


class Rule:
    def __init__(self, p: FuzzySet | FuzzyExpressionTree, q: FuzzySet):
        self.p = p
        self.q = q

    def __call__(self, x: LinguisticVariable) -> float | np.ndarray:
        x_q = LinguisticVariable.from_dict({self.q.universe.name: self.q.universe.domain})
        return np.fmin(self.p(x), self.q(x_q))

    def __str__(self):
        return f'IF {self.p} THEN {self.q}'

    def __repr__(self):
        return str(self)


