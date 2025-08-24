from typing import Callable

from . import operators
from .config import Config


class _ExpressionEngine:

    def __call__(self, *args): pass
    def __add__(self, other): return self.__evaluate(other, lambda x, y: x + y, '+')
    def __sub__(self, other): return self.__evaluate(other, lambda x, y: x - y, '-')
    def __mul__(self, other): return self.__evaluate(other, lambda x, y: x * y, '*')
    def __truediv__(self, other): return self.__evaluate(other, lambda x, y: x / y, '/')
    def __pow__(self, other): return self.__evaluate(other, lambda x, y: x ** y, '**')
    def __rpow__(self, other): return self.__evaluate(other, lambda x, y: y ** x, '**')
    def __radd__(self, other): return self.__add__(other)
    def __rsub__(self, other): return self.__evaluate(other, lambda x, y: y - x, '-')
    def __rmul__(self, other): return self.__mul__(other)
    def __rtruediv__(self, other): return self.__evaluate(other, lambda x, y: y / x, '/')
    def __or__(self, other): return self.__evaluate(other, Config.tconorm, operators._TCONORM_STR)
    def __and__(self, other): return self.__evaluate(other, Config.tnorm, operators._TNORM_STR)

    def __evaluate(self, other, operator: Callable, operator_str: str):
        if hasattr(other, '__call__'):
            return _Expression(lambda x: operator(self(x), other(x)), f'({self} {operator_str} {other})')
        elif isinstance(other, (int, float)):
            return _Expression(lambda x: operator(self(x), other), f'({self} {operator_str} {other})')
        else:
            raise ValueError(f'Cannot apply operator with {type(other)}')


class _Expression(_ExpressionEngine):
    def __init__(self, func: Callable, repr: str):
        self.__func = func
        self.__str = repr

    def __call__(self, x) -> float:
        return self.__func(x)

    def __repr__(self):
        return self.__str
