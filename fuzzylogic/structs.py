from dataclasses import dataclass
from typing import Callable

import numpy as np


class Universe:

    def __init__(self, name: str, domain: np.ndarray):
        self.name: str = name
        self.domain: np.ndarray = domain
        self.terms: dict[str, Callable] = {}

    def __repr__(self):
        return self.name


@dataclass
class LinguisticVariable:
    value: float
    universe: str

    def __repr__(self):
        return f'{self.universe.name} = {self.value}'
