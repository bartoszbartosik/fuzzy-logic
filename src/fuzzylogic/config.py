from . import operators
from dataclasses import dataclass


@dataclass
class Config:
    tnorm = operators.minimum
    tconorm = operators.maximum
