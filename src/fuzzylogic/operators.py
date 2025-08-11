import numpy as np


_TNORM_STR = 'AND'
_TNORM_SYMBOL_STR = '&'
_TCONORM_STR = 'OR'
_TCONORM_SYMBOL_STR = '|'


# ----------------- #
#   T - N O R M S   #
# ----------------- #

def minimum(a, b):
    return np.fmin(a, b)


def product(a, b):
    return a * b


def lukasiewicz(a, b):
    return np.fmax(0, a + b - 1)


def drastic_tnorm(a, b):
    if a >= 1.0:
        return b
    elif b >= 1.0:
        return a
    else:
        return 0.0


def hamacher(a, b):
    if a + b == 0:
        return 0.0
    return (a * b) / (a + b - a * b)


# --------------------- #
#   T - C O N O R M S   #
# --------------------- #

def maximum(a, b):
    return np.fmax(a, b)


def probsum(a, b):
    return a + b - a * b


def boundedsum(a, b):
    return np.fmin(1, a + b)


def drastic_tconorm(a, b):
    if a <= 0.0:
        return b
    elif b <= 0.0:
        return a
    else:
        return 1.0


def einstein(a, b):
    return (a + b) / (1 + a * b)
