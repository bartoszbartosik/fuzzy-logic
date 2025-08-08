import numpy as np


def trimf(abc, inverse=False):
    assert len(abc) == 3, 'abc must have 3 elements'
    a, b, c = abc

    if inverse:
        y_1 = lambda x: (b - x + 1e-10) / (b - a + 1e-10)
        y_2 = lambda x: (x - b + 1e-10) / (c - b + 1e-10)
        y_3 = lambda x: np.ones_like(x)
        return lambda x: np.min([np.max([y_1(x), y_2(x)], axis=0), y_3], axis=0)

    y_1 = lambda x: (x - a + 1e-10) / (b - a + 1e-10)
    y_2 = lambda x: (c - x + 1e-10) / (c - b + 1e-10)
    y_3 = lambda x: np.zeros_like(x)
    return lambda x: np.max([np.min([y_1(x), y_2(x)], axis=0), y_3(x)], axis=0)


def trapmf(abcd, inverse=False):
    assert len(abcd) == 4, 'abcd must have 4 elements'
    a, b, c, d = abcd

    if inverse:
        y_1 = lambda x: (b - x + 1e-10) / (b - a + 1e-10)
        y_2 = lambda x: np.zeros_like(x)
        y_3 = lambda x: (x - c + 1e-10) / (d - c + 1e-10)
        y_4 = lambda x: np.ones_like(x)
        return lambda x: np.min([np.max([y_1(x), y_2(x), y_3(x)], axis=0), y_4(x)], axis=0)

    y_1 = lambda x: (x - a + 1e-10) / (b - a + 1e-10)
    y_2 = lambda x: np.ones_like(x)
    y_3 = lambda x: (d - x + 1e-10) / (d - c + 1e-10)
    y_4 = lambda x: np.zeros_like(x)
    return lambda x: np.max([np.min([y_1(x), y_2(x), y_3(x)], axis=0), y_4(x)], axis=0)
