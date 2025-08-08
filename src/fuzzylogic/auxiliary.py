import numpy as np


def _evaluate_tree(sets: dict, lvar):

    condition = next(iter(sets))
    children = [{child: sets[condition][child]} for child in sets[condition]]

    values = []
    for child in children:
        key = next(iter(child))
        if key in ['AND', 'OR']:
            value = _evaluate_tree(child, lvar)
            values.append(value)
        else:
            fuzzyset = next(iter(child.values()))
            value = fuzzyset(lvar)
            values.append(value)

    if condition == 'OR':
        # Combine values using maximum for 'or'
        result = np.fmax(*values)
    elif condition == 'AND':
        # Combine values using minimum for 'and'
        result = np.fmin(*values)

    return result
