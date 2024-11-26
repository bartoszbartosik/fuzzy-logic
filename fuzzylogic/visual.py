import numpy as np
from matplotlib import pyplot as plt

from fuzzylogic import FuzzyInferenceSystem


def plot_membership_functions(fis: FuzzyInferenceSystem):
    fig, axs = plt.subplots(len(fis.universes), 1, figsize=(12, len(fis.universes) * 5))
    for i, (key, value) in enumerate(fis.universes.items()):
        axs[i].set_title(f'{key} memberships')
        axs[i].set_xlabel(key)
        axs[i].set_ylabel(r'$\mu$')
        for name, mf in fis.memfuncs[key].items():
            axs[i].plot(fis.universes[key], mf['f'](fis.universes[key]), label=name)
        axs[i].legend()
    plt.show()


def plot_aggregation(fis: FuzzyInferenceSystem, x: np.ndarray, universe: str):
    assert universe in fis.universes.keys(), 'Universe not found'

    # Aggregate
    u, agg = fis.aggregate(x, universe)

    # Inference
    centroid = fis.defuzz(u, agg)

    # Plot
    plt.fill_between(u, 0, agg, alpha=0.5, color='tab:blue')
    # Plot centroid
    plt.plot([centroid, centroid], [0, np.max(agg)], '--', color='tab:blue')
    # Show centroid value
    plt.text(centroid * 0.8, np.max(agg) * 0.9, f'{centroid:.2f}', color='tab:blue')
    # Plot membership functions
    for name, mf in fis.memfuncs[universe].items():
        plt.plot(u, mf['f'](u), '--', label=name, alpha=0.5, color='tab:orange')
    plt.xlabel(universe)
    plt.ylabel(r'$\mu$')
    plt.show()
