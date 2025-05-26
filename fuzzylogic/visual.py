import numpy as np
from matplotlib import pyplot as plt

from fuzzylogic import FuzzyInferenceSystem
from .structs import Universe


def plot_membership_functions(fis: FuzzyInferenceSystem):
    fig, axs = plt.subplots(len(fis.universes), 1, figsize=(12, len(fis.universes) * 5))
    for i, universe in enumerate(fis.universes):
        axs[i].set_title(f'{universe.name} memberships')
        axs[i].set_xlabel(universe.name)
        axs[i].set_ylabel(r'$\mu$')
        for term, mu in universe.terms:
            axs[i].plot(universe.domain, mu(universe.domain), label=term)
        axs[i].legend()
    plt.show()


def plot_aggregation(fis: FuzzyInferenceSystem, x: np.ndarray, universe: Universe):
    # Aggregate
    agg = fis.aggregate(x, universe.domain)

    # Inference
    centroid = fis.defuzz(agg, universe.domain)

    # Plot
    plt.fill_between(universe.domain, 0, agg, alpha=0.5, color='tab:blue')
    # Plot centroid
    plt.plot([centroid, centroid], [0, np.max(agg)], '--', color='tab:blue')
    # Show centroid value
    plt.text(centroid * 0.8, np.max(agg) * 0.9, f'{centroid:.2f}', color='tab:blue')
    # Plot membership functions
    for term, mu in universe.terms:
        plt.plot(universe.domain, mu(universe.domain), '--', label=term, alpha=0.5, color='tab:orange')
    plt.xlabel(universe.name)
    plt.ylabel(r'$\mu$')
    plt.show()
