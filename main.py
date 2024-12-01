import numpy as np
from matplotlib import pyplot as plt, animation

from fuzzylogic import FuzzyInferenceSystem
from invertedpendulum import InvertedPendulum, plot
from fis import initialize_fis


def main():

    # ------------------ #
    # INITIAL CONDITIONS #
    # ------------------ #

    th = 40      # [deg]
    dth = 10.0     # [deg/s]
    x = 0.0       # [m]
    dx = 0.0     # [m/s]

    # State vector
    state = np.array([th, dth, x, dx])


    # ---------- #
    # SIMULATION #
    # ---------- #

    # Time
    dt = 0.02
    t_max = 2
    t = [0.0, t_max, dt]

    # Desired position
    x_pos = [
        [0, t_max],     # time [s]
        [0, 0]      # reference position [m]
        # [0, t_max],     # time [s]
        # [0, 0]      # reference position [m]
    ]

    inverted_pendulum = InvertedPendulum(initial_state=state, desired_position=x_pos, time_domain=t)

    fis = initialize_fis([th, dth], plot=True)

    aggs, centroids = inverted_pendulum.run(fis)
    plot(inverted_pendulum, fis, aggs, centroids, animate=True, save=True)


    # Animate fuzzy inferences over time
    # animate_aggregation(inverted_pendulum, fis, aggs, centroids, universe='force')


if __name__ == '__main__':
    main()
