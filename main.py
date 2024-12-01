import numpy as np
from matplotlib import pyplot as plt, animation

from fuzzylogic import FuzzyInferenceSystem
from invertedpendulum import InvertedPendulum, plot
from fis import initialize_fis


def animate_aggregation(ip, fis, aggs, centroids, universe):
    fig, ax = plt.subplots()

    def init():
        ax.set_xlim(fis.universes[universe][0], fis.universes[universe][-1])
        ax.set_ylim(0, 1)
        return []

    def animate(i):
        ax.clear()
        ax.set_xlim(fis.universes[universe][0], fis.universes[universe][-1])
        ax.set_ylim(0, 1)

        u = fis.universes[universe]
        agg = aggs[i]
        centroid = centroids[i]

        ax.fill_between(u, 0, agg, alpha=0.5, color='tab:blue')
        ax.plot([centroid, centroid], [0, np.max(agg)], '--', color='tab:blue')
        ax.text(centroid * 0.8, np.max(agg) * 0.9, f'{centroid:.2f}', color='tab:blue')

        for name, mf in fis.memfuncs[universe].items():
            ax.plot(u, mf['f'](u), '--', label=name, alpha=0.5, color='tab:orange')

        ax.set_xlabel(universe)
        ax.set_ylabel(r'$\mu$')
        return []

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=range(1, len(ip.t)),
                                       interval=ip.t[-1] / len(ip.t) * 1000, blit=False)
    writergif = animation.PillowWriter(fps=len(ip.t) / ip.t[-1])
    anim.save("anims/anim_fis.gif", writer=writergif)


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
    plot(inverted_pendulum, animate=True, save=True)


    # Animate fuzzy inferences over time
    animate_aggregation(inverted_pendulum, fis, aggs, centroids, universe='force')


if __name__ == '__main__':
    main()
