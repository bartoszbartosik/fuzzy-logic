import itertools
import os

from matplotlib import gridspec, pyplot as plt, animation
from matplotlib.patches import Rectangle
from numpy import pi, sin, cos


# VISUALIZE SIMULATION ON A GRAPH
def plot(model, animate, save):

    # ASSIGN SELF.SOLUTION AND SELF.L AS LOCAL VARIABLES
    solution = model.solution
    L = model.L

    # GET PENDULUM'S ANGLE
    ths = solution[0]                            # [rad]
    thsdeg = [rad * 180.0 / pi for rad in ths]   # [deg]

    # GET PENDULUM'S ANGULAR VELOCITY
    dths = solution[1]                           # [rad/s]
    dthsdeg = [rad * 180.0 / pi for rad in ths]  # [deg/s]

    # GET PENDULUM'S POSITION
    xs = solution[2]                             # [m]

    # GET PENDULUM'S VELOCITY
    dxs = solution[3]                            # [m/s]

    # CALCULATE PENDULUM'S COORDINATES
    x_pos = L * sin(ths) + xs
    y_pos = L * cos(ths)

    model.xlim = (x_pos[0] - L * 1.25, x_pos[0] + L * 1.25)
    ylim = (-L * 1.25, L * 1.25)

    # IF ANIMATE IS TRUE, VISUALISE PENDULUM'S MOVEMENT IN X-Y COORDINATE SYSTEM
    if animate:

        # DEFINE 4x2 [rows x columns] SUBPLOTS GRID
        gs = gridspec.GridSpec(4, 2, width_ratios=[1, 1],
                               height_ratios=[1, 1, 1, 1])

        # CREATE A FIGURE
        fig = plt.figure(figsize=(15, 15), dpi=80)

        # CREATE X-Y SUBPLOT WITH VISUALISATION OF INVERTED PENDULUM
        ax = fig.add_subplot(gs[:, 0])
        ax.set_xlim(model.xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.grid()
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

        # Plot pendulum
        line_pendulum, = ax.plot([], [], 'o-', color='0.3', lw=3)
        # Plot cart
        cart = ax.add_patch(Rectangle((0, 0),
                                      width=model.cart_width,
                                      height=model.cart_height,
                                      facecolor='0.6',
                                      edgecolor='0'))

        # Plot desired position
        x_des_plot, = ax.plot([], [], '--', color='0.3', lw=1)

        # Create text templates
        time_template = 'time = %.1f s'
        time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)
        theta_template = 'theta = %.1f\N{DEGREE SIGN}'
        theta_text = ax.text(0.3, 0.95, '', transform=ax.transAxes)
        xdes_template = 'x_des = %.1f m'
        xdes_text = ax.text(0.7, 0.95, '', transform=ax.transAxes)

        # CREATE THETA SUBPLOT
        axth = fig.add_subplot(gs[0, 1])
        axth.grid()
        axth.set_xlabel('t [s]')
        axth.set_ylabel('\u03B8 [\N{DEGREE SIGN}]')
        axth.plot(model.t, thsdeg, '-', color='0.3', lw=3)
        # Plot theta reference
        axth.plot([0, model.t_max], [0, 0], '--', color='0.3', lw=1)

        # CREATE POSITION SUBPLOT
        axx = fig.add_subplot(gs[1, 1])
        axx.grid()
        axx.set_xlabel('t [s]')
        axx.set_ylabel('x [m]')
        axx.plot(model.t, xs, '-', color='0.3', lw=3)
        # Plot x reference
        t_ref = [[0]]
        t_ref.extend([[model.x_ref[0][i], model.x_ref[0][i]] for i in range(len(model.x_ref[0]) - 1)])
        t_ref = list(itertools.chain.from_iterable(t_ref))
        t_ref.append(model.t_max)

        x_ref = []
        x_ref.extend([[model.x_ref[1][i], model.x_ref[1][i]] for i in range(len(model.x_ref[1]))])
        x_ref = list(itertools.chain.from_iterable(x_ref))
        axx.plot(t_ref, x_ref, '--', color='0.3', lw=1)

        # CREATE ANGULAR VELOCITY SUBPLOT
        axdth = fig.add_subplot(gs[2, 1])
        axdth.grid()
        axdth.set_xlabel('t [s]')
        axdth.set_ylabel('\u03C9 [1/s]')
        axdth.plot(model.t, dths, '-', color='0.3', lw=3)

        # CREATE VELOCITY SUBPLOT
        axdx = fig.add_subplot(gs[3, 1])
        axdx.grid()
        axdx.set_xlabel('t [s]')
        axdx.set_ylabel('v [m/s]')
        axdx.plot(model.t, dxs, '-', color='0.3', lw=3)

        # IF PLOT IS ABOUT TO BE SAVED, SET PROPER blit PARAMETER FOR ANIMATION FUNCTION
        if save:
            blit = False
        else:
            blit = True

        # INITIALIZE INVERTED PENDULUM'S X-Y SUBPLOT
        def init():
            line_pendulum.set_data([], [])
            cart.set_xy(([], []))
            time_text.set_text('')
            theta_text.set_text('')
            xdes_text.set_text('')

            return line_pendulum, cart, time_text, theta_text, xdes_text

        # ANIMATE FUNCTION
        def animate(i):
            # GET VARIABLES
            thist = model.t[i]                           # current time
            thisth = ths[i]                             # current theta
            thisx = [xs[i], x_pos[i]]                   # current pendulum's (x, y)
            thisy = [0, y_pos[i]]                       # current cart's (x, y)
            thisdx = dxs[i]                             # current cart's velocity
            thisxdes = model.step(thist)                 # current x_des

            # UPDATE INVERTED PENDULUM'S POSITION
            line_pendulum.set_data(thisx, thisy)
            cart.set_xy((xs[i] - model.cart_width / 2, -model.cart_height / 2))

            # UPDATE INVERTED PENDULUM'S X-Y PLOT LIMIT
            if xs[i] + 1.25 * model.cart_width / 2 > model.xlim[1]:
                model.xlim = (xs[i] + 1.25 * model.cart_width / 2 - 2* L * 1.25, xs[i] + 1.25 * model.cart_width / 2)
                ax.set_xlim(model.xlim)
            elif xs[i] - 1.25 * model.cart_width / 2 < model.xlim[0]:
                model.xlim = (xs[i] - 1.25 * model.cart_width / 2, xs[i] - 1.25 * model.cart_width / 2 + 2 * L * 1.25)
                ax.set_xlim(model.xlim)

            # UPDATE TEXT LABELS
            time_text.set_text(time_template % thist)
            theta_text.set_text(theta_template % (thisth * 180.0 / pi))
            xdes_text.set_text(xdes_template % thisxdes)

            # UPDATE DESIRED POSITION
            x_des_plot.set_data([thisxdes, thisxdes, thisxdes], [-2 * L, 0, 2 * L])

            # UPDATE PLOT'S LIMITS
            axth.set_xlim(left=0, right=thist)
            axx.set_xlim(left=0, right=thist)
            axdth.set_xlim(left=0, right=thist)
            axdx.set_xlim(left=0, right=thist)

            return line_pendulum, cart, time_text, theta_text, xdes_text, ax, axth, axx, axdth, axdx, x_des_plot

        # INVOKE ANIMATION FUNCTION
        anim = animation.FuncAnimation(fig, animate, frames=range(1, len(model.t)),
                                       interval=model.t[-1] / len(model.t) * 1000, blit=blit, init_func=init)

        # IF SAVE IS TRUE
        if save:
            # SAVE ANIMATION AS GIF
            writergif = animation.PillowWriter(fps=len(model.t) / model.t[-1])
            os.makedirs('../anims', exist_ok=True)
            anim.save("anims/anim_ip.gif", writer=writergif)
        else:
            # SHOW THE PLOT IN A WINDOW
            plt.show()

    # IF ANIMATION IS FALSE
    else:

        # DEFINE 4x1 [rows x columns] SUBPLOTS GRID
        gs = gridspec.GridSpec(4, 1, width_ratios=[1],
                               height_ratios=[1, 1, 1, 1])

        # CREATE A FIGURE
        fig = plt.figure(figsize=(10, 15), dpi=80)

        # CREATE THETA SUBPLOT
        axth = fig.add_subplot(gs[0, 0])
        axth.grid()
        axth.set_xlim(left=0, right=model.t_max)
        axth.set_xlabel('t [s]')
        axth.set_ylabel('\u03B8 [\N{DEGREE SIGN}]')
        axth.plot(model.t, thsdeg, '-', color='0.3', lw=3)
        # Plot theta reference
        axth.plot([0, model.t_max], [0, 0], '--', color='0.3', lw=1)

        # CREATE POSITION SUBPLOT
        axx = fig.add_subplot(gs[1, 0])
        axx.grid()
        axx.set_xlim(left=0, right=model.t_max)
        axx.set_xlabel('t [s]')
        axx.set_ylabel('x [m]')
        axx.plot(model.t, xs, '-', color='0.3', lw=3)
        # Plot x reference
        t_ref = [[0]]
        t_ref.extend([[model.x_ref[0][i], model.x_ref[0][i]] for i in range(len(model.x_ref[0]) - 1)])
        t_ref = list(itertools.chain.from_iterable(t_ref))
        t_ref.append(model.t_max)

        x_ref = []
        x_ref.extend([[model.x_ref[1][i], model.x_ref[1][i]] for i in range(len(model.x_ref[1]))])
        x_ref = list(itertools.chain.from_iterable(x_ref))
        axx.plot(t_ref, x_ref, '--', color='0.3', lw=1)

        # CREATE ANGULAR VELOCITY SUBPLOT
        axdth = fig.add_subplot(gs[2, 0])
        axdth.grid()
        axdth.set_xlim(left=0, right=model.t_max)
        axdth.set_xlabel('t [s]')
        axdth.set_ylabel('\u03C9 [1/s]')
        axdth.plot(model.t, dths, '-', color='0.3', lw=3)

        # CREATE VELOCITY SUBPLOT
        axdx = fig.add_subplot(gs[3, 0])
        axdx.grid()
        axdx.set_xlim(left=0, right=model.t_max)
        axdx.set_xlabel('t [s]')
        axdx.set_ylabel('v [m/s]')
        axdx.plot(model.t, dxs, '-', color='0.3', lw=3)

        # IF SAVE IS TRUE
        if save:
            # SAVE FIGURE TO A FILE
            plt.savefig('plots/plot.png')
        else:
            # SHOW THE PLOT IN A WINDOW
            plt.show()