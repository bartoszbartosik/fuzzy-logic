import numpy as np

from scipy import integrate
from numpy import sin, cos, pi

from fuzzylogic.fis import FuzzyInferenceSystem
from fuzzylogic.visual import plot_aggregation


class InvertedPendulum:

    # CONSTRUCTOR
    def __init__(self,
                 initial_state,
                 desired_position,
                 time_domain,
                 pendulum_length=0.8,
                 pendulum_mass=0.3,
                 pendulum_friction=0.05,
                 cart_mass=0.6,
                 cart_friction=0.1,
                 g=-9.81):

        # PENDULUM'S PARAMETERS
        self.L = pendulum_length
        self.m = pendulum_mass
        self.b = pendulum_friction
        self.I = 1 / 3 * self.m * self.L ** 2

        # CART'S PARAMETERS
        self.cart_width = 0.5 * self.L
        self.cart_height = 0.3 * self.cart_width
        self.M = cart_mass
        self.B = cart_friction

        # INITIAL STATE
        self.state = initial_state
        self.state[0] = self.th = initial_state[0] * pi / 180
        self.state[1] = self.dth = initial_state[1] * pi / 180
        self.state[2] = self.x = initial_state[2]
        self.state[3] = self.dx = initial_state[3]

        # DESIRED POSITION
        self.x_ref = desired_position

        # GRAVITY
        self.g = g

        # SIMULATION TIME
        self.t_0 = time_domain[0]
        self.t_max = time_domain[1]
        self.dt = time_domain[2]
        self.t = np.arange(self.t_0, self.t_max, self.dt)

        # CONTROLLER COEFFICIENTS
        # self.K = [41, 19, 1.5, 3.8]
        # self.K = [50, 20, 3.1, 4.8]
        self.K = [0, 0, 0, 0]

        # SOLUTION
        self.solution = None

        # X LIMIT FOR VISUALISATION PLOT
        self.xlim = (-1.25*self.L, 1.25*self.L)


    # DEFINE DESIRED CART POSITION(S)
    def step(self, t):

        # Create timestamps array
        timestamps = self.x_ref[0]

        # Create positions array
        positions = self.x_ref[1]

        # Check position corresponding to current time
        for i in range(len(timestamps)):
            if t <= timestamps[i]:
                return positions[i]
            if timestamps[i] < t <= timestamps[i+1]:
                return positions[i+1]
            elif t >= self.t_max:
                return positions[-1]
            else:
                continue


    # SOLVE PENDULUM'S DIFFERENTIAL EQUATION AND STORE RESULTS
    def run(self, fis: FuzzyInferenceSystem):
        aggs = []
        Fs = []
        termination_counter = []
        # FUNCTION COMPUTING STATE VARIABLES VALUES IN A SINGLE TIMESTAMPS
        def derivs(t, state):

            # Create array of zeros of a self.state size
            dsdt = np.zeros_like(state)

            # Assign proper state variables to each state vector indexes
            th = state[0]
            dth = state[1]
            x = state[2]
            dx = state[3]

            # Check the desired position
            x0 = self.step(t)

            # Calculate a value of force needed to stabilize the pendulum at a desired x0 position
            # F = self.K[0] * th + self.K[1] * dth + self.K[2] * (x - x0) + self.K[3] * dx
            fis_x = np.array([np.rad2deg(th), dth]), 'force'
            u, agg = fis.aggregate(*fis_x)
            F = fis.defuzz(u, agg)

            aggs.append(agg)
            Fs.append(F)

            print(f't: {t}, F: {F}, ang: {np.rad2deg(th)}, ang_vel: {dth}')

            # Calculate state variables
            dsdt[0] = dth
            dsdt[1] = (2*self.B*self.L*dx*self.m*cos(th) - 2*F*self.L*self.m*cos(th) - self.L**2*dth**2*self.m**2*sin(2*th)/2 - 2*self.L*self.M*self.g*self.m*sin(th) -
                       2*self.L*self.g*self.m**2*sin(th) - 4*self.M*self.b*dth - 4*self.b*dth*self.m)/(4*self.I*self.M + 4*self.I*self.m + self.L**2*self.M*self.m + self.L**2*self.m**2*sin(th)**2)
            dsdt[2] = dx
            dsdt[3] = (-4*self.B*self.I*dx - self.B*self.L**2*dx*self.m + 4*F*self.I + F*self.L**2*self.m + 2*self.I*self.L*dth**2*self.m*sin(th) + self.L**3*dth**2*self.m**2*sin(th)/2
                       + self.L**2*self.g*self.m**2*sin(2*th)/2 + 2*self.L*self.b*dth*self.m*cos(th))/(4*self.I*self.M + 4*self.I*self.m + self.L**2*self.M*self.m + self.L**2*self.m**2*sin(th)**2)

            # Limit force to prevent values from running up to infinity
            termination_counter.append(True)

            return dsdt

        # SIMULATION STOP CONDITION
        def limit(t, state):
            # If pendulum's angle is less than 90 deg, stop simulation
            if len(termination_counter) > 999000:
                state[0] = 0
                return state[0]
            # Otherwise keep performing it
            else:
                return 1

        # If true, terminate integration after event (limit function) occurs
        limit.terminal = True

        # Solve pendulum's differential equation
        sol = integrate.solve_ivp(derivs, method='LSODA', y0=self.state, t_span=[0, self.t_max], min_step=0.001, max_step=self.dt, events=limit)

        # Store time domain and solution values
        self.t = sol.t
        self.solution = sol.y

        return aggs, Fs


    # GET ANGULAR POSITION INTEGRAL OVER TIME
    def get_theta_integral(self):

        # Make an absolute out of each value in solved theta array
        ths = [abs(x) for x in self.solution[0]]

        # Initialize integral
        sum = 0
        # itheta = []

        # Integrate
        for i in range(len(ths)):
            if i < len(ths)-1:
                dt = self.t[i+1] - self.t[i]

            sum += dt * ths[i]
            # itheta.append(sum)

        # plt.plot(self.t, itheta)
        # plt.grid()
        # plt.draw()

        return sum


    # GET CART'S POSITION INTEGRAL OVER TIME
    def get_x_integral(self):

        # Get solved positions array
        xs = self.solution[2]

        # Make an absolute out of each value in solved x array and subtracted desired position value
        xs_ref = [abs(xs[i] - self.step(self.t[i])) for i in range(len(xs))]

        # Initialize integral
        sum = 0
        # ix = []

        # Integrate
        for i in range(len(xs)):
            if i < len(xs_ref)-1:
                dt = self.t[i+1] - self.t[i]
            sum += dt * xs_ref[i]
            # ix.append(sum)

        # plt.plot(self.t, ix)
        # plt.grid()
        # plt.draw()

        return sum
