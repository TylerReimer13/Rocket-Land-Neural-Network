import numpy as np
from casadi import *
import matplotlib.pyplot as plt
from animate import animated_plot

"""
Finds the optimal trajectory and controls for a rocket with the specified intial conditions. These trajectories are
used as the training data for the neural network
"""


class TrajPlanner:
    def __init__(self, init, term, gen):
        self.T = gen['T'] if 'T' in gen.keys() else 25.  # Time horizon
        self.N = gen['nu'] if 'nu' in gen.keys() else 100  # number of control intervals
        self.grav = 9.81  # Earth gravity
        self.Isp = 100
        self.g0 = 9.81
        self.h = 70  # Falcon 9 height
        self.r = 3.7 / 2  # Falcon 9 radius
        self.Iz = None
        self.thrust_mag = 1000.

        # ---------Initial Conditions----------
        self.x0 = init['x'] if 'x' in init.keys() else -25.
        self.y0 = init['y'] if 'y' in init.keys() else 350.
        self.thet0 = init['thet'] if 'thet' in init.keys() else -3.
        self.m0 = init['m'] if 'm' in init.keys() else 10_000
        self.x_dot0 = init['x_dot'] if 'x_dot' in init.keys() else -5.
        self.y_dot0 = init['y_dot'] if 'y_dot' in init.keys() else -5.
        self.thet_dot0 = init['thet_dot'] if 'thet_dot' in init.keys() else -5.

        # ---------Final Conditions------------
        self.xf = term['x'] if 'x' in term.keys() else 0.
        self.yf = term['y'] if 'y' in term.keys() else 0.
        self.thetf = term['thet'] if 'thet' in term.keys() else 0.
        self.x_dotf = term['x_dot'] if 'x_dot' in term.keys() else 0.
        self.y_dotf = term['y_dot'] if 'y_dot' in term.keys() else 0.
        self.thet_dotf = term['thet_dot'] if 'thet_dot' in term.keys() else 0.

        # -------------Bounds--------------
        self.u1min = 0.
        self.u1max = 1.
        self.u2min = -.6
        self.u2max = .6

        self.xmin = -250.
        self.xmax = 250.
        self.ymin = 0.
        self.ymax = 1000.
        self.thetmin = -np.pi
        self.thetmax = np.pi
        self.x_dotmin = -inf
        self.x_dotmax = inf
        self.y_dotmin = -inf
        self.y_dotmax = inf
        self.thet_dotmin = -inf
        self.thet_dotmax = inf

        # Dynamical states
        self.NUM_STATES = 7
        self.NUM_INPUTS = 2
        self.x = SX.sym('x')
        self.y = SX.sym('y')
        self.thet = SX.sym('thet')
        self.m = SX.sym('m')
        self.x_dot = SX.sym('x_dot')
        self.y_dot = SX.sym('y_dot')
        self.thet_dot = SX.sym('thet_dot')

        self.x = vertcat(self.x, self.y, self.thet, self.m, self.x_dot, self.y_dot, self.thet_dot)

        # Controls
        self.u1 = SX.sym('u1')
        self.u2 = SX.sym('u2')
        self.u = vertcat(self.u1, self.u2)

        # Results
        self.x_calc = None
        self.y_calc = None

    def dynamics(self):
        self.Iz = self.m * (3 * (self.r ** 2) + (self.h ** 2)) / 12.

        xdot = self.x_dot
        ydot = self.y_dot
        thetdot = self.thet_dot
        mdot = -((self.u1 * self.thrust_mag) / (self.g0 * self.Isp))
        xddot = ((self.u1 * self.thrust_mag) * casadi.sin(self.thet + self.u2)) / self.m
        yddot = (((self.u1 * self.thrust_mag) * casadi.cos(self.thet + self.u2)) / self.m) - self.grav
        thetddot = (self.h * (self.u1 * self.thrust_mag) * casadi.sin(self.u2)) / (2 * self.Iz)

        return vertcat(xdot, ydot, thetdot, mdot, xddot, yddot, thetddot)

    def cost_func(self):
        return self.u1 ** 2 + self.u2 ** 2

    def solver(self, x_dot, method='euler'):
        f = Function('f', [self.x, self.u], [x_dot, self.cost_func()])  # Maps [x, u] -> [x_dot, cost_func]
        X0 = SX.sym('X0', self.NUM_STATES)
        U = SX.sym('U', self.NUM_INPUTS)
        X = X0
        Q = 0

        if method == 'euler':
            DT = self.T / self.N
            k1, k1_q = f(X, U)
            X = X + k1 * DT  # Integrated state ODE's
            Q = Q + k1_q * DT
            F = Function('F', [X0, U], [X, Q], ['x0', 'p'], ['xf', 'qf'])  # Maps initial states to final states (and Q)

        else:
            M = 4  # RK4 steps per interval
            DT = self.T / self.N / M
            for j in range(M):
                k1, k1_q = f(X, U)
                k2, k2_q = f(X + DT / 2 * k1, U)
                k3, k3_q = f(X + DT / 2 * k2, U)
                k4, k4_q = f(X + DT * k3, U)
                X = X + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)  # Integrated state ODE's
                Q = Q + DT / 6 * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
            F = Function('F', [X0, U], [X, Q], ['x0', 'p'], ['xf', 'qf'])  # Maps initial states to final states (and Q)

        return F

    def gen_constraints(self, F):
        w = []
        w0 = []  # Initial Guess for controls
        lbw = []  # Lower bound for controls
        ubw = []  # Upper bound for controls
        J = 0  # Cost function
        g = []  # Inequality constraint
        lbg = []  # Inequality constraint lower bound
        ubg = []  # Inequality constraint upper bound

        Xk = SX.sym('X0', self.NUM_STATES)
        w += [Xk]
        lbw += [self.x0, self.y0, self.thet0, self.m0, self.x_dot0, self.y_dot0, self.thet_dot0]
        ubw += [self.x0, self.y0, self.thet0, self.m0, self.x_dot0, self.y_dot0, self.thet_dot0]
        w0 += [self.x0, self.y0, self.thet0, self.m0, self.x_dot0, self.y_dot0, self.thet_dot0]

        Xk = SX([self.x0, self.y0, self.thet0, self.m0, self.x_dot0, self.y_dot0, self.thet_dot0])
        for k in range(self.N):
            # New NLP variable for the control
            Uk = SX.sym('U_' + str(k), self.NUM_INPUTS)

            w += [Uk[0]]
            lbw += [self.u1min]  # Control u1 lower bound
            ubw += [self.u1max]  # Control u1 upper bound
            w0 += [self.u1max]  # Control u1 guess

            w += [Uk[1]]
            lbw += [self.u2min]  # Control u2 lower bound
            ubw += [self.u2max]  # Control u2 upper bound
            w0 += [0.]  # Control u2 guess

            # Integrate till the end of the interval
            Fk = F(x0=Xk, p=Uk)
            Xk_end = Fk['xf']
            J = J + Fk['qf']

            # Final state bounds
            if k == self.N - 1:
                lbd_x = self.xf
                upd_x = self.xf

                lbd_y = self.yf
                upd_y = self.yf

                lbd_thet = self.thetf
                upd_thet = self.thetf

                lbd_x_dot = self.x_dotf
                upd_x_dot = self.x_dotf

                lbd_y_dot = self.y_dotf
                upd_y_dot = self.y_dotf

                lbd_thet_dot = self.thet_dotf
                upd_thet_dot = self.thet_dotf

            # Trajectory bounds
            else:
                lbd_x = self.xmin
                upd_x = self.xmax

                lbd_y = self.ymin
                upd_y = self.ymax

                lbd_thet = self.thetmin
                upd_thet = self.thetmax

                lbd_x_dot = self.x_dotmin
                upd_x_dot = self.x_dotmax

                lbd_y_dot = self.y_dotmin
                upd_y_dot = self.y_dotmax

                lbd_thet_dot = self.thet_dotmin
                upd_thet_dot = self.thet_dotmax

            # New NLP variable for state at end of interval
            Xk = SX.sym('X_' + str(k + 1), self.NUM_STATES)
            w += [Xk]
            lbw += [lbd_x, lbd_y, lbd_thet, 0., lbd_x_dot, lbd_y_dot, lbd_thet_dot]
            ubw += [upd_x, upd_y, upd_thet, self.m0, upd_x_dot, upd_y_dot, upd_thet_dot]
            w0 += [0, 0, 0, 0, 0, 0, 0]

            # Add equality constraint
            g += [Xk_end - Xk]
            lbg += [0, 0, 0, 0, 0, 0, 0]
            ubg += [0, 0, 0, 0, 0, 0, 0]

        print(type(Xk[4]))
        return w, w0, lbw, ubw, J, g, lbg, ubg

    def __call__(self, solver='euler', show_plot=False):
        xdot = self.dynamics()
        F = self.solver(xdot, solver)

        w, w0, lbw, ubw, J, g, lbg, ubg = self.gen_constraints(F)

        prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
        solver = nlpsol('solver', 'ipopt', prob)

        # Solve the NLP
        # w0 = Initial Guess for controls (u)
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

        w_opt = sol['x'].full().flatten()

        # Plot the solution
        self.x_opt = w_opt[0::self.NUM_STATES + self.NUM_INPUTS]
        self.y_opt = w_opt[1::self.NUM_STATES + self.NUM_INPUTS]
        self.thet_opt = w_opt[2::self.NUM_STATES + self.NUM_INPUTS]
        self.m_opt = w_opt[3::self.NUM_STATES + self.NUM_INPUTS]
        self.x_dot_opt = w_opt[4::self.NUM_STATES + self.NUM_INPUTS]
        self.y_dot_opt = w_opt[5::self.NUM_STATES + self.NUM_INPUTS]
        self.thet_dot_opt = w_opt[6::self.NUM_STATES + self.NUM_INPUTS]

        self.u1_opt = w_opt[7::self.NUM_STATES + self.NUM_INPUTS]
        self.u2_opt = w_opt[8::self.NUM_STATES + self.NUM_INPUTS]

        self.dt = self.T / self.N

        if show_plot:
            self.plot()

        return solver.stats()['return_status']

    def plot(self):
        states_time = [n * self.dt for n in range(self.N + 1)]
        controls_time = [n * self.dt for n in range(self.N)]

        plt.title('States')
        plt.plot(states_time, self.x_opt, label='x')
        plt.plot(states_time, self.y_opt, label='y')
        plt.plot(states_time, self.thet_opt, label='thet')
        plt.plot(states_time, self.m_opt, label='m')
        plt.plot(states_time, self.x_dot_opt, label='x_dot')
        plt.plot(states_time, self.y_dot_opt, label='y_dot')
        plt.plot(states_time, self.thet_dot_opt, label='thet_dot')

        plt.legend()
        plt.grid()
        plt.xlabel('Time (s)')
        plt.ylabel('States')
        plt.show()

        plt.title('Inputs')
        plt.plot(controls_time, self.u1_opt, label='u1')
        plt.plot(controls_time, self.u2_opt, label='u2')
        plt.legend()
        plt.grid()
        plt.xlabel('Time (s)')
        plt.ylabel('Control Effort')
        plt.show()

        plt.title('Trajectory')
        plt.plot(self.x_opt, self.y_opt, label='X-Z')
        plt.legend()
        plt.grid()
        plt.xlabel('Downrange (m)')
        plt.ylabel('Altitude (m)')
        plt.show()


if __name__ == "__main__":
    mass = 100.
    start = {'x': -100., 'y': 150., 'thet': 0., 'm': mass, 'x_dot': 0., 'y_dot': 0., 'thet_dot': 0.}
    end = {'x': 0., 'y': 0., 'thet': 0., 'x_dot': 0., 'y_dot': 0., 'thet_dot': 0.}

    dist = ((start['x'] - end['x']) ** 2 + (1.25 * (start['y'] - end['y']) ** 2)) ** .5
    print('DIST: ', dist)
    misc = {'T': 18., 'nu': 100}

    guidance = TrajPlanner(start, end, misc)
    status = guidance(solver='euler', show_plot=True)

    print('------------STATUS: ', status)
    print('FINAL MASS: ', guidance.m_opt[-1])

    animated_plot(guidance.x_opt, guidance.y_opt, guidance.thet_opt, guidance.u1_opt, guidance.u2_opt, misc['nu'])
