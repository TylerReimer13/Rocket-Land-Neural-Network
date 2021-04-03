import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from rocket_landing_multiple_shooting import *

"""
This file will generate a specified number of random initial conditions for the rocket, and then call the multiple 
shooting script to attempt to solve for the optimal trajectory and controls for each condition. If an infeasible 
solution is obtained, the output is written as an array of '-999', so that it can be passed over in the training loop. 
Each initial condition, along with the optimal trajectory generated from it, is saved to an excel file for training.
"""


def time_map(this_dist):
    # Attempts to find a feasible end time for the solver based on the initial range from the goal state
    dists = [33.5, 75., 114.6, 150., 167.7,  225.]
    times = [9.5, 13.5,   15.,  16.,   17.,  18.]
    t = np.interp(this_dist, dists, times)
    return t


def sample(num_samples=250, rad=150., show_plot=False):
    # Uniformly samples starting points from the half circle around the goal position (0, 0)
    phi = np.random.uniform(size=num_samples) * np.pi
    r = np.sqrt(np.random.uniform(size=num_samples)) * rad

    x = r * np.cos(phi)
    y = r * np.sin(phi)

    if show_plot:
        f = plt.figure(figsize=(12, 12))
        a = f.add_subplot(111)
        a.scatter(x, y, marker='.')
        a.set_aspect('equal')
        plt.title('Uniformly Selected Intial Conditions')
        plt.grid()
        plt.xlabel('Downrange (meters)')
        plt.ylabel('Altitude (meters)')
        plt.show()

    return x, y


if __name__ == "__main__":
    # n_samples=500, nu=100 takes 16 minutes

    n_samples = 500  # Number of starting points
    radius = 150.  # Sampling radius (meters)
    x_start, y_start = sample(num_samples=n_samples, rad=radius, show_plot=True)

    start_mass = 100.  # Initial mass of rocket
    garbage_ctr = 0  # Initialize counter for infeasible solutions

    nu = 100  # Number of control points
    Init_full = -inf * np.ones((4, n_samples))  # Placeholder for initial conditions
    U_full = -inf * np.ones((nu * 2, n_samples))  # Placeholder for controls

    for i in range(n_samples):
        start = {'x': x_start[i], 'y': y_start[i], 'thet': 0., 'm': start_mass, 'x_dot': 0., 'y_dot': 0., 'thet_dot': 0.}
        end = {'x': 0., 'y': 0., 'thet': 0., 'x_dot': 0., 'y_dot': 0., 'thet_dot': 0.}

        dist = ((start['x'] - end['x']) ** 2 + (1.25 * (start['y'] - end['y']) ** 2)) ** .5
        t = time_map(dist)
        misc = {'T': t, 'nu': nu}

        guidance = TrajPlanner(start, end, misc)
        status = guidance(solver='euler', show_plot=False)  # Return solver status

        # If solution is feasible, and end state is within acceptable limits
        if status == 'Solve_Succeeded' and abs(guidance.x_opt[-1]) <= 5. and -.5 <= guidance.y_opt[-1] <= 5.:
            traj_init = np.array([x_start[i], y_start[i], t, nu]).flatten()

            traj_u1 = guidance.u1_opt
            traj_u2 = guidance.u2_opt
            traj_U = np.array([traj_u1, traj_u2]).flatten()

            Init_full[:, i] = traj_init  # Add initial conditions array to placeholder array
            U_full[:, i] = traj_U  # Add optimal control array to placeholder array

        # If infeasible, leave this array as '-999' so it can be filtered out later
        else:
            garbage_ctr += 1

    print('NUM GARBAGE: ', garbage_ctr)
    headers = ['Traj_' + str(i + 1) for i in range(n_samples)]

    # Store initial conditions for each trajectory, and write to excel
    init_df = pd.DataFrame(Init_full, columns=headers)
    init_df.to_excel('data/landing_data_init.xlsx', index=False)

    # Store optimal controls for each trajectory, and write to excel
    control_df = pd.DataFrame(U_full, columns=headers)
    control_df.to_excel('data/landing_data_control.xlsx', index=False)
