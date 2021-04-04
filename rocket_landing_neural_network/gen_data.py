import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from rocket_landing_multiple_shooting_free_time import *

"""
This file will generate a specified number of random initial conditions for the rocket, and then call the multiple 
shooting script to attempt to solve for the optimal trajectory and controls for each condition. If an infeasible 
solution is obtained, the output is written as an array of '-999', so that it can be passed over in the training loop. 
Each initial condition, along with the optimal trajectory generated from it, is saved to an excel file for training.
"""


def circle_sample(num_samples=250, rad=150., show_plot=False):
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
        plt.title('Uniformly Selected Initial Conditions')
        plt.grid()
        plt.xlabel('Downrange (meters)')
        plt.ylabel('Altitude (meters)')
        plt.show()

    return x, y


def box_sample(num_samples=250, x_bds=(), y_bds=(), thet_bds=(), m_bds=(), vel_x_bds=(), vel_y_bds=(),
               thet_dot_bds=(), show_plot=False):

    x = np.random.uniform(x_bds[0], x_bds[1], size=num_samples)
    y = np.random.uniform(y_bds[0], y_bds[1], size=num_samples)
    thet = np.random.uniform(thet_bds[0], thet_bds[1], size=num_samples)
    m = np.random.uniform(m_bds[0], m_bds[1], size=num_samples)
    vel_x = np.random.uniform(vel_x_bds[0], vel_x_bds[1], size=num_samples)
    vel_y = np.random.uniform(vel_y_bds[0], vel_y_bds[1], size=num_samples)
    thet_dot = np.random.uniform(thet_dot_bds[0], thet_dot_bds[1], size=num_samples)

    if show_plot:
        f = plt.figure(figsize=(12, 12))
        a = f.add_subplot(111)
        a.scatter(x, y, marker='.')
        a.set_aspect('equal')
        plt.title('Uniformly Selected Initial Conditions')
        plt.grid()
        plt.xlabel('Downrange (meters)')
        plt.ylabel('Altitude (meters)')
        plt.savefig('results/Initial Conditions')
        plt.show()

    return x, y, thet, m, vel_x, vel_y, thet_dot


if __name__ == "__main__":
    n_samples = 8500  # Number of starting points
    # radius = 150.  # Sampling radius (meters)
    # x_start, y_start = circle_sample(num_samples=n_samples, rad=radius, show_plot=True)

    x_rng = (-100., 100.)
    y_rng = (2., 180.)
    thet_rng = (-.35, .35)
    m_rng = (75., 100.)
    vx_rng = (-7.5, 7.5)
    vy_rng = (-25., 5.)
    thet_dot_rng = (-.15, .15)
    xr, yr, tr, mr, vxr, vyr, tdr = box_sample(n_samples, x_rng, y_rng, thet_rng, m_rng, vx_rng, vy_rng, thet_dot_rng,
                                               show_plot=True)

    garbage_ctr = 0  # Initialize counter for infeasible solutions

    nu = 100  # Number of control points
    Init_full = -inf * np.ones((9, n_samples))  # Placeholder for initial conditions
    U_full = -inf * np.ones((nu * 2, n_samples))  # Placeholder for controls

    for i in range(n_samples):
        start = {'x': xr[i], 'y': yr[i], 'thet': tr[i], 'm': mr[i], 'x_dot': vxr[i], 'y_dot': vyr[i],
                 'thet_dot': tdr[i]}
        end = {'x': 0., 'y': 0., 'thet': 0., 'x_dot': 0., 'y_dot': 0., 'thet_dot': 0.}

        misc = {'nu': nu}

        guidance = TrajPlanner(start, end, misc)
        status = guidance(show_plot=False)  # Return solver status

        # If solution is feasible, and end state is within acceptable limits
        if status == 'Solve_Succeeded' and abs(guidance.x_opt[-1]) <= 2. and -.5 <= guidance.y_opt[-1] <= 2.:
            traj_init = np.array([xr[i], yr[i], tr[i], mr[i], vxr[i], vyr[i], tdr[i], guidance.DT_opt, nu]).flatten()

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
