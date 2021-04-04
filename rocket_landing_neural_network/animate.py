import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from math import sin, cos, asin, acos, atan

update_itr = 0


def animated_plot(x_pos, y_pos, ang, u1, u2, n_timesteps):
    fig = plt.figure()
    ax = plt.axes(xlim=(-100., 100.),
                  ylim=(-10., 190.))

    body_width = 8.5
    body_height = 22.5

    rocket = patches.Rectangle((0, 0), 0, 0, fc='gray')
    left_leg = patches.Rectangle((0, 0), 0, 0, fc='black')
    right_leg = patches.Rectangle((0, 0), 0, 0, fc='black')
    engine = patches.Rectangle((0, 0), 0, 0, fc='r')
    engine2 = patches.Rectangle((0, 0), 0, 0, fc='yellow')
    base = patches.Rectangle((0, 0), 0, 0, fc='black')

    def init():
        ax.add_patch(rocket)
        ax.add_patch(left_leg)
        ax.add_patch(right_leg)
        ax.add_patch(engine)
        ax.add_patch(engine2)
        ax.add_patch(base)
        return (rocket,) + (left_leg,) + (right_leg,) + (engine,) + (engine2,) + (base,)

    def update(i):
        global update_itr

        body_x = x_pos[update_itr]
        body_y = y_pos[update_itr]

        update_itr = update_itr % n_timesteps

        traj.set_data([x_pos, y_pos])

        rocket.set_width(body_width)
        rocket.set_height(body_height)
        rocket.set_xy([body_x - (body_width / 2), body_y])
        rocket.angle = -np.rad2deg(ang[update_itr])

        left_leg.set_width(4.)
        left_leg.set_height(2.)
        left_leg.set_xy([body_x - (body_width / 1.5), body_y])
        left_leg.angle = rocket.angle + 250.

        right_leg.set_width(4.)
        right_leg.set_height(2.)
        right_leg.set_xy([body_x + (body_width / 2.5), body_y + 10. * atan(np.deg2rad(rocket.angle))])
        right_leg.angle = rocket.angle - 70.

        engine.set_width(body_width / 4)
        engine.set_height(-u1[update_itr] * 12.)
        engine.set_xy([body_x - (body_width / 8), body_y + cos(np.deg2rad(rocket.angle))])
        engine.angle = -np.rad2deg(ang[update_itr] - u2[update_itr])

        engine2.set_width(body_width / 6.5)
        engine2.set_height(-u1[update_itr] * 8.)
        engine2.set_xy([body_x - (body_width / 8) + .45, body_y + cos(np.deg2rad(rocket.angle))])
        engine2.angle = engine.angle

        base.set_width(11.25)
        base.set_height(2.5)
        base.set_xy([body_x - (body_width / 1.5), body_y])
        base.angle = rocket.angle

        update_itr += 1
        return (traj,) + (rocket,) + (left_leg,) + (right_leg,) + (engine,) + (engine2,) + (base,)

    traj, = plt.plot([], [], 'g--', alpha=0.)
    plt.plot([-1000., 1000.], [-23., -23.], 'g-', linewidth=50.)
    ax.set_facecolor('xkcd:sky blue')
    plt.xlabel('Downrange (m)')
    plt.ylabel('Altitude (m)')

    # Larger interval = slower animation
    ani = animation.FuncAnimation(fig, update, init_func=init, frames=int(n_timesteps), interval=70, repeat=True)
    ani.save("results/rocket_land_neural_net.gif")
    plt.show()


