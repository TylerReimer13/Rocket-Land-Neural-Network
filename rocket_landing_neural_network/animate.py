import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

update_itr = 0


def animated_plot(x_pos, y_pos, ang, u1, u2, n_timesteps):
    fig = plt.figure()
    ax = plt.axes(xlim=(-100., 100.),
                  ylim=(-10., 190.))

    body_width = 8.5
    body_height = 22.5

    rocket = patches.Rectangle((0, 0), 0, 0, fc='black')
    engine = patches.Rectangle((0, 0), 0, 0, fc='r')

    def init():
        ax.add_patch(rocket)
        ax.add_patch(engine)
        return (rocket,) + (engine,)

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

        engine.set_width(body_width / 4)
        engine.set_height(-u1[update_itr] * 15.)
        engine.set_xy([body_x - (body_width / 8), body_y])
        engine.angle = -np.rad2deg(ang[update_itr] - u2[update_itr])

        update_itr += 1
        return (traj,) + (rocket,) + (engine,)

    traj, = plt.plot([], [], 'g--', alpha=0.)
    plt.plot([-1000., 1000.], [-23., -23.], 'g-', linewidth=50.)
    ax.set_facecolor('xkcd:sky blue')
    plt.xlabel('Downrange (m)')
    plt.ylabel('Altitude (m)')

    # Larger interval = slower animation
    ani = animation.FuncAnimation(fig, update, init_func=init, frames=int(n_timesteps), interval=70, repeat=True)
    ani.save("rocket_land.gif")
    plt.show()


