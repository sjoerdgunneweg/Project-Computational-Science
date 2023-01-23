"""
Authors:     Sjoerd Gunneweg; Rinji Le; Pjotr Piet
ID:          13133330; 13344552; 12714933
Date:        20-01-2020
Description:
This file contains the logic for implementing the boids algorithm.
The different rules are implemented in the simulate function.
Every boid is represented as a 1d array with different values for 1: position,
2: speed vector. The different rules are implemented according to the rules
notes in Shiflet and Shiflet. We also implemented a way to simulate all the
different time steps and a function to visualize these timesteps.
"""

import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import matplotlib.animation as animation

X_POS = 0
Y_POS = 1
X_VEL = 2
Y_VEL = 3

POS = [X_POS, Y_POS]
VEL = [X_VEL, Y_VEL]


class Model:
    def __init__(self, height, width, num_fish, dt, align_radius, align_weight,
                 cohesion_radius, cohesion_weight, separation_radius,
                 separation_weight):
        self.height = height
        self.width = width
        self.dt = dt
        self.align_radius = align_radius
        self.align_weight = align_weight
        self.cohesion_radius = cohesion_radius
        self.cohesion_weight = cohesion_weight
        self.separation_radius = separation_radius
        self.separation_weight = separation_weight

        self.time = 0
        self.speed = 2
        self.fish = self.spawn_fish(num_fish)

    def spawn_fish(self, num_fish):
        """
        Spawns a number of fish in the model.
        The positions of the fish are uniformly distributed.
        """
        fish = []

        for _ in range(num_fish):
            x = np.random.uniform() * self.width
            y = np.random.uniform() * self.height

            angle = np.random.uniform() * 2 * np.pi
            new_fish = [x, y,
                        self.speed * np.cos(angle), self.speed * np.sin(angle)]
            fish.append(new_fish)

        return np.array(fish)

    def get_neighbours(self, fish, current_fish, radius):
        """
        Returns all the fish that are within a certain radius of the current
        fish.
        """
        neighbours = []

        for f in fish:
            # Don't include itself
            if np.array_equal(f, current_fish):
                continue

            # Calculate the Euclidean distance
            distance = np.linalg.norm(current_fish[POS] - f[POS])

            if distance <= radius:
                neighbours.append(f)

        return np.array(neighbours)

    def alignment(self, fish, current_fish):
        """
        Aligns the fish with its neighbours.
        """
        neighbours = self.get_neighbours(fish, current_fish, self.align_radius)

        if len(neighbours) == 0:
            return np.array([0, 0], dtype=float)

        return np.mean(neighbours[:, VEL], axis=0) - current_fish[VEL]

    def cohesion(self, fish, current_fish):
        """
        Moves the fish towards the mean position of its neighbours.
        """
        neighbours = self.get_neighbours(fish, current_fish,
                                         self.cohesion_radius)

        if len(neighbours) == 0:
            return np.array([0, 0], dtype=float)

        # TODO: - current_fish[VEL]?
        return np.mean(neighbours[:, POS], axis=0) - current_fish[POS]

    def separation(self, fish, current_fish):
        """
        Moves the fish away from its neighbours.
        """
        neighbours = self.get_neighbours(fish, current_fish,
                                         self.separation_radius)

        if len(neighbours) == 0:
            return np.array([0, 0], dtype=float)

        new_vel = np.array([0, 0], dtype=float)

        for n in neighbours:
            distance = np.linalg.norm(current_fish[POS] - n[POS])
            diff = current_fish[POS] - n[POS]
            diff /= distance**2

            new_vel += diff

        return new_vel / len(neighbours) - current_fish[VEL]

    def update_position(self, f):
        f[POS] += f[VEL] * self.dt

        if f[X_POS] < 0:
            f[X_POS] = self.width
        elif f[X_POS] > self.width:
            f[X_POS] = 0

        if f[Y_POS] < 0:
            f[Y_POS] = self.height
        elif f[Y_POS] > self.height:
            f[Y_POS] = 0

    def update_velocity(self, f):
        alignment_vel = self.alignment(self.fish, f) * self.align_weight
        cohesion_vel = self.cohesion(self.fish, f) * self.cohesion_weight
        separation_vel = self.separation(self.fish, f) * self.separation_weight

        f[VEL] += alignment_vel + cohesion_vel + separation_vel

        # Keep the speed constant
        f[VEL] = f[VEL] / np.linalg.norm(f[VEL]) * self.speed

    def step(self):
        self.time += self.dt

        for f in self.fish:
            self.update_position(f)
            self.update_velocity(f)


def animate(i):
    """
    Perform animation step.
    """
    model.step()
    ax.set_title(f'Time: {model.time:.2f}')

    # Update the fish of the animation
    fish_animation.set_data(model.fish[:, 0], model.fish[:, 1])

    return fish_animation, rect


if __name__ == '__main__':
    # Model parameters
    height = 5
    width = 5
    num_fish = 20  # TODO: of density?
    dt = 1 / 30  # 30 fps
    align_radius = 0.5
    align_weight = 0.5
    cohesion_radius = 0.5
    cohesion_weight = 0.5
    separation_radius = 0.2
    separation_weight = 0.5

    model = Model(height=height,
                  width=width,
                  num_fish=num_fish,
                  dt=dt,
                  align_radius=align_radius,
                  align_weight=align_weight,
                  cohesion_radius=cohesion_radius,
                  cohesion_weight=cohesion_weight,
                  separation_radius=separation_radius,
                  separation_weight=separation_weight)

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                         xlim=[0, model.width], ylim=[0, model.height])

    fish_animation, = ax.plot([], [], 'o', color='white', ms=5)

    rect = plt.Rectangle((0, 0), model.width, model.height, ec='none')
    ax.add_patch(rect)

    # Set up the animation
    ani = animation.FuncAnimation(fig, animate, frames=600, interval=10)

    # ani.save('particle_box.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    plt.show()
