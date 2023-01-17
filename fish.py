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
# from scipy.spatial.distance import pdist, squareform
# import scipy.integrate as integrate

NEIGH_RANGE = 10
SEPARATION_RANGE = 2

POS = 0
VEL = 1
DIS = 2
X = 0
Y = 1


def get_neighs(fish, current_fish, radius=NEIGH_RANGE):
    """
    Find the neighbouring fish of the fish at index i.
    """
    neighs = []

    for f in fish:
        # We don't want the current fish to be in the list of neighbours
        if np.array_equal(f, current_fish):
            continue

        # Calculates the Euclidean distance
        distance = np.linalg.norm(np.array(current_fish[:2]) - np.array(f[:2]))
        if distance <= radius:
            neighs.append(f + [distance])

    return neighs


def normalize(vector):
    """
    Normalizes a vector.
    """
    return tuple(vector / np.linalg.norm(vector))


def separation(current_fish, separation_neighs):
    """
    Calculates the new direction based on two fish that are too close to
    each other.
    """
    if len(separation_neighs) == 0:
        return np.array([0, 0])
    x_pos = [n[0] for n in separation_neighs]
    y_pos = [n[1] for n in separation_neighs]

    mean_pos = np.array([np.mean(x_pos), np.mean(y_pos)])
    current_pos = np.array(current_fish[:2])

    # This flip of the subtraction ensures that all the fish move away from
    # each other instead of towards each other like in the cohesion rule.
    direction = current_pos - mean_pos

    return direction


# @njit
def alignment(neighs):
    """
    Calculates and returns the new normalized direction based on the directions
    of the neighbours.
    """
    x_dir = [n[2] for n in neighs]
    y_dir = [n[3] for n in neighs]

    direction = np.array([np.mean(x_dir), np.mean(y_dir)])

    return direction


def cohesion(current_fish, neighs):
    """
    Calculates and returns the normalized direction based on the average
    position of the neighs.
    """
    if len(neighs) == 0:
        return np.array([0, 0])

    x_pos = [n[0] for n in neighs]
    y_pos = [n[1] for n in neighs]

    mean_pos = np.array([np.mean(x_pos), np.mean(y_pos)])
    current_pos = np.array(current_fish[:2])

    direction = mean_pos - current_pos

    return direction


# @njit
def simulate(fish):
    """
    For the simulation we will calculate the new positions in parallel to
    ensure every update is based on the same iteration.
    """
    for i in range(len(fish)):
        current_fish = fish[i]
        neighs = get_neighs(fish, current_fish)
        separation_neighs = get_neighs(fish, current_fish, SEPARATION_RANGE)

        separation_dir = separation(current_fish, separation_neighs)

        cohesion_dir = cohesion(current_fish, neighs)
        alignment_dir = alignment(neighs)

        # Calculate the new direction
        new_direction = (cohesion_dir + alignment_dir + separation_dir) / 3

        # Calculate the new position
        new_pos = np.array(current_fish[:2]) + new_direction * 0.1

        # Update the fish
        fish[i] = [new_pos, new_direction, current_fish[2:]]


class Model:
    def __init__(self, height=50, width=50, num_fish=10, fish_size=0.1,
                 dt=1 / 30):
        self.height = height
        self.width = width
        self.fish_size = fish_size
        self.dt = dt

        self.fish = self.spawn_fish(num_fish)
        self.time = 0

    def spawn_fish(self, num_fish):
        fish = []

        for _ in range(num_fish):
            x = np.random.uniform() * self.width
            y = np.random.uniform() * self.height

            # Direction angle in radians
            angle = np.random.uniform() * 2 * np.pi

            # TODO: speed aanpassen (nu is het altijd 1)

            new_fish = [x, y, np.cos(angle), np.sin(angle)]
            fish.append(new_fish)

        return np.array(fish)

    def step(self):
        self.time += self.dt

        self.fish[:, :2] += self.dt * self.fish[:, 2:]

        # TODO: fixen
        # for i in range(len(self.fish)):
        #     current_fish = self.fish[i]
        #     neighs = get_neighs(self.fish, current_fish)
        #     separation_neighs = get_neighs(self.fish, current_fish,
        #                                    SEPARATION_RANGE)

        #     separation_dir = separation(current_fish, separation_neighs)

        #     cohesion_dir = cohesion(current_fish, neighs)
        #     alignment_dir = alignment(neighs)

        #     # calculate the new direction
        #     new_direction = (cohesion_dir + alignment_dir + separation_dir) / 3

        #     # calculate the new position
        #     new_pos = np.array(current_fish[:2]) + new_direction * 0.1

        #     # update the fish
        #     self.fish[i] = np.concatenate((new_pos, new_direction))


        # # find pairs of particles undergoing a collision
        # D = squareform(pdist(self.state[:, :2]))
        # ind1, ind2 = np.where(D < 2 * self.size)
        # unique = (ind1 < ind2)
        # ind1 = ind1[unique]
        # ind2 = ind2[unique]


        # TODO: mooier maken?
        # Prevent the fish from going out of bound
        crossed_x1 = (self.fish[:, 0] < self.fish_size)
        crossed_x2 = (self.fish[:, 0] > self.width - self.fish_size)
        crossed_y1 = (self.fish[:, 1] < self.fish_size)
        crossed_y2 = (self.fish[:, 1] > self.height - self.fish_size)

        self.fish[crossed_x1, 0] = self.fish_size
        self.fish[crossed_x2, 0] = self.width - self.fish_size

        self.fish[crossed_y1, 1] = self.fish_size
        self.fish[crossed_y2, 1] = self.height - self.fish_size

        self.fish[crossed_x1 | crossed_x2, 2] *= -1
        self.fish[crossed_y1 | crossed_y2, 3] *= -1


if __name__ == '__main__':
    # Model parameters
    height = 5
    width = 5
    num_fish = 10
    fish_size = 0.1
    dt = 1 / 30  # 30 fps

    model = Model(height=height, width=width, num_fish=num_fish,
                  fish_size=fish_size, dt=dt)

    # Set up figure and animation
    fig = plt.figure()
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                         xlim=[0, width], ylim=[0, height])

    fish_animation, = ax.plot([], [], 'o', color='blue', ms=6)

    rect = plt.Rectangle((0, 0), width, height,
                         ec='none', lw=2, fc='none')
    ax.add_patch(rect)

    def animate(i):
        """perform animation step"""
        model.step()

        # Update pieces of the animation
        rect.set_edgecolor('k')
        fish_animation.set_data(model.fish[:, 0], model.fish[:, 1])
        # fish_animation.set_markersize(10)
        return fish_animation, rect

    ani = animation.FuncAnimation(fig, animate, frames=600,
                                  interval=10, blit=True)

    #ani.save('particle_box.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    plt.show()
