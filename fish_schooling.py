"""
Authors:     Sjoerd Gunneweg; Rinji Le; Pjotr Piet
ID:          13133330; 13344552; 12714933
Date:        20-01-2020
Description:
TODO: update

This file contains the logic for implementing the boids algorithm.
The different rules are implemented in the simulate function.
Every boid is represented as a 1d array with different values for 1: position,
2: speed vector. The different rules are implemented according to the rules
notes in Shiflet and Shiflet. We also implemented a way to simulate all the
different time steps and a function to visualize these timesteps.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyics import Model, GUI
# import numba as nb

X_POS = 0
Y_POS = 1
X_VEL = 2
Y_VEL = 3

POS = [X_POS, Y_POS]
VEL = [X_VEL, Y_VEL]


class Simulation(Model):
    def __init__(self, width, height, num_fish, speed, alignment_radius,
                 alignment_weight, cohesion_radius, cohesion_weight,
                 separation_radius, separation_weight):
        Model.__init__(self)

        self.make_param('width', width)
        self.make_param('height', height)
        self.make_param('num_fish', num_fish)
        self.make_param('speed', speed)
        self.make_param('alignment_radius', alignment_radius)
        self.make_param('alignment_weight', alignment_weight)
        self.make_param('cohesion_radius', cohesion_radius)
        self.make_param('cohesion_weight', cohesion_weight)
        self.make_param('separation_radius', separation_radius)
        self.make_param('separation_weight', separation_weight)

        self.time = 0
        self.dt = 1 / 30  # 30 fps  # TODO?
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
        neighbours = self.get_neighbours(fish, current_fish,
                                         self.alignment_radius)

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
        # Randomly change the velocity
        f[VEL] += np.array([np.random.uniform(-0.75, 0.75),
                            np.random.uniform(-0.75, 0.75)])

        alignment_vel = self.alignment(self.fish, f) * self.alignment_weight
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

    def draw(self):
        plt.cla()
        plt.title(f'Time = {self.time:.2f}')
        plt.xlim(0, self.width)
        plt.ylim(0, self.height)
        plt.fill([0, self.width, self.width, 0],
                 [0, 0, self.height, self.height], color='deepskyblue')
        plt.plot(self.fish[:, 0], sim.fish[:, 1], 'o', color='white', ms=5)

    def reset(self):
        self.time = 0
        self.fish = self.spawn_fish(self.num_fish)


if __name__ == '__main__':
    sim = Simulation(width=5,
                     height=5,
                     num_fish=20,
                     speed=2,
                     alignment_radius=0.5,
                     alignment_weight=0.6,
                     cohesion_radius=0.5,
                     cohesion_weight=0.8,
                     separation_radius=0.4,
                     separation_weight=0.2)
    gui = GUI(sim)
    gui.start()
