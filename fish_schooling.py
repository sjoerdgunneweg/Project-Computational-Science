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
DIS = 4

POS = [X_POS, Y_POS]
VEL = [X_VEL, Y_VEL]


class Simulation(Model):
    def __init__(self, width, height, fish_density, speed, alignment_radius,
                 alignment_weight, cohesion_radius, cohesion_weight,
                 separation_radius, separation_weight):
        Model.__init__(self)

        self.make_param('width', width)
        self.make_param('height', height)
        self.make_param('fish_density', fish_density)
        self.make_param('speed', speed)
        self.make_param('alignment_radius', alignment_radius)
        self.make_param('alignment_weight', alignment_weight)
        self.make_param('cohesion_radius', cohesion_radius)
        self.make_param('cohesion_weight', cohesion_weight)
        self.make_param('separation_radius', separation_radius)
        self.make_param('separation_weight', separation_weight)

        self.time = 0
        self.dt = 1 / 30  # 30 fps  # TODO?
        self.fish = self.spawn_fish()

    def spawn_fish(self):
        """
        Spawns a number of fish in the model.
        The positions of the fish are uniformly distributed.
        """
        fish = []
        num_fish = int(self.width * self.height * self.fish_density)

        for _ in range(num_fish):
            x = np.random.uniform() * self.width
            y = np.random.uniform() * self.height

            angle = np.random.uniform() * 2 * np.pi
            new_fish = [x, y,
                        self.speed * np.cos(angle), self.speed * np.sin(angle)]
            fish.append(new_fish)

        # fish = [[1.0, 1.0, -1.0, 0.0]]
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
            distance = np.sqrt((current_fish[X_POS] - f[X_POS])**2 +
                               (current_fish[Y_POS] - f[Y_POS])**2)

            if distance <= radius:
                neighbours.append(np.append(f, distance))

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

        return (np.mean(neighbours[:, POS], axis=0) - current_fish[POS] -
                current_fish[VEL])

    def separation(self, fish, current_fish):
        """
        Moves the fish away from its neighbours.
        """
        neighbours = self.get_neighbours(fish, current_fish,
                                         self.separation_radius)
        new_vel = np.array([0, 0], dtype=float)

        if len(neighbours) == 0:
            return new_vel

        for n in neighbours:
            if n[DIS] != 0:
                # TODO: optimize
                new_vel += (current_fish[POS] - n[POS]) / n[DIS]**2

        return new_vel / len(neighbours) - current_fish[VEL]

    def update_position(self, f):
        f[POS] += f[VEL] * self.dt
        padding = 0.5

        # Left border
        if f[X_POS] < padding:
            # Random direction to the right
            angle = np.random.uniform(-0.5 * np.pi, 0.5 * np.pi)
            f[VEL] = np.array([self.speed * np.cos(angle),
                               self.speed * np.sin(angle)])
        # Right border
        elif f[X_POS] > self.width - padding:
            # Random direction to the left
            angle = np.random.uniform(0.5 * np.pi, 1.5 * np.pi)
            f[VEL] = np.array([self.speed * np.cos(angle),
                               self.speed * np.sin(angle)])
        # Top border
        if f[Y_POS] < padding:
            # Random direction down
            angle = np.random.uniform(0, np.pi)
            f[VEL] = np.array([self.speed * np.cos(angle),
                               self.speed * np.sin(angle)])
        # Bottom border
        elif f[Y_POS] > self.height - padding:
            # Random direction up
            angle = np.random.uniform(np.pi, 2 * np.pi)
            f[VEL] = np.array([self.speed * np.cos(angle),
                               self.speed * np.sin(angle)])

    def update_velocity(self, f):
        # Randomly change the velocity
        f[VEL] += np.array([np.random.uniform(-0.75, 0.75),
                            np.random.uniform(-0.75, 0.75)])

        alignment_vel = self.alignment(self.fish, f) * self.alignment_weight
        cohesion_vel = self.cohesion(self.fish, f) * self.cohesion_weight
        separation_vel = self.separation(self.fish, f) * self.separation_weight

        f[VEL] += alignment_vel + cohesion_vel + separation_vel

        # Fix the speed
        current_speed = np.sqrt(f[X_VEL]**2 + f[Y_VEL]**2)
        f[VEL] = f[VEL] / current_speed * self.speed

    def step(self):
        self.time += self.dt

        for f in self.fish:
            self.update_velocity(f)
            self.update_position(f)

    def draw(self):
        plt.cla()
        plt.title(f'Time: {self.time:.2f}')
        plt.xlim(0, self.width)
        plt.ylim(0, self.height)
        plt.fill([0, self.width, self.width, 0],
                 [0, 0, self.height, self.height], color='cornflowerblue')
        plt.quiver(self.fish[:, X_POS], self.fish[:, Y_POS],
                   self.fish[:, X_VEL], self.fish[:, Y_VEL], color='white',
                   width=0.01, headwidth=2, headlength=3)

    def reset(self):
        self.time = 0
        self.fish = self.spawn_fish()


if __name__ == '__main__':
    sim = Simulation(width=5,
                     height=5,
                     fish_density=1.0,
                     speed=2,
                     alignment_radius=0.5,
                     alignment_weight=0.6,
                     cohesion_radius=0.5,
                     cohesion_weight=0.2,
                     separation_radius=0.4,
                     separation_weight=0.2)
    gui = GUI(sim)
    gui.start()
