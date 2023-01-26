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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pyics import Model, GUI, paramsweep
# import numba as nb
import sys
import time

X_POS = 0
Y_POS = 1
X_VEL = 2
Y_VEL = 3
DIS = 4

POS = [X_POS, Y_POS]
VEL = [X_VEL, Y_VEL]

XMIN = 0
XMAX = 1
YMIN = 2
YMAX = 3


class Simulation(Model):
    def __init__(self,
                 width=5,
                 height=5,
                 num_fish=25,
                 speed=2,
                 alignment_radius=0.5,
                 alignment_weight=0.5,
                 cohesion_radius=0.5,
                 cohesion_weight=0.2,
                 separation_radius=0.2,
                 separation_weight=0.3,
                 padding=0.2,
                 tunnel_width=2,
                 tunnel_height=2,
                 direction_change_period=5,
                 experiment=False,
                 end_time=100,
                 timestep=1,
                 cluster_period=5
                 ):
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
        self.make_param('tunnel_width', tunnel_width)
        self.make_param('tunnel_height', tunnel_height)

        self.time = 0
        self.end_time = end_time
        self.timestep = timestep

        self.direction_change_period = direction_change_period
        self.cluster_period = cluster_period
        self.num_clusters = 0

        # NOTE: obstacles should have the same xmin and xmax
        self.obstacles = [
            [(width - tunnel_width) / 2, (width + tunnel_width) / 2,
             0, (height - tunnel_height) / 2],
            [(width - tunnel_width) / 2, (width + tunnel_width) / 2,
             (height + tunnel_height) / 2, height]]
        self.padding = padding
        self.experiment = experiment

        self.loner_counter = np.zeros(num_fish)
        self.left_counter = np.zeros(num_fish)
        self.right_counter = np.zeros(num_fish)
        self.tunnel_counter = np.zeros(num_fish)

        self.loner_time = 0
        self.left_time = 0
        self.right_time = 0
        self.tunnel_time = 0

        self.fish = self.spawn_fish()

    def get_random_position(self):
        x = np.random.uniform(self.padding, self.width - self.padding)
        y = np.random.uniform(self.padding, self.height - self.padding)

        while (self.get_positioning(x - self.padding, y) == 'lower_obstacle' or
               self.get_positioning(x, y - self.padding) == 'lower_obstacle' or
               self.get_positioning(x + self.padding, y) == 'lower_obstacle' or
               self.get_positioning(x - self.padding, y) == 'upper_obstacle' or
               self.get_positioning(x, y + self.padding) == 'upper_obstacle' or
               self.get_positioning(x + self.padding, y) == 'upper_obstacle'
            #    or self.get_positioning(x, y) != 'left'
            #    or self.get_positioning(x, y) != 'tunnel'
               ):
            x = np.random.uniform(self.padding, self.width - self.padding)
            y = np.random.uniform(self.padding, self.height - self.padding)

        return x, y

    def spawn_fish(self):
        """
        Spawns a number of fish in the model.
        The positions of the fish are uniformly distributed.
        """
        fish = []

        for _ in range(self.num_fish):
            angle = np.random.uniform() * 2 * np.pi
            new_fish = [*self.get_random_position(),
                        self.speed * np.cos(angle), self.speed * np.sin(angle)]
            fish.append(new_fish)

        return np.array(fish)

    def get_neighbours(self, current_fish_index, radius):
        """
        Returns all the fish that are within a certain radius of the current
        fish.
        """
        neighbours = []

        current_fish = self.fish[current_fish_index]
        for i in range(len(self.fish)):
            f = self.fish[i]
            # Don't include itself
            if i == current_fish_index:
                continue

            # Calculate the Euclidean distance
            distance = np.sqrt((current_fish[X_POS] - f[X_POS])**2 +
                               (current_fish[Y_POS] - f[Y_POS])**2)

            if distance <= radius:
                neighbours.append(np.append(f, distance))

        return np.array(neighbours)

    def alignment(self, i):
        """
        Aligns the fish with its neighbours.
        """
        neighbours = self.get_neighbours(i, self.alignment_radius)

        if neighbours.size == 0:
            self.loner_counter[i] += 1
            return np.array([0, 0], dtype=float)

        current_fish = self.fish[i]
        return np.mean(neighbours[:, VEL], axis=0) - current_fish[VEL]

    def cohesion(self, i):
        """
        Moves the fish towards the mean position of its neighbours.
        """
        neighbours = self.get_neighbours(i, self.cohesion_radius)

        if neighbours.size == 0:
            return np.array([0, 0], dtype=float)

        current_fish = self.fish[i]
        return (np.mean(neighbours[:, POS], axis=0) - current_fish[POS] -
                current_fish[VEL])

    def separation(self, i):
        """
        Moves the fish away from its neighbours.
        """
        neighbours = self.get_neighbours(i, self.separation_radius)
        new_vel = np.array([0, 0], dtype=float)

        if neighbours.size == 0:
            return new_vel

        current_fish = self.fish[i]

        for n in neighbours:
            if n[DIS] != 0:
                # TODO: optimize
                new_vel += (current_fish[POS] - n[POS]) / n[DIS]**2

        return new_vel / len(neighbours) - current_fish[VEL]

    def get_positioning(self, x, y):
        if (0 < x < self.obstacles[0][XMIN] and 0 < y < self.height):
            return 'left'
        elif (self.obstacles[0][XMAX] < x < self.width and
              0 < y < self.height):
            return 'right'
        elif (self.obstacles[0][XMIN] < x < self.obstacles[0][XMAX] and
              self.obstacles[0][YMIN] < y < self.obstacles[0][YMAX]):
            return 'lower_obstacle'
        elif (self.obstacles[1][XMIN] < x < self.obstacles[1][XMAX] and
              self.obstacles[1][YMIN] < y < self.obstacles[1][YMAX]):
            return 'upper_obstacle'
        elif (self.obstacles[0][XMIN] < x < self.obstacles[0][XMAX] and
                self.obstacles[0][YMAX] < y < self.obstacles[1][YMIN]):
            return 'tunnel'
        else:
            return 'outside'

    def correct_position_and_velocity(self, f, a, b, old_pos):
        angle = np.random.uniform(a, b)
        f[VEL] = np.array([self.speed * np.cos(angle),
                           self.speed * np.sin(angle)])
        f[POS] = old_pos

    def update_position(self, i):
        f = self.fish[i]
        old_pos = f[POS]
        f[POS] += f[VEL] * 0.05  # To prevent fish from going too fast

        # Left border
        if f[X_POS] < self.padding:
            # Random direction to the right
            self.correct_position_and_velocity(f, -0.5 * np.pi, 0.5 * np.pi,
                                               old_pos)
        # Right border
        elif f[X_POS] > self.width - self.padding:
            # Random direction to the left
            self.correct_position_and_velocity(f, 0.5 * np.pi, 1.5 * np.pi,
                                               old_pos)
        # Top border
        if f[Y_POS] < self.padding:
            # Random direction up
            self.correct_position_and_velocity(f, 0, np.pi, old_pos)
        # Bottom border
        elif f[Y_POS] > self.height - self.padding:
            # Random direction down
            self.correct_position_and_velocity(f, np.pi, 2 * np.pi,
                                               old_pos)

        if (self.get_positioning(*old_pos[POS]) == 'left' and
                (self.get_positioning(f[X_POS] + self.padding,
                                      f[Y_POS]) == 'lower_obstacle' or
                 self.get_positioning(f[X_POS] + self.padding,
                                      f[Y_POS]) == 'upper_obstacle')):
            # Random direction to the left
            self.correct_position_and_velocity(f, 0.5 * np.pi, 1.5 * np.pi,
                                               old_pos)
        elif (self.get_positioning(*old_pos[POS]) == 'right' and
                (self.get_positioning(f[X_POS] - self.padding,
                                      f[Y_POS]) == 'lower_obstacle' or
                 self.get_positioning(f[X_POS] - self.padding,
                                      f[Y_POS]) == 'upper_obstacle')):
            # Random direction to the right
            self.correct_position_and_velocity(f, -0.5 * np.pi, 0.5 * np.pi,
                                               old_pos)
        elif (self.get_positioning(*old_pos[POS]) == 'tunnel'):
            if (self.get_positioning(f[X_POS], f[Y_POS] - self.padding)
                    == 'lower_obstacle'):
                # Random direction up
                self.correct_position_and_velocity(f, 0, np.pi, old_pos)
            elif (self.get_positioning(f[X_POS], f[Y_POS] + self.padding)
                    == 'upper_obstacle'):
                # Random direction down
                self.correct_position_and_velocity(f, np.pi, 2 * np.pi,
                                                   old_pos)

    def update_velocity(self, i):
        f = self.fish[i]

        # Randomly change the velocity a certain period
        if self.time % self.direction_change_period == 0:
            current_angle = np.arctan2(f[Y_VEL], f[X_VEL])
            new_angle = np.random.normal(current_angle, 0.5 * np.pi)
            f[VEL] += np.array([np.cos(new_angle), np.sin(new_angle)])

        alignment_vel = self.alignment(i) * self.alignment_weight
        cohesion_vel = self.cohesion(i) * self.cohesion_weight
        separation_vel = self.separation(i) * self.separation_weight

        f[VEL] += alignment_vel + cohesion_vel + separation_vel

        # Fix the speed
        current_speed = np.sqrt(f[X_VEL]**2 + f[Y_VEL]**2)
        f[VEL] = f[VEL] / current_speed * self.speed

    def update_position_counter(self, i):
        f = self.fish[i]

        if self.get_positioning(*f[POS]) == 'left':
            self.left_counter[i] += 1
        elif self.get_positioning(*f[POS]) == 'right':
            self.right_counter[i] += 1
        elif self.get_positioning(*f[POS]) == 'tunnel':
            self.tunnel_counter[i] += 1

    def calculate_times(self):
        self.loner_time = np.mean(self.loner_counter / self.time) * 100
        self.left_time = np.mean(self.left_counter / self.time) * 100
        self.right_time = np.mean(self.right_counter / self.time) * 100
        self.tunnel_time = np.mean(self.tunnel_counter / self.time) * 100

    def get_num_clusters(self):
        if len(self.fish) > 2:
            clusters = np.arange(2, min(11, len(self.fish)))  # TODO: tot len(self.fish)?
            scores = np.zeros(len(clusters))
            positions = self.fish[:, POS]

            for i, k in enumerate(clusters):
                kmeans = KMeans(n_clusters=k, n_init=10).fit(positions)
                scores[i] = silhouette_score(positions, kmeans.labels_)

            self.num_clusters = clusters[np.argmax(scores)]

    def step(self):
        self.time += self.timestep

        if self.experiment and self.time > self.end_time:
            return True

        # Calculate the new positions and velocities
        for i in range(len(self.fish)):
            self.update_position_counter(i)
            self.update_velocity(i)
            self.update_position(i)

        if self.experiment and self.time % self.cluster_period == 0:
            self.get_num_clusters()

        # Calculate at the last timestep
        if (self.experiment and len(self.fish) > 0 and
                self.time > self.end_time - self.timestep):
            self.calculate_times()

    def draw_rect(self, xmin, xmax, ymin, ymax, color):
        plt.fill([xmin, xmax, xmax, xmin],
                 [ymin, ymin, ymax, ymax], color=color, edgecolor='black')

    def draw(self):
        plt.cla()
        plt.title(f'Time: {self.time}')
        plt.xlim(0, self.width)
        plt.ylim(0, self.height)

        self.draw_rect(0, self.width, 0, self.height, 'cornflowerblue')

        if self.tunnel_width > 0 and self.tunnel_height > 0:
            for obstacle in self.obstacles:
                self.draw_rect(*obstacle, 'darkgray')

        plt.quiver(self.fish[:, X_POS], self.fish[:, Y_POS],
                   self.fish[:, X_VEL], self.fish[:, Y_VEL], color='white',
                   width=0.01, headwidth=2, headlength=3, pivot='mid')

    def reset(self):
        self.time = 0
        self.loner_time = 0
        self.left_time = 0
        self.right_time = 0
        self.tunnel_time = 0

        self.loner_counter = np.zeros(self.num_fish)
        self.left_counter = np.zeros(self.num_fish)
        self.right_counter = np.zeros(self.num_fish)
        self.tunnel_counter = np.zeros(self.num_fish)

        self.obstacles = [
            [(self.width - self.tunnel_width) / 2,
             (self.width + self.tunnel_width) / 2,
             0, (self.height - self.tunnel_height) / 2],
            [(self.width - self.tunnel_width) / 2,
             (self.width + self.tunnel_width) / 2,
             (self.height + self.tunnel_height) / 2, self.height]]

        if self.tunnel_width == 0 or self.tunnel_height == 0:
            self.obstacles = np.zeros((2, 4))

        self.num_clusters = 0
        self.fish = self.spawn_fish()


def experiment():
    num_runs = 2
    num_fish = np.arange(0, 55, 5)

    paramsweep(sim, num_runs,
               {'width': 5, 'height': 5,
                'num_fish': num_fish,
                'speed': 3,
                'alignment_radius': 0.5, 'alignment_weight': 0.6,
                'cohesion_radius': 0.5, 'cohesion_weight': 0.2,
                'separation_radius': 0.4, 'separation_weight': 0.2},
               ['num_clusters'],
               csv_base_filename='results')


if __name__ == '__main__':
    sim = Simulation()

    if len(sys.argv) > 1 and sys.argv[1] == '--experiment':
        sim.experiment = True
        start = time.time()
        experiment()
        print(f'Experiment took {(time.time() - start):.2f} seconds')
    else:
        gui = GUI(sim)
        gui.start()
