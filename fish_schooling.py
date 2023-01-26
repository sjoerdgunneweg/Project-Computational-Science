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
from pyics import Model, GUI, paramsweep
# import numba as nb

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

        # NOTE: obstacles should have the same xmin and xmax
        self.obstacles = [[1.5, 3.5, 0, 1.5], [1.5, 3.5, 3.5, 5]]
        self.padding = 0.2
        self.end_time = 2

        self.loner_counter = []
        self.left_counter = []
        self.right_counter = []
        self.tunnel_counter = []

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
        num_fish = int(self.width * self.height * self.fish_density)

        for _ in range(num_fish):
            angle = np.random.uniform() * 2 * np.pi
            new_fish = [*self.get_random_position(),
                        self.speed * np.cos(angle), self.speed * np.sin(angle)]
            fish.append(new_fish)

        self.loner_counter = np.zeros(num_fish)
        self.left_counter = np.zeros(num_fish)
        self.right_counter = np.zeros(num_fish)
        self.tunnel_counter = np.zeros(num_fish)

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

    def alignment(self, current_fish):
        """
        Aligns the fish with its neighbours.
        """
        neighbours = self.get_neighbours(self.fish, current_fish,
                                         self.alignment_radius)

        if len(neighbours) == 0:
            # TODO: op andere plek ophogen?
            self.loner_counter[np.where(self.fish == current_fish)[0][0]] += 1
            return np.array([0, 0], dtype=float)

        return np.mean(neighbours[:, VEL], axis=0) - current_fish[VEL]

    def cohesion(self, current_fish):
        """
        Moves the fish towards the mean position of its neighbours.
        """
        neighbours = self.get_neighbours(self.fish, current_fish,
                                         self.cohesion_radius)

        if len(neighbours) == 0:
            return np.array([0, 0], dtype=float)

        return (np.mean(neighbours[:, POS], axis=0) - current_fish[POS] -
                current_fish[VEL])

    def separation(self, current_fish):
        """
        Moves the fish away from its neighbours.
        """
        neighbours = self.get_neighbours(self.fish, current_fish,
                                         self.separation_radius)
        new_vel = np.array([0, 0], dtype=float)

        if len(neighbours) == 0:
            return new_vel

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

    def update_position(self, f):
        old_pos = f[POS]
        f[POS] += f[VEL] * self.dt

        # Left border
        if f[X_POS] < self.padding:
            # Random direction to the right
            angle = np.random.uniform(-0.5 * np.pi, 0.5 * np.pi)
            f[VEL] = np.array([self.speed * np.cos(angle),
                               self.speed * np.sin(angle)])
            f[POS] = old_pos
        # Right border
        elif f[X_POS] > self.width - self.padding:
            # Random direction to the left
            angle = np.random.uniform(0.5 * np.pi, 1.5 * np.pi)
            f[VEL] = np.array([self.speed * np.cos(angle),
                               self.speed * np.sin(angle)])
            f[POS] = old_pos
        # Top border
        if f[Y_POS] < self.padding:
            # Random direction up
            angle = np.random.uniform(0, np.pi)
            f[VEL] = np.array([self.speed * np.cos(angle),
                               self.speed * np.sin(angle)])
            f[POS] = old_pos
        # Bottom border
        elif f[Y_POS] > self.height - self.padding:
            # Random direction down
            angle = np.random.uniform(np.pi, 2 * np.pi)
            f[VEL] = np.array([self.speed * np.cos(angle),
                               self.speed * np.sin(angle)])
            f[POS] = old_pos

        if (self.get_positioning(*old_pos[POS]) == 'left' and
                (self.get_positioning(f[X_POS] + self.padding,
                                      f[Y_POS]) == 'lower_obstacle' or
                 self.get_positioning(f[X_POS] + self.padding,
                                      f[Y_POS]) == 'upper_obstacle')):
            # Random direction to the left
            angle = np.random.uniform(0.5 * np.pi, 1.5 * np.pi)
            f[VEL] = np.array([self.speed * np.cos(angle),
                               self.speed * np.sin(angle)])
            f[POS] = old_pos
        elif (self.get_positioning(*old_pos[POS]) == 'right' and
                (self.get_positioning(f[X_POS] - self.padding,
                                      f[Y_POS]) == 'lower_obstacle' or
                 self.get_positioning(f[X_POS] - self.padding,
                                      f[Y_POS]) == 'upper_obstacle')):
            # Random direction to the right
            angle = np.random.uniform(-0.5 * np.pi, 0.5 * np.pi)
            f[VEL] = np.array([self.speed * np.cos(angle),
                               self.speed * np.sin(angle)])
            f[POS] = old_pos
        elif (self.get_positioning(*old_pos[POS]) == 'tunnel'):
            if (self.get_positioning(f[X_POS], f[Y_POS] - self.padding)
                    == 'lower_obstacle'):
                # Random direction up
                angle = np.random.uniform(0, np.pi)
                f[VEL] = np.array([self.speed * np.cos(angle),
                                   self.speed * np.sin(angle)])
                f[POS] = old_pos
            elif (self.get_positioning(f[X_POS], f[Y_POS] + self.padding)
                    == 'upper_obstacle'):
                # Random direction down
                angle = np.random.uniform(np.pi, 2 * np.pi)
                f[VEL] = np.array([self.speed * np.cos(angle),
                                   self.speed * np.sin(angle)])
                f[POS] = old_pos

    def update_velocity(self, f):
        # Randomly change the velocity
        current_angle = np.arctan2(f[Y_VEL], f[X_VEL])
        new_angle = np.random.normal(current_angle, 0.5 * np.pi)
        f[VEL] += np.array([np.cos(new_angle), np.sin(new_angle)])

        # TODO: 1 radius?
        alignment_vel = self.alignment(f) * self.alignment_weight
        cohesion_vel = self.cohesion(f) * self.cohesion_weight
        separation_vel = self.separation(f) * self.separation_weight

        f[VEL] += alignment_vel + cohesion_vel + separation_vel

        # Fix the speed
        current_speed = np.sqrt(f[X_VEL]**2 + f[Y_VEL]**2)
        f[VEL] = f[VEL] / current_speed * self.speed

    def update_position_counter(self, f):
        if self.get_positioning(*f[POS]) == 'left':
            self.left_counter[np.where(self.fish == f)[0][0]] += 1
        elif self.get_positioning(*f[POS]) == 'right':
            self.right_counter[np.where(self.fish == f)[0][0]] += 1
        elif self.get_positioning(*f[POS]) == 'tunnel':
            self.tunnel_counter[np.where(self.fish == f)[0][0]] += 1

    def calculate_times(self):
        num_timesteps = self.time / self.dt

        self.loner_time = np.mean(self.loner_counter / num_timesteps) * 100
        self.left_time = np.mean(self.left_counter / num_timesteps) * 100
        self.right_time = np.mean(self.right_counter / num_timesteps) * 100
        self.tunnel_time = np.mean(self.tunnel_counter / num_timesteps) * 100

    def step(self):
        self.time += self.dt

        # if self.time > self.end_time:
        #     return True

        for f in self.fish:
            self.update_position_counter(f)
            self.update_velocity(f)
            self.update_position(f)

        # Calculate at the last timestep
        if len(self.fish) > 0 and self.time > self.end_time - self.dt:
            self.calculate_times()

    def draw_rect(self, xmin, xmax, ymin, ymax, color):
        plt.fill([xmin, xmax, xmax, xmin],
                 [ymin, ymin, ymax, ymax], color=color, edgecolor='black')

    def draw(self):
        plt.cla()
        plt.title(f'Time: {self.time:.2f}')
        plt.xlim(0, self.width)
        plt.ylim(0, self.height)

        self.draw_rect(0, self.width, 0, self.height, 'cornflowerblue')

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
        self.fish = self.spawn_fish()


if __name__ == '__main__':
    sim = Simulation(width=5,
                     height=5,
                     fish_density=1.0,
                     speed=3,
                     alignment_radius=0.5,
                     alignment_weight=0.6,
                     cohesion_radius=0.5,
                     cohesion_weight=0.2,
                     separation_radius=0.4,
                     separation_weight=0.2)
    gui = GUI(sim)
    gui.start()


    # num_runs = 2
    # density_range = np.linspace(0, 1, 11)

    # paramsweep(sim, num_runs,
    #            {'width': 5, 'height': 5,
    #             'fish_density': density_range,
    #             'speed': 3,
    #             'alignment_radius': 0.5, 'alignment_weight': 0.6,
    #             'cohesion_radius': 0.5, 'cohesion_weight': 0.2,
    #             'separation_radius': 0.4, 'separation_weight': 0.2},
    #            ['loner_time'],
    #            csv_base_filename='results')
