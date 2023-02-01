"""
Authors:     Sjoerd Gunneweg; Rinji Le; Pjotr Piet
ID:          13133330; 13344552; 12714933
Date:        February 1, 2023
Description:
This file contains the code for the simulation of fish schooling. The
simulation is based on the Boids algorithm. The visualisation is done using
Matplotlib.
Instead of using a Fish class, the fish are represented by a
2D numpy array. The first two columns contain the position of the fish, the
next two columns contain the velocity of the fish. This data structure is
more optimal for the simulation than using a Fish class.
The simulation is run using the 'GUI' class from the 'pyics' folder.

Usage:
python3 simulation.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pyics import Model, GUI

# Useful constants for indexing
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
    """
    The simulation class. Uses the model class from the 'pyics' folder.
    """
    def __init__(self,
                 width=5.0,
                 height=5.0,
                 num_fish=25,
                 speed=2.0,
                 alignment_radius=0.5,
                 alignment_weight=0.5,
                 cohesion_radius=0.5,
                 cohesion_weight=0.2,
                 separation_radius=0.2,
                 separation_weight=0.3,
                 spawn_location='random',
                 padding=0.2,
                 tunnel_width=2.0,
                 tunnel_height=2.0,
                 direction_change_period=5,
                 experiment=False,
                 end_time=100,
                 timestep=1,
                 cluster_period=5
                 ):
        """
        Initializes the simulation. The fish are placed in the simulation area
        with a random direction. It also creates the tunnel.
        :param width: The width of the simulation area.
        :param height: The height of the simulation area.
        :param num_fish: The number of fish in the simulation.
        :param speed: The speed of the fish.
        :param alignment_radius: The radius for the alignment rule.
        :param alignment_weight: The weight for the alignment rule.
        :param cohesion_radius: The radius for the cohesion rule.
        :param cohesion_weight: The weight for the cohesion rule.
        :param separation_radius: The radius for the separation rule.
        :param separation_weight: The weight for the separation rule.
        :param spawn_location: The location where the fish are spawned.
        The options are 'random', 'tunnel', 'left', and 'right'.
        :param padding: The padding around the simulation area and the tunnel.
        :param tunnel_width: The width of the tunnel.
        :param tunnel_height: The height of the tunnel.
        :param direction_change_period: The period for changing the direction.
        :param experiment: Whether the simulation is an experiment.
        :param end_time: The end time of the simulation.
        :param timestep: The timestep of the simulation.
        :param cluster_period: The period for calculating the clusters.
        """
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
        self.make_param('spawn_location', spawn_location)

        self.time = 0
        self.end_time = end_time
        self.timestep = timestep

        self.direction_change_period = direction_change_period
        self.cluster_period = cluster_period
        self.num_clusters = 0
        self.padding = padding
        self.experiment = experiment

        # NOTE: obstacles should have the same xmin and xmax
        self.obstacles = [
            [(width - tunnel_width) / 2, (width + tunnel_width) / 2,
             0, (height - tunnel_height) / 2],
            [(width - tunnel_width) / 2, (width + tunnel_width) / 2,
             (height + tunnel_height) / 2, height]]

        self.loner_counter = np.zeros(num_fish)
        self.left_counter = np.zeros(num_fish)
        self.right_counter = np.zeros(num_fish)
        self.tunnel_counter = np.zeros(num_fish)

        self.loner_time = 0
        self.left_time = 0
        self.right_time = 0
        self.tunnel_time = 0

        self.fish = self.spawn_fish()

    def on_obstacle(self, x, y):
        """
        Checks if the given position is on an obstacle (or in its padding).
        There are six cases to check.
        """
        return (
            self.get_positioning(x - self.padding, y) == 'lower_obstacle' or
            self.get_positioning(x, y - self.padding) == 'lower_obstacle' or
            self.get_positioning(x + self.padding, y) == 'lower_obstacle' or
            self.get_positioning(x - self.padding, y) == 'upper_obstacle' or
            self.get_positioning(x, y + self.padding) == 'upper_obstacle' or
            self.get_positioning(x + self.padding, y) == 'upper_obstacle')

    def get_random_position(self):
        """
        Gets a random position in the spawn area.
        :return: A random position in the simulation area.
        """
        x = np.random.uniform(self.padding, self.width - self.padding)
        y = np.random.uniform(self.padding, self.height - self.padding)

        if self.spawn_location == 'random':
            # Prevent fish from spawning on obstacles
            while (self.on_obstacle(x, y)):
                x = np.random.uniform(self.padding, self.width - self.padding)
                y = np.random.uniform(self.padding, self.height - self.padding)
        else:
            # Spawn fish on the given location
            while (self.on_obstacle(x, y) or
                   self.get_positioning(x, y) != self.spawn_location):
                x = np.random.uniform(self.padding, self.width - self.padding)
                y = np.random.uniform(self.padding, self.height - self.padding)

        return x, y

    def spawn_fish(self):
        """
        Spawns a number of fish in the model.
        The positions of the fish are uniformly distributed.
        Each fish gets a random direction with the given speed.
        :return: A list that represents the fish ([xPos, yPos, xVel, yVel])
        """
        fish = []

        for _ in range(self.num_fish):
            angle = np.random.uniform() * 2 * np.pi
            new_fish = [*self.get_random_position(),
                        self.speed * np.cos(angle), self.speed * np.sin(angle)]
            fish.append(new_fish)

        return np.array(fish)

    def get_neighbours(self, i, radius, store_distances=False):
        """
        Gets all the fish that are within a certain radius of the current
        fish.
        :param i: The index of the current fish.
        :param radius: The radius to check for neighbours.
        :param store_distances: Whether to store the distances of the
        neighbours.
        :return: A list of neighbours.
        """
        neighbours = []

        current_fish = self.fish[i]
        for j in range(len(self.fish)):
            f = self.fish[j]
            # Don't include itself
            if j == i:
                continue

            # Calculate the Euclidean distance
            distance = np.sqrt((current_fish[X_POS] - f[X_POS])**2 +
                               (current_fish[Y_POS] - f[Y_POS])**2)

            if distance <= radius:
                if store_distances:
                    f = np.append(f, distance)
                neighbours.append(f)

        return np.array(neighbours)

    def alignment(self, i, neighbours):
        """
        Aligns the fish with its neighbours by moving in the same direction.
        :param i: The index of the current fish.
        :param neighbours: The neighbours of the current fish.
        :return: The new velocity vector of the current fish.
        """
        if neighbours is None:
            neighbours = self.get_neighbours(i, self.alignment_radius)

        if neighbours.size == 0:
            self.loner_counter[i] += 1
            return np.array([0, 0], dtype=float)

        current_fish = self.fish[i]
        return np.mean(neighbours[:, VEL], axis=0) - current_fish[VEL]

    def cohesion(self, i, neighbours):
        """
        Moves the fish towards the mean position of its neighbours.
        :param i: The index of the current fish.
        :param neighbours: The neighbours of the current fish.
        :return: The new velocity vector of the current fish.
        """
        if neighbours is None:
            neighbours = self.get_neighbours(i, self.cohesion_radius)

        if neighbours.size == 0:
            return np.array([0, 0], dtype=float)

        current_fish = self.fish[i]
        return (np.mean(neighbours[:, POS], axis=0) - current_fish[POS] -
                current_fish[VEL])

    def separation(self, i):
        """
        Moves the fish away from its neighbours if they are too close.
        :param i: The index of the current fish.
        :return: The new velocity vector of the current fish.
        """
        neighbours = self.get_neighbours(i, self.separation_radius, True)
        new_vel = np.array([0, 0], dtype=float)

        if neighbours.size == 0:
            return new_vel

        current_fish = self.fish[i]

        for n in neighbours:
            if n[DIS] != 0:
                # Neighbour that are further away have a smaller effect
                new_vel += (current_fish[POS] - n[POS]) / n[DIS]**2

        return new_vel / len(neighbours) - current_fish[VEL]

    def get_positioning(self, x, y):
        """
        Gets the positioning in the area of the given position.
        The positioning is either 'left', 'right', 'lower_obstacle',
        'upper_obstacle', 'tunnel' or 'outside'.
        :param x: The x position.
        :param y: The y position.
        :return: The positioning.
        """
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
        """
        Corrects the position and velocity of the fish if it is outside the
        borders.
        :param f: The fish array.
        :param a: The lower bound of the random angle.
        :param b: The upper bound of the random angle.
        :param old_pos: The old position of the fish.
        """
        angle = np.random.uniform(a, b)
        f[VEL] = np.array([self.speed * np.cos(angle),
                           self.speed * np.sin(angle)])
        f[POS] = old_pos

    def update_position(self, i):
        """
        Updates the position of the fish.
        It checks if the fish is outside the borders and corrects its position
        and velocity if necessary.
        :param i: The index of the fish.
        """
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
        """
        Updates the velocity of the fish.
        It changes the velocity a certain period and calculates the alignment,
        cohesion and separation velocities.
        The new velocity is the sum of the old velocity and the alignment,
        cohesion and separation velocities with certain weights.
        :param i: The index of the fish.
        """
        f = self.fish[i]

        # Randomly change the velocity a certain period
        if self.time % self.direction_change_period == 0:
            current_angle = np.arctan2(f[Y_VEL], f[X_VEL])
            new_angle = np.random.normal(current_angle, 0.5 * np.pi)
            f[VEL] += np.array([np.cos(new_angle), np.sin(new_angle)])

        if self.alignment_radius == self.cohesion_radius:
            neighbours = self.get_neighbours(i, self.alignment_radius)
        else:
            neighbours = None

        alignment_vel = self.alignment(i, neighbours) * self.alignment_weight
        cohesion_vel = self.cohesion(i, neighbours) * self.cohesion_weight
        separation_vel = self.separation(i) * self.separation_weight

        f[VEL] += alignment_vel + cohesion_vel + separation_vel

        # Fix the speed to keep the fish moving at the same speed
        current_speed = np.sqrt(f[X_VEL]**2 + f[Y_VEL]**2)
        f[VEL] = f[VEL] / current_speed * self.speed

    def update_position_counter(self, i):
        """
        Updates the counters for the time spent on the left, right and tunnel.
        :param i: The index of the fish.
        """
        f = self.fish[i]

        if self.get_positioning(*f[POS]) == 'left':
            self.left_counter[i] += 1
        elif self.get_positioning(*f[POS]) == 'right':
            self.right_counter[i] += 1
        elif self.get_positioning(*f[POS]) == 'tunnel':
            self.tunnel_counter[i] += 1

    def calculate_times(self):
        """
        Calculates the percentage of time spent on the left, right and tunnel
        and the percentage of time spent alone.
        """
        self.loner_time = np.mean(self.loner_counter / self.time) * 100
        self.left_time = np.mean(self.left_counter / self.time) * 100
        self.right_time = np.mean(self.right_counter / self.time) * 100
        self.tunnel_time = np.mean(self.tunnel_counter / self.time) * 100

    def get_num_clusters(self):
        """
        Calculates the optimal number of clusters using the KMeans algorithm
        and the silhouette score.
        The optimal number of clusters is the number of clusters that gives
        the highest silhouette score.
        It iterates over the number of clusters from 2 to 10 or the number of
        fish, whichever is smaller.
        """
        if len(self.fish) > 2:
            clusters = np.arange(2, min(11, len(self.fish)))
            scores = np.zeros(len(clusters))
            positions = self.fish[:, POS]

            for i, k in enumerate(clusters):
                kmeans = KMeans(n_clusters=k, n_init=10).fit(positions)
                scores[i] = silhouette_score(positions, kmeans.labels_)

            self.num_clusters = clusters[np.argmax(scores)]

    def step(self):
        """
        Updates the positions and velocities of the fish.
        If the experiment mode is on, it calculates the number of clusters
        after a certain period.
        Furthermore, it calculates the percentage of time spent on the left,
        right and tunnel, and the percentage of time spent alone at the last
        time step.
        """
        self.time += self.timestep

        if self.experiment and self.time > self.end_time:
            return True

        # Calculate the new positions and velocities
        for i in range(len(self.fish)):
            self.update_position_counter(i)
            self.update_velocity(i)
            self.update_position(i)

        if self.experiment:
            # Calculate after a certain period
            if self.time % self.cluster_period == 0:
                self.get_num_clusters()

            # Calculate at the last timestep
            if (len(self.fish) > 0 and
                    self.time > self.end_time - self.timestep):
                self.calculate_times()

    def draw_rect(self, xmin, xmax, ymin, ymax, color):
        """
        Draws a rectangle in the plot.
        :param xmin: The minimum x value.
        :param xmax: The maximum x value.
        :param ymin: The minimum y value.
        :param ymax: The maximum y value.
        :param color: The color of the rectangle.
        """
        plt.fill([xmin, xmax, xmax, xmin],
                 [ymin, ymin, ymax, ymax], color=color, edgecolor='black')

    def draw(self):
        """
        Draws the fish and the obstacles (tunnel) in the plot.
        """
        plt.cla()
        plt.title(f'Time: {self.time}')
        plt.xlim(0, self.width)
        plt.ylim(0, self.height)

        self.draw_rect(0, self.width, 0, self.height, 'cornflowerblue')

        # Only draw the tunnel if there is one
        if self.tunnel_width > 0 and self.tunnel_height > 0:
            for obstacle in self.obstacles:
                self.draw_rect(*obstacle, 'darkgray')

        # The fish are represented by arrows
        plt.quiver(self.fish[:, X_POS], self.fish[:, Y_POS],
                   self.fish[:, X_VEL], self.fish[:, Y_VEL], color='white',
                   width=0.01, headwidth=2, headlength=3, pivot='mid')

    def reset(self):
        """
        Resets the model and its local variables.
        It also resets the fish positions and velocities.
        """
        self.time = 0
        self.loner_time = 0
        self.left_time = 0
        self.right_time = 0
        self.tunnel_time = 0

        self.loner_counter = np.zeros(self.num_fish)
        self.left_counter = np.zeros(self.num_fish)
        self.right_counter = np.zeros(self.num_fish)
        self.tunnel_counter = np.zeros(self.num_fish)

        if self.tunnel_width == 0 or self.tunnel_height == 0:
            self.obstacles = np.zeros((2, 4))
        else:
            # The tunnel is always in the center of the plot
            self.obstacles = [
                [(self.width - self.tunnel_width) / 2,
                 (self.width + self.tunnel_width) / 2,
                 0, (self.height - self.tunnel_height) / 2],
                [(self.width - self.tunnel_width) / 2,
                 (self.width + self.tunnel_width) / 2,
                 (self.height + self.tunnel_height) / 2, self.height]]

        self.num_clusters = 0
        self.fish = self.spawn_fish()


if __name__ == '__main__':
    sim = Simulation()
    gui = GUI(sim)
    gui.start()
