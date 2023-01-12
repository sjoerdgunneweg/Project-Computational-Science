"""
Name: Sjoerd Gunneweg & Rinji Le & Pjotr Piet
UvA ID: 13133330 & 13344552 & 12714933
Course: Project Computational Science
Date: TODO

Description:
TODO
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pyics import Model

# Define the states of the cells
WATER = 0
FISH = 1
PREDATOR = 2
OBSTACLE = 3

NUM_STATES = 4


class Simulation(Model):
    def __init__(self):
        Model.__init__(self)

        self.make_param('width', 100)
        self.make_param('height', 100)
        self.make_param('fish_density', 1)

    def spawn_fish(self):
        while len(self.fish_positions) != self.num_fish:
            row = np.random.randint(self.height)
            col = np.random.randint(self.width)

            if (row, col) not in self.fish_positions:
                self.fish_positions.append((row, col))
                self.grid[row, col] = FISH

    def reset(self):
        self.t = 0
        self.grid = np.full((self.height, self.width), WATER)
        self.num_fish = int(self.fish_density / 100 * self.height * self.width)
        self.fish_positions = []

        self.spawn_fish()

    def draw(self):
        plt.cla()

        # Color values of blue, white, red, and white
        colors = [(0.329, 0.544, 0.890), (1, 1, 1),
                  (0.930, 0.214, 0.214), (0, 0, 0)]
        colormap = LinearSegmentedColormap.from_list('', colors)

        plt.title(f"Time = {self.t}")
        # plt.figtext(0.01, 0.01, f'Number of fish: {self.num_fish}')

        plt.imshow(self.grid, interpolation='none', vmin=0,
                   vmax=NUM_STATES - 1, cmap=colormap)

    def step(self):
        self.t += 1

        if self.t == 10:
            return True


if __name__ == '__main__':
    sim = Simulation()
    from pyics import GUI
    cx = GUI(sim)
    cx.start()
