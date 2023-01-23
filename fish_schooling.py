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
import numba as nb

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
                direction = np.random.randint(1, 9)
                fish = Fish((row, col), 1, direction, self.height, self.width)
                self.fish.append(fish)
                self.fish_positions.append((row, col))
                self.grid[row, col] = FISH

    def reset(self):
        self.t = 0
        self.grid = np.full((self.height, self.width), WATER)
        self.num_fish = int(self.fish_density / 100 * self.height * self.width)
        self.fish = []
        self.fish_positions = []

        self.spawn_fish()

    def draw(self):
        plt.cla()

        # Color values of blue (water), white (fish), red (predator),
        # and black (obstacle)
        colors = [(0.329, 0.544, 0.890), (1, 1, 1),
                  (0.930, 0.214, 0.214), (0, 0, 0)]
        colormap = LinearSegmentedColormap.from_list('', colors)

        plt.title(f'Time = {self.t}')
        # plt.figtext(0.01, 0.01, f'Number of fish: {self.num_fish}')

        plt.imshow(self.grid, interpolation='none', vmin=0,
                   vmax=NUM_STATES - 1, cmap=colormap)

    def step(self):
        self.t += 1

        if self.t == 10:
            return True

        for fish in self.fish:
            self.grid[fish.pos] = WATER
            fish.move()
            self.grid[fish.pos] = FISH


class Fish:
    def __init__(self, init_pos, speed, direction, grid_height, grid_width):
        self.pos = init_pos
        self.speed = speed
        self.direction = direction
        self.grid_height = grid_height
        self.grid_width = grid_width

    def get_new_pos(self):
        row, col = self.pos

        if self.direction == 1:
            row -= 1
            col -= 1
        elif self.direction == 2:
            row -= 1
        elif self.direction == 3:
            row -= 1
            col += 1
        elif self.direction == 4:
            col -= 1
        elif self.direction == 5:
            col += 1
        elif self.direction == 6:
            row += 1
            col -= 1
        elif self.direction == 7:
            row += 1
        elif self.direction == 8:
            row += 1
            col += 1

        return row, col

    def move(self):
        row, col = self.get_new_pos()

        if 0 <= row < self.grid_height and 0 <= col < self.grid_width:
            self.pos = (row, col)
        else:
            pass
            # while (row < 0 or row >= self.grid_height or col < 0 or
            #        col >= self.grid_width):
            #     print(list(range(1, 9)).remove(self.direction))
            #     self.direction = np.random.choice(list(
            #         range(1, 9)).remove(self.direction))
            #     row, col = self.get_new_pos()


if __name__ == '__main__':
    sim = Simulation()
    from pyics import GUI
    cx = GUI(sim)
    cx.start()
