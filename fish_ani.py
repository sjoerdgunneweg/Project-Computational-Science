"""
Authors:     Sjoerd Gunneweg; Rinji Le; Pjotr Piet
ID:          13133330; 13344552; 12714933
Date:        20-01-2020
Description: This file contains the logic for implementing the boids algorithm. The different rules are implemented in
the simulate function. Every boid is represented as a 1d array with different values for 1: position, 2: speed vector.
The different rules are implemented according to the rules notes in shiflet and shiflet. We also implemented a way to
simulate all the different time steps and a function to visualize these timesteps.
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import matplotlib.animation as animation


NEIGH_RANGE = 10
SEPARATION_RANGE = 2

POS = 0
VEL = 1
DIS = 2
X = 0
Y = 1

"""find the neighbouring fish of the fish at index i"""
def get_neighs(fish, current_fish, radius=NEIGH_RANGE):
    neighs = []

    for f in fish:
        # we don't want the current fish to be in the list of neighbours
        if f == current_fish:
            continue

        distance = np.linalg.norm(np.array(current_fish[POS]) - np.array(f[POS]))  # calculates the euclidean distance
        if distance <= radius:
            neighs.append(f + [distance])

    return neighs


"""Normalizes a vector"""
def normalize(vector):
    return tuple(vector / np.linalg.norm(vector))


"""calculates the new direction based on two fish that are too close to eachother"""
# @njit
def seperation(current_fish, separation_neighs):
    if len(separation_neighs) == 0:
        return np.array([0, 0])
    x_pos = [n[POS][X] for n in separation_neighs]
    y_pos = [n[POS][Y] for n in separation_neighs]

    mean_pos = np.array([np.mean(x_pos), np.mean(y_pos)])
    current_pos = np.array(current_fish[POS])

    # this flip of the subtraction ensures that all the fish move away from eachother instead of towards eachother like
    # in the cohesion rule.
    direction = current_pos - mean_pos
    return direction


"""Calculates and returns the new normalized direction based on the directions of the neighbours"""
# @njit
def alignment(neighs):
    x_dir = [n[VEL][X] for n in neighs]
    y_dir = [n[VEL][Y] for n in neighs]

    direction = np.array([np.mean(x_dir), np.mean(y_dir)])
    return direction


"""Calculates and returns the normalized direction based on the average position of the neighs"""
def cohesion(current_fish, neighs):
    if len(neighs) == 0:
        return np.array([0, 0])
    x_pos = [n[POS][X] for n in neighs]
    y_pos = [n[POS][Y] for n in neighs]OK

    mean_pos = np.array([np.mean(x_pos), np.mean(y_pos)])
    current_pos = np.array(current_fish[POS])

    direction = mean_pos - current_pos
    return direction

"""
For the simulation we will calculate the new positions in parallel to ensure every update is based on the same 
iteration.
"""
# @njit
def simulate(fish):
    for i in range(len(fish)):
        current_fish = fish[i]
        neighs = get_neighs(fish, current_fish)
        separation_neighs = get_neighs(fish, current_fish, SEPARATION_RANGE)

        separation_dir = seperation(current_fish, separation_neighs)

        cohesion_dir = cohesion(current_fish, neighs)
        alignment_dir = alignment(neighs)

        # calculate the new direction
        new_direction = (cohesion_dir + alignment_dir + separation_dir) / 3

        # calculate the new position
        new_pos = np.array(current_fish[POS]) + new_direction * 0.1

        # update the fish
        fish[i] = [new_pos, new_direction, current_fish[VEL]]

    return fish


def main():
    xmin, ymin = 0, 0
    xmax, ymax = 100, 100
    # generate random fish
    number_of_fish = 50
    fish = []

    for _ in range(number_of_fish):
        x = np.random.uniform() * (xmax - xmin)
        y = np.random.uniform() * (ymax - ymin)
        angle = np.random.uniform() * 2 * np.pi  # direction value in radians
        direction = (np.cos(angle), np.sin(angle))  # direction vector
        # speed = np.random.randint(0, 5)  # speed in [0, 5]
        speed = 1
        new_fish = [(x, y), direction, speed]
        # fish = np.append(fish, new_fish)
        # print(fish, new_fish)
        fish.append(new_fish)

    xs = [f[POS][X] for f in fish]
    ys = [f[POS][Y] for f in fish]

    test = seperation(fish[0], get_neighs(fish, fish[0]))
    print(test)

    # ani = animation.FuncAnimation(fig, animate, frames=600,
    #                           interval=10, blit=True, init_func=init)


    

    fig = plt.scatter(xs, ys) 
    anim = animation.FuncAnimation(fig, simulate)
    plt.show()




    # number_of_iterations = 0
    # # while true:
    # for _ in range(number_of_iterations):
    #     fish = simulate(fish)
    #     print(fish)
    #     print('kaas')


if __name__ == '__main__':
    main()

