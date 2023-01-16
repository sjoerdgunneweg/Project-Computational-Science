import numpy
import matplotlib.pyplot as plt
import random


class Fish():
    def __init__(self, position:tuple, direction:int, speed:int):
        self.position = position
        self.direction = direction
        self.speed = speed
        self.neighs = []


"""calculates the distance between two fish (two tuples (x, y))"""
def distance(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 2  # use the eucledian distance


neigh_range = 10
max_neighs = 5

"""find the neighbouring fish of the fish at index i"""
def get_neighs(fish, i):
    global neigh_range
    global max_neighs
    current_fish = fish[i]
    neighs = []

    for f in fish:
        if distance(current_fish.position, f.position) < neigh_range:
            neighs.append(f)
            if len(neighs) == max_neighs:
                return neighs

    return neighs

# how to simulate the fish exactly? What rules come first and what exactly
# does the cohesion rule mean?
def simulate(fish):
    pass


def main():
    xmin, ymin = 0, 0
    xmax, ymax = 100, 100

    # generate random fish
    fish = []
    number_of_fish = 50
    for _ in range(number_of_fish):
        x = random.random() * (xmax - xmin)
        y = random.random() * (ymin - ymax)
        direction = random.random() * 360  # direction value in degrees 0-360
        speed = random.randint(0, 5)  # speed in [0, 5]
        new_fish = Fish((x, y), direction, speed)
        fish.append(new_fish)

    number_of_iterations = 500
    # while true:
    for _ in range(number_of_iterations):
        fish = simulate(fish)




if __name__ == '__main__':
    main()
