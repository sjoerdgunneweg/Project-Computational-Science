# Movement of Fish Schooling

## Table of Contents
* [Introduction](#introduction)
* [Prerequisites](#prerequisites)
* [Running the Code](#running-the-code)
* [Code Review](#code-review)

## Introduction
This project is a simulation of fish schooling. With our own Agent-Based model, we use the Boids algorithm to simulate the movement of fish. For this algorithm, three rules are used to determine the direction of the fish:
1. Separation: fish move away from their neighbors.
2. Alignment: fish move in the same direction as their neighbors.
3. Cohesion: fish move towards the center of mass of their neighbors.

## Prerequisites
Run the following command to install the required packages:
```
pip3 install -r requirements.txt
```

## Running the Code
The simulation can be run by executing the following command:
```
python3 simulation.py
```
To execute the experiments, run:
```
python3 experiment.py
```

Note that the experiments take around hours to run. The results when we ran the experiments are already saved in the `results/` folder.

Finally, to generate all the plots, run:
```
python3 plot.py
```

Again, the plots of the already generated results are saved in the `plots/` folder.

## Code Review
Run the following command to generate one of our plots:
```
python3 plot.py -code_review
```
The plot will be saved as `code_review.png` in the `plots/` folder.
This plot should correspond to the following image:

![alt text](https://github.com/rinjile/Project-Computational-Science/blob/main/plots/tunnel_height_num_clusters.png?raw=true)
