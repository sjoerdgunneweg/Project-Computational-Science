"""
Authors:     Sjoerd Gunneweg; Rinji Le; Pjotr Piet
ID:          13133330; 13344552; 12714933
Date:        January 31, 2023
Description:
This file contains the code for the experiments of the fish schooling
simulation. The experiments are run using the 'paramsweep' function. This
function runs the simulation for a given number of runs and a given set of
parameters. The results of the simulations are saved in a csv file in the
'results' folder. For each experiment, multiple metrics are saved:
the number of clusters, the time the fish spend in the left,
right and tunnel area, and the time the fish spend as a loner.

Usage:
python3 experiment.py

NOTE: The experiments take a long time to run (± 1 hour). The results are
already saved in the 'results' folder.
"""

import numpy as np
import time
from simulation import Simulation
from pyics import paramsweep


def num_fish_experiment(sim):
    """
    Runs the experiment for different numbers of fish.
    :param sim: The simulation object.
    """
    num_runs = 20
    num_fish = np.arange(1, 51)

    paramsweep(sim, num_runs,
               {'num_fish': num_fish},
               ['num_clusters', 'loner_time', 'left_time', 'right_time',
                'tunnel_time'],
               csv_base_filename='results/num_fish_results')


def tunnel_height_experiment(sim):
    """
    Runs the experiment for different tunnel heights.
    :param sim: The simulation object.
    """
    num_runs = 20
    tunnel_height = np.arange(0.0, 5.1, 0.1)

    paramsweep(sim, num_runs,
               {'tunnel_height': tunnel_height},
               ['num_clusters', 'loner_time', 'left_time', 'right_time',
                'tunnel_time'],
               csv_base_filename='results/tunnel_height_results')


def tunnel_width_experiment(sim):
    """
    Runs the experiment for different tunnel widths.
    :param sim: The simulation object.
    """
    num_runs = 20
    tunnel_width = np.arange(0.0, 5.1, 0.1)

    paramsweep(sim, num_runs,
               {'tunnel_width': tunnel_width},
               ['num_clusters', 'loner_time', 'left_time', 'right_time',
                'tunnel_time'],
               csv_base_filename='results/tunnel_width_results')


def spawn_left_experiment(sim):
    """
    Runs the experiment for different tunnel heights by spawning the fish
    on the left side of the tunnel.
    :param sim: The simulation object.
    """
    num_runs = 20
    tunnel_height = np.arange(0.1, 5.0, 0.1)

    paramsweep(sim, num_runs,
               {'tunnel_height': tunnel_height, 'spawn_location': 'left'},
               ['num_clusters', 'loner_time', 'left_time', 'right_time',
                'tunnel_time'],
               csv_base_filename='results/spawn_left_results')


if __name__ == '__main__':
    sim = Simulation()
    sim.experiment = True

    start = time.time()
    num_fish_experiment(sim)
    print(f'Number of fish experiment took {(time.time() - start):.2f} '
          'seconds')

    start = time.time()
    tunnel_height_experiment(sim)
    print(f'Tunnel height experiment took {(time.time() - start):.2f} '
          'seconds')

    start = time.time()
    tunnel_width_experiment(sim)
    print(f'Tunnel width experiment took {(time.time() - start):.2f} '
          'seconds')

    start = time.time()
    spawn_left_experiment(sim)
    print(f'Spawn left experiment took {(time.time() - start):.2f} '
          'seconds')
