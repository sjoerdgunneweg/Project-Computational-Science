"""
Authors:     Sjoerd Gunneweg; Rinji Le; Pjotr Piet
ID:          13133330; 13344552; 12714933
Date:        January 31, 2023
Description:
This file plots the results of the experiments. The plots are saved in the
'plots' folder.

Usage:
python3 plot.py
or
python3 plot.py -code_review

NOTE: It takes a longer time than as expected, because the plots are saved
with a high resolution.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys


def get_times(xs, filename):
    """
    Gets the times for the given values of x from the given file.
    :param xs: The values of x.
    :param filename: The file to get the data from.
    :return: The times, the lower bound of the 95% confidence interval, and
    the upper bound of the 95% confidence interval.
    """
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)

    times = np.zeros(len(xs))
    ci_low = np.zeros(len(xs))
    ci_high = np.zeros(len(xs))

    for i, x in enumerate(xs):
        x_data = data[data[:, 0] == x]
        times[i] = np.mean(x_data[:, -1])

        # 95% confidence interval
        ci_low[i] = np.percentile(x_data[:, -1], 2.5)
        ci_high[i] = np.percentile(x_data[:, -1], 97.5)

    return times, ci_low, ci_high


def get_num_clusters(xs, filename, value, skip_cols=3):
    """
    Gets the number of clusters for the given values of x from the given file.
    :param xs: The values of x.
    :param filename: The file to get the data from.
    :param value: The value of the first parameter in the file.
    :param skip_cols: The number of columns to skip before the data.
    :return: The number of clusters, the lower bound of the 95% confidence
    interval, and the upper bound of the 95% confidence interval.
    """
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    data = data[np.round(data[:, 0], 1) == value]
    data = data[:, skip_cols:]

    num_clusters = np.zeros(len(xs))
    ci_low = np.zeros(len(xs))
    ci_high = np.zeros(len(xs))

    for i, x in enumerate(xs):
        x_data = data[:, x]
        num_clusters[i] = np.mean(x_data)

        # 95% confidence interval
        ci_low[i] = np.percentile(x_data, 2.5)
        ci_high[i] = np.percentile(x_data, 97.5)

    return num_clusters, ci_low, ci_high


def plot_num_fish(filename='plots/num_fish_loner_time', show=False):
    """
    Plots the average percentage of time spent as a loner for different numbers
    of fish. The confidence intervals are also plotted.
    :param filename: The filename to save the plot to.
    :param show: Whether to show the plot.
    """
    num_fish = np.arange(1, 51)
    data_file = 'results/num_fish_results_loner_time.csv'
    times, ci_low, ci_high = get_times(num_fish, data_file)

    plt.title('The average percentage of time spent as a loner',
              fontweight='bold')
    plt.xlabel('number of fish')
    plt.ylabel('time (%)')

    plt.plot(num_fish, times, '-')
    plt.fill_between(num_fish, ci_low, ci_high, alpha=0.2, label='95% CI')

    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(filename, dpi=1000)

    if show:
        plt.show()

    plt.close()


def plot_tunnel_height_results(filename='plots/tunnel_height_loner_time',
                               show=False):
    """
    Plots the average percentage of time spent as a loner for different tunnel
    heights. The confidence intervals are also plotted.
    :param filename: The filename to save the plot to.
    :param show: Whether to show the plot.
    """
    tunnel_height = np.arange(0.0, 5.1, 0.1)
    data_file = 'results/tunnel_height_results_loner_time.csv'
    times, ci_low, ci_high = get_times(tunnel_height, data_file)

    plt.title('The percentage of time spent as a loner', fontweight='bold')
    plt.xlabel('tunnel height')
    plt.ylabel('time (%)')

    plt.plot(tunnel_height, times, 'o-')
    plt.fill_between(tunnel_height, ci_low, ci_high, alpha=0.2, label='95% CI')

    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(filename, dpi=1000)

    if show:
        plt.show()

    plt.close()


def plot_tunnel_width_results(filename='plots/tunnel_width_loner_time',
                              show=False):
    """
    Plots the average percentage of time spent as a loner for different tunnel
    widths. The confidence intervals are also plotted.
    :param filename: The filename to save the plot to.
    :param show: Whether to show the plot.
    """
    tunnel_width = np.arange(0.0, 5.1, 0.1)
    data_file = 'results/tunnel_width_results_loner_time.csv'
    times, ci_low, ci_high = get_times(tunnel_width, data_file)

    plt.title('The percentage of time spent as a loner', fontweight='bold')
    plt.xlabel('tunnel width')
    plt.ylabel('time (%)')

    plt.plot(tunnel_width, times, 'o-')
    plt.fill_between(tunnel_width, ci_low, ci_high, alpha=0.2, label='95% CI')

    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(filename, dpi=1000)

    if show:
        plt.show()

    plt.close()


def plot_spawn_left_results(filename='plots/spawn_left_right_time',
                            show=False):
    """
    Plots the average percentage of time spent on the right side with the
    initial spawn location on the left. The confidence intervals are also
    plotted.
    :param filename: The filename to save the plot to.
    :param show: Whether to show the plot.
    """
    tunnel_height = np.arange(0.1, 5.0, 0.1)
    data_file = 'results/spawn_left_results_right_time.csv'
    times, ci_low, ci_high = get_times(tunnel_height, data_file)

    plt.title('The average percentage of time spent on the right side',
              fontweight='bold')
    plt.xlabel('tunnel height')
    plt.ylabel('time (%)')

    plt.plot(tunnel_height, times, '-')
    plt.fill_between(tunnel_height, ci_low, ci_high, alpha=0.2, label='95% CI')

    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(filename, dpi=1000)

    if show:
        plt.show()

    plt.close()


def plot_num_clusters(filename='plots/num_clusters', show=False):
    """
    Plots the average number of clusters over time for different numbers of
    fish. The time step is 5, because the number of clusters is calculated
    every 5 time steps.
    :param filename: The filename to save the plot to.
    :param show: Whether to show the plot.
    """
    time = np.arange(5, 105, 5)
    data_file = 'results/num_fish_results_num_clusters.csv'

    for num_fish in [10, 20, 30, 40, 50]:
        num_clusters, _, _ = get_num_clusters(
            time, data_file, num_fish, skip_cols=3)
        plt.plot(time, num_clusters, 'o-', label=f'{num_fish} fish')

    plt.title('The average number of clusters over time',
              fontweight='bold')
    plt.xlabel('time')
    plt.ylabel('number of clusters')

    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(filename, dpi=1000)

    if show:
        plt.show()

    plt.close()


def plot_tunnel_height_num_clusters(
        filename='plots/tunnel_height_num_clusters', show=False):
    """
    Plots the average number of clusters over time for different tunnel
    heights. The time step is 5, because the number of clusters is calculated
    every 5 time steps.
    :param filename: The filename to save the plot to.
    :param show: Whether to show the plot.
    """
    time = np.arange(5, 105, 5)
    data_file = 'results/tunnel_height_results_num_clusters.csv'

    for height in range(1, 5):
        num_clusters, _, _ = get_num_clusters(
            time, data_file, height, skip_cols=3)
        plt.plot(time, num_clusters, 'o-', label=f'tunnel height: {height}')

    plt.title('The average number of clusters over time',
              fontweight='bold')
    plt.xlabel('time')
    plt.ylabel('number of clusters')

    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(filename, dpi=1000)

    if show:
        plt.show()

    plt.close()


def plot_tunnel_width_num_clusters(
        filename='plots/tunnel_width_num_clusters', show=False):
    """
    Plots the average number of clusters over time for different tunnel
    widths. The time step is 5, because the number of clusters is calculated
    every 5 time steps.
    :param filename: The filename to save the plot to.
    :param show: Whether to show the plot.
    """
    time = np.arange(5, 105, 5)
    data_file = 'results/tunnel_width_results_num_clusters.csv'

    for width in range(1, 5):
        num_clusters, _, _ = get_num_clusters(
            time, data_file, width, skip_cols=3)
        plt.plot(time, num_clusters, 'o-', label=f'tunnel width: {width}')

    plt.title('The number of clusters over time',
              fontweight='bold')
    plt.xlabel('time')
    plt.ylabel('number of clusters')

    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(filename, dpi=1000)

    if show:
        plt.show()

    plt.close()


def plot_spawn_left_num_clusters(filename='plots/spawn_left_num_clusters',
                                 show=False):
    """
    Plots the average number of clusters over time for different tunnel
    heights with the initial spawn location on the left. The time step is 5,
    because the number of clusters is calculated every 5 time steps.
    :param filename: The filename to save the plot to.
    :param show: Whether to show the plot.
    """
    time = np.arange(5, 105, 5)
    data_file = 'results/spawn_left_results_num_clusters.csv'

    for height in range(1, 5):
        num_clusters, _, _ = get_num_clusters(
            time, data_file, height, skip_cols=4)
        plt.plot(time, num_clusters, 'o-', label=f'tunnel height: {height}')

    plt.title('The number of clusters over time',
              fontweight='bold')
    plt.xlabel('time')
    plt.ylabel('number of clusters')

    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(filename, dpi=1000)

    if show:
        plt.show()

    plt.close()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '-code_review':
            plot_tunnel_height_num_clusters(filename='plots/code_review')
    else:
        plot_num_fish()
        plot_tunnel_height_results()
        plot_tunnel_width_results()
        plot_spawn_left_results()
        plot_num_clusters()
        plot_tunnel_height_num_clusters()
        plot_tunnel_width_num_clusters()
        plot_spawn_left_num_clusters()
