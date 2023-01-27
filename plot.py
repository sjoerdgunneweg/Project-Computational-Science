"""
Authors:     Sjoerd Gunneweg; Rinji Le; Pjotr Piet
ID:          13133330; 13344552; 12714933
Date:        27-01-2023
Description:
TODO
"""

import numpy as np
import matplotlib.pyplot as plt


def get_times(filename):
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)

    num_fish = np.arange(0, 51)
    times = np.zeros(len(num_fish))
    ci_low = np.zeros(len(num_fish))
    ci_high = np.zeros(len(num_fish))

    for i, n in enumerate(num_fish):
        num_fish_data = data[data[:, 0] == n]
        times[i] = np.mean(num_fish_data[:, -1])
        ci_low[i] = np.percentile(num_fish_data[:, -1], 2.5)
        ci_high[i] = np.percentile(num_fish_data[:, -1], 97.5)

    return num_fish, times, ci_low, ci_high


def plot(x, y, ci_low, ci_high, title, filename, show=False):
    plt.title(title, fontweight='bold')
    plt.xlabel('number of fish')
    plt.ylabel('time (%)')
    # plt.figtext(0.01, 0.01,
    #             '10 repetitions per data point\n'
    #             'Tunnel size: 2x2')

    plt.plot(x, y, 'o-')
    plt.fill_between(x, ci_low, ci_high, alpha=0.2, label='95% CI')

    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(filename)

    if show:
        plt.show()

    plt.close()


if __name__ == '__main__':
    for title, filename, plot_filename in [
            ('The percentage of time spent as a loner',
             'results/num_fish_results_loner_time.csv', 'plots/loner_time'),
            ('The percentage of time spent on the left side',
             'results/num_fish_results_left_time.csv', 'plots/left_time'),
            ('The percentage of time spent on the right side',
             'results/num_fish_results_right_time.csv', 'plots/right_time'),
            ('The percentage of time spent in the tunnel',
             'results/num_fish_results_tunnel_time.csv', 'plots/tunnel_time')
         ]:
        num_fish, tunnel_times, ci_low, ci_high = get_times(filename)
        plot(num_fish, tunnel_times, ci_low, ci_high, title, plot_filename)
