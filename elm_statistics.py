import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import load_data

#FILENAME = '/usr/src/app/shared_volume/ecebes_166434.h5'
FILENAME = 'ecebes_166434.h5'

data_start = 0
data_end = 5800

data_sets = []
for channel in range(1, 65):
    data_dict = load_data(FILENAME, 'BESFU', channel, data_start, data_end)

    # compute some basic stuff that is hopefully informative
    # elms are signals greater than 1 V
    scale = 1
    if channel > 32:
        scale = 2  # upper channels have half range
    elms = np.abs(data_dict['signal']) > (1.0 / scale)
    fudge_factor = 100
    win = np.ones(fudge_factor)
    elms = (np.convolve(elms, win) > 0).astype(bool)[:-(fudge_factor - 1)]
    # this is slightly slower, but more compact
    elm_starts_mask = np.hstack(([0, np.diff(elms.astype(int))])) > 0
    elm_starts = data_dict['time'][elm_starts_mask]
    data_sets.append(np.diff(elm_starts))

time_diffs = np.hstack(data_sets)
rate_parameter = time_diffs.shape[0] / (data_end - data_start) / len(data_sets)


def exp_dist(times: np.array) -> np.array:
    counts = np.exp(-1 * rate_parameter * times[:-1]) - \
             np.exp(-1 * rate_parameter * times[1:])
    return counts


fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex='col')
axs = axs.flatten()
counts, edges, patches = axs[0].hist(time_diffs,
                                     bins=[i for i in range(0, 101, 5)])
theory_counts = time_diffs.shape[0] * exp_dist(edges)
axs[0].plot(edges[:-1] + edges[1] / 2, theory_counts, '--r')
axs[2].bar(edges[:-1], counts - theory_counts, width=edges[1])
axs[1].bar(edges[:-1], counts, width=edges[1])
axs[1].plot(edges[:-1] + edges[1] / 2, theory_counts, '--r')
axs[3].bar(edges[:-1], np.abs(counts - theory_counts), width=edges[1])

axs[1].set_yscale('log')
axs[3].set_yscale('log')
axs[2].set_xlabel('time difference (ms)')
axs[3].set_xlabel('time difference (ms)')
axs[0].set_ylabel('counts')
axs[1].set_ylabel('counts')
axs[0].set_title('linear scale')
axs[1].set_title('log scale')
plt.show()
