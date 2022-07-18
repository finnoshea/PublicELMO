import json
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

from elm_finder_pkg import Elmo, _mask_by_time
from data_loader import SMITHSHOTS

from typing import Dict


def get_slow_data(elmo: Elmo) -> Dict[str, np.ndarray]:
    slow_data = {}
    start_time = elmo.params['start_time']
    end_time = elmo.params['end_time']
    with h5py.File(elmo.params['filename'], 'r') as hdfgrp:
        int_mask = _mask_by_time(hdfgrp['interferometer/times'][()],
                                 start_time, end_time)
        slow_data['int_time'] = hdfgrp['interferometer/times'][int_mask]
        for name in ['denv2f', 'denv3f']:
            key = 'interferometer/' + name
            slow_data[name.lower()] = hdfgrp[key][int_mask]
        fil_mask = _mask_by_time(hdfgrp['filterscope/times'][()],
                                 start_time, end_time)
        slow_data['fil_time'] = hdfgrp['filterscope/times'][fil_mask]
        for name in ['FS02', 'FS03', 'FS04']:
            key = 'filterscope/' + name
            slow_data[name.lower()] = hdfgrp[key][fil_mask]
    return slow_data


def plot_before_and_after_dct(elmo: Elmo, saveplot: bool = False):
    slow_data = get_slow_data(elmo)
    fig, axs = plt.subplots(5, 1, sharex='col', squeeze=True, figsize=(6, 10))

    times = ['int_time', 'int_time', 'fil_time', 'fil_time', 'fil_time']
    plots = ['denv2f', 'denv3f', 'fs02', 'fs03', 'fs04']
    for ax, t, p in zip(axs, times, plots):
        ax.plot(elmo.data['time'], elmo.data[p], '-b')
        ax.scatter(slow_data[t], slow_data[p], s=5, marker='o', color='r')
        ax.set_ylabel(p)
    axs[-1].set_xlabel('time (ms)')
    axs[-1].set_xlim([5625.0, 5625.5])
    axs[-1].ticklabel_format(useOffset=False)
    if saveplot:
        fig.savefig('article_images/before_and_after_dct.png', dpi=300)
    else:
        plt.show()

def plot_dct_space(elmo: Elmo, saveplot: bool = False):
    slow_data = get_slow_data(elmo)
    new_length = elmo.data['bes'].shape[0]
    fig, axs = plt.subplots(5, 1, sharex='col', sharey='col',
                            squeeze=True, figsize=(6, 10))

    plots = ['denv2f', 'denv3f', 'fs02', 'fs03', 'fs04']
    for ax, p in zip(axs, plots):
        final_arr = np.zeros(new_length)
        y = dct(slow_data[p], norm='ortho')
        final_arr[:y.shape[0]] = y
        ax.plot(np.abs(final_arr))
        ax.set_ylabel('dct({:s})'.format(p))
        ax.axvline(y.shape[0], linestyle='dashed', color='r')
        ax.set_yscale('log')
    axs[-1].set_xlabel('frequency (arb.)')
    axs[-1].set_xlim([0, 25000])

    if saveplot:
        fig.savefig('article_images/dct_space.png', dpi=300)
    else:
        plt.show()


def plot_pr_curve(saveplot: bool = False):
    with open('quantile_stats.json', 'r') as jf:
        quants = json.load(jf)

    df = pd.DataFrame.from_dict(quants, orient='index')
    df['precision'] = df.apply(lambda row: row['tp'] / (row['tp'] + row['fp']),
                               axis=1)
    df['recall'] = df.apply(lambda row: row['tp'] / (row['tp'] + row['fn']),
                            axis=1)
    df['AUC'] = df['precision'] * df['recall']

    dfs = df.sort_values('AUC', ascending=False).reset_index(drop=True)
    print(dfs.loc[0])

    fig, ax = plt.subplots(1, 1)

    for thresh in df['bes_thresh'].unique():
        mask = df['bes_thresh'] == thresh
        recall = np.hstack(([1.0], df.loc[mask, 'recall'], [0.0]))
        precision = np.hstack(([0.0], df.loc[mask, 'precision'], [1.0]))
        plt.plot(recall, precision)

    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    ax.set_xticks([n / 5 for n in range(6)])
    ax.set_yticks([n / 5 for n in range(6)])
    # ax.legend(loc='lower left')
    if saveplot:
        fig.savefig('article_images/pr_curve.png', dpi=300)
    else:
        plt.show()

def plot_example_ELM(saveplot: bool = False):
    data = {}
    # example_name = '179859-11'  # this example shows an offset between signals
    example_name = '166597-03'
    with h5py.File(SMITHSHOTS, 'r') as hdf:
        grp = hdf[example_name]
        for key, value in grp.items():
            data[key] = value[()]
    fig, axs = plt.subplots(3, 1, sharex='col', figsize=(4, 6), squeeze=True)
    axs[0].plot(data['time'], data['denv2f'], label='denv2f')
    axs[0].plot(data['time'], data['denv3f'], label='denv3f')
    axs[0].set_ylabel('interferometer (arb.)')
    axs[0].set_yticks([])
    axs[1].plot(data['time'], data['FS02'], label='FS02')
    axs[1].plot(data['time'], data['FS03'], label='FS03')
    axs[1].plot(data['time'], data['FS04'], label='FS04')
    axs[1].set_ylabel('filterscope (arb.)')
    axs[1].set_yticks([])
    axs[2].plot(data['time'], data['bes'], label='bes')
    axs[2].set_ylabel('bes (arb.)')
    axs[2].set_xlabel('time (ms)')
    axs[2].set_yticks([])
    axt = axs[2].twinx()
    axt.fill(data['time'], data['labels'].astype(int), facecolor='k',
             linestyle=None, alpha=0.3)
    axt.set_yticks([])
    xmin = int(data['time'][0])
    xmax = int(data['time'][-1]) + 1
    xticks = np.linspace(xmin, xmax, 5, endpoint=True, dtype=float)
    axs[2].set_xticks(xticks)
    if saveplot:
        fig.savefig('article_images/example_{:s}.png'.format(example_name),
                    dpi=300)
    else:
        plt.show()


if __name__ == "__main__":
    elmo = Elmo('elm_data_166576.h5', start_time=5600, end_time=5800,
                percentile=0.997)
    # elmo = Elmo('/usr/src/app/elm_data/elm_data_184452.h5',
    #             start_time=4000, end_time=4200, percentile=0.997)
    # df = elmo.find_elms()