import json
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import SMITHSHOTS
from elm_finder import find_big_elms_easy, find_and_label_elms
from elm_scoring import compare_finder_and_smith


def plot_dataframe(shot_name: str = '175035-05',
                   quantile: float = 0.95,
                   bes_thresh: float = 1.0) -> None:
    with h5py.File(SMITHSHOTS, 'r') as hdf:
        shot = hdf[shot_name]
        _, _, df = find_and_label_elms(elm_file=shot,
                                       start_time=shot['time'][0],
                                       end_time=shot['time'][-1],
                                       quantile=quantile,
                                       bes_thresh=bes_thresh
                                       )
        labels = shot['labels'][:-1]
        fig, axs = plt.subplots(2, 1, sharex='col', figsize=(8, 8))
        axs[0].plot(df['time'], df['int_elms'], label='int_elms')
        axs[0].plot(df['time'], df['fil_elms'], label='fil_elms')
        axs[0].plot(df['time'], df['bes'], label='bes')
        axs[0].set_ylabel('active')
        axs[0].set_title(shot_name)
        axs[0].legend(loc='upper right')

        axs[1].plot(df['time'], df['elms'], label='elms')
        axs[1].plot(df['time'], labels, label='labels')
        axs[1].legend(loc='upper right')
        axs[1].set_xlabel('time (ms)')
        axs[1].set_ylabel('active')

        plt.show()


def find_bad_examples(quantile: float = 0.95, bes_thresh: float = 1.0):
    with h5py.File(SMITHSHOTS, 'r') as hdf:
        for shot_value in hdf.values():
            rd = compare_finder_and_smith(shot=shot_value, quantile=quantile,
                                          bes_thresh=bes_thresh)
            if rd['fp'] > 0 or rd['fn'] > 0:
                print(shot_value.name, rd)


with open('quantile_stats.json', 'r') as jf:
    quants = json.load(jf)

df = pd.DataFrame.from_dict(quants, orient='index')
df['precision'] = df.apply(lambda row: row['tp'] / (row['tp'] + row['fp']),
                           axis=1)
df['recall'] = df.apply(lambda row: row['tp'] / (row['tp'] + row['fn']),
                        axis=1)
df['AUC'] = df['precision'] * df['recall']

fig, ax = plt.subplots(1, 1)

for thresh in df['bes_thresh'].unique():
    mask = df['bes_thresh'] == thresh
    plt.plot(df.loc[mask, 'recall'], df.loc[mask, 'precision'],
             label='bes >= {:2.1f}'.format(thresh))
ax.set_xlabel('recall')
ax.set_ylabel('precision')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.legend(loc='lower left')
plt.show()
