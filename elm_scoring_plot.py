import json
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import SMITHSHOTS
from elm_finder import find_and_label_elms, plot_elms
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


def _get_edges(labels: np.array) -> np.ndarray:
    edges = np.hstack(([False], np.abs(np.diff(labels)).astype(bool)))
    # deal with the edges
    if labels[0].astype(bool):  # if the first element is one
        edges[0] = 1
    if labels[-1].astype(bool):  # if the last element is one
        edges[-1] = 1
    return edges


def add_labeled_periods(ax: plt.axis, times: np.array, labels: np.array):
    edges = _get_edges(labels)
    span_time = times[edges]
    for idx in range(0, span_time.shape[0], 2):
        label = 'labels' if idx == 0 else None
        ax.axvspan(span_time[idx], span_time[idx + 1], color='g', alpha=0.5,
                   linestyle=None, label=label)


def plot_bad_example(elm_file: h5py.Group,
                     bad_dict: dict,
                     quantile: float = 0.95,
                     bes_thresh: float = 1.0,
                     save_fig: bool = False,
                     save_name: str = 'save_fig.png') -> None:
    out, signals, df = find_and_label_elms(elm_file,
                                           elm_file['time'][0],
                                           elm_file['time'][-1],
                                           quantile, bes_thresh)
    fig, axs = plt.subplots(3, 1, sharex='col')
    title = elm_file.name + ': ' + \
            ', '.join('{:2s}:{:>2d}'.format(k, v) for k, v in bad_dict.items())
    # interferometer data
    axs[0].set_title(title)
    add_labeled_periods(axs[0], df['time'].values, df['int_elms'].values)
    axs[0].plot(out['time'], out['denv2f'], 'b', label='denv2f')
    axs[0].plot(out['time'], out['denv3f'], 'r', label='denv3f')
    axs[0].set_ylabel('Line Avg Dens (AU)')
    axs[0].legend(loc='upper right')
    axt0 = axs[0].twinx()
    axt0.plot(out['time'][:-1], signals['denv2f'], 'b', alpha=0.3)
    axt0.plot(out['time'][:-1], signals['denv3f'], 'r', alpha=0.3)
    axt0.set_ylabel('time derivative')
    axt0.set_yscale('log')
    # filterscope data
    add_labeled_periods(axs[1], df['time'].values, df['fil_elms'].values)
    axs[1].plot(out['time'], out['FS02'], 'b', label='FS02')
    axs[1].plot(out['time'], out['FS03'], 'r', label='FS03')
    axs[1].plot(out['time'], out['FS04'], 'm', label='FS04')
    axs[1].set_ylabel('D alpha')
    axs[1].legend(loc='upper right')
    axt1 = axs[1].twinx()
    axt1.plot(out['time'][:-1], signals['FS02'], 'b', alpha=0.3)
    axt1.plot(out['time'][:-1], signals['FS03'], 'r', alpha=0.3)
    axt1.plot(out['time'][:-1], signals['FS04'], 'm', alpha=0.3)
    axt1.set_ylabel('time derivative')
    axt1.set_yscale('log')
    # BES data
    span_time = df['time'][elm_file['labels'][:-1]].values
    axs[2].axvspan(span_time[0], span_time[-1], color='k', alpha=0.5,
                   linestyle=None, label='hand labels')
    add_labeled_periods(axs[2], df['time'].values, df['bes'].values)
    axs[2].plot(out['time'], out['bes'], label='Avg BES')
    axs[2].set_ylabel('BES')
    axs[2].legend(loc='upper right')
    elm_times = df['time'][df['elms']]
    axs[2].plot(elm_times, np.zeros_like(elm_times), '*r')

    axs[-1].set_xlabel('time (ms)')

    if save_fig:
        fig.savefig(fname=save_name, dpi=300)
    else:
        fig.show()
    plt.close(fig=fig)


def find_bad_examples(quantile: float = 0.95, bes_thresh: float = 1.0,
                      make_plots: bool = False):
    with h5py.File(SMITHSHOTS, 'r') as hdf:
        for shot_name, shot_value in hdf.items():
            rd = compare_finder_and_smith(shot=shot_value, quantile=quantile,
                                          bes_thresh=bes_thresh)
            if rd['fp'] > 0 or rd['fn'] > 0:
                print(shot_value.name, rd)
                if make_plots:
                    save_name = 'images/' + shot_name + '.png'
                    plot_bad_example(elm_file=shot_value,
                                     bad_dict=rd,
                                     quantile=quantile,
                                     bes_thresh=bes_thresh,
                                     save_fig=True,
                                     save_name=save_name)




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
