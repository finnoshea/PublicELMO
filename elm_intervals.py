import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd

from elm_finder_pkg import Elmo


def make_example(elmo: Elmo, time_index: int, n: int = 500):
    dd = {}
    for name, signal in elmo.data.items():
        dd[name] = signal[time_index]
        if name in ['bes', 'fs02', 'fs03', 'fs04']:
            baseline = signal[time_index - n - 200: time_index - n].mean()
            dd['b_' + name] = baseline
            trace = signal[time_index - n: time_index + n] - baseline
            mask = trace >= 0
            dd['i_' + name] = trace[mask].sum() / (2 * n)
    return dd


def scatter_elms(df: pd.DataFrame, traces: dict, title: str = ''):
    fig, axs = plt.subplots(2, 1, squeeze=True, figsize=(8, 8),
                            gridspec_kw={'height_ratios': [3, 1]})

    names = ['fs02', 'fs03', 'fs04']
    markers = ['x', 'o', 's']
    colors = ['r', 'm', 'b']
    size = 4
    x = df['dt'].values
    for name, c, m in zip(names, colors, markers):
        y = df['i_' + name].values
        axs[0].scatter(x, y, s=size, color=c, marker=m, label=name)
        fit_mask = (x > 10) & (x < 100)
        coefs = np.polyfit(x[fit_mask], y[fit_mask], deg=1)
        axs[0].plot([10, 50],
                    [coefs[1] + coefs[0] * 10, coefs[1] + coefs[0] * 50],
                    color=c, linestyle='-')
    axs[0].set_xlim([0, 50])
    axs[0].set_ylim()
    axs[0].set_xlabel('time to next ELM (ms)')
    axs[0].set_ylabel('integrated area (arb)')
    axs[0].set_title(title)
    axs[0].legend(loc='upper left')

    for key in traces.keys():  # only one trace, but convenient
        if key != 'times':
            axs[1].plot(traces['times'], traces[key])
            axs[1].set_ylabel(key + ' (arb)')
    axs[1].set_xlabel('time (ms)')

    plt.show()


def get_limits(filename: str, block_size: float) -> (float, float):
    with h5py.File(filename, 'r') as hdf:
        start = hdf['BESFU/times'][0]
        stop = hdf['BESFU/times'][-1]
    dt = stop - start
    stop = start + int(dt / block_size) * block_size
    return start, stop

CHUNKSIZE = 200  # ms

# FILENAME = 'elm_data_166576.h5'
FILENAME = 'elm_data_175035.h5'
# FILENAME = '/usr/src/app/elm_data/elm_data_184452.h5'
start, end_point = get_limits(FILENAME, CHUNKSIZE)

elms = []
times = []
traces = []
while True:
    end = start + CHUNKSIZE
    if end > end_point:
        break
    print(start, end)
    elmo = Elmo(FILENAME, start_time=start, end_time=end,
                percentile=0.997, bes_threshold=1.0)
    df = elmo.find_elms()
    times.append(elmo.data['time'])
    traces.append(elmo.data['fs02'])
    # use peak value (or integrated value) in FS/BES to calculate ELM size
    peaks = df[df['peaks']]
    for ir, row in peaks.iterrows():
        elms.append(make_example(elmo, ir))
    start = end


plot_these = {'times': np.hstack(times), 'trace': np.hstack(traces)}
df = pd.DataFrame(elms)
peak_times = df['time'].values
df['dt'] = np.hstack((peak_times[1:] - peak_times[:-1], [0]))
# plot ELM size (x) versus time-to-next ELM (y)
scatter_elms(df, plot_these, FILENAME)
