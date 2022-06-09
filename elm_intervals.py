import numpy as np
import time
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
            dd['i_' + name] = trace[mask].mean()
    return dd


def scatter_elms(df: pd.DataFrame, name: str = ''):
    fig, ax = plt.subplots(1, 1, squeeze=True, figsize=(8, 8))
    names = ['fs02', 'fs03', 'fs04']
    markers = ['x', 'o', 's']
    colors = ['r', 'm', 'b']
    size = 4
    x = df['dt']
    for name, c, m in zip(names, colors, markers):
        y = df['i_' + name]
        ax.scatter(x, y, s=size, color=c, marker=m, label=name)
    ax.set_xlim([0, 50])
    ax.set_ylim()
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('integrated area (arb)')
    ax.legend(loc='upper left')
    plt.show()


FILENAME = 'elm_data_166576.h5'
CHUNKSIZE = 200  # ms

elms = []
start = 600  # 600
while True:
    end = start + CHUNKSIZE
    if end > 5800:  # 5800
        break
    elmo = Elmo(FILENAME, start_time=start, end_time=end,
                percentile=0.997, bes_threshold=1.0)
    df = elmo.find_elms()
    # use peak value (or integrated value) in FS/BES to calculate ELM size
    peaks = df[df['peaks']]
    for ir, row in peaks.iterrows():
        elms.append(make_example(elmo, ir))
    start = end

df = pd.DataFrame(elms)
times = df['time'].values
df['dt'] = np.hstack((times[1:] - times[:-1], [0]))
# plot ELM size (x) versus time-to-next ELM (y)
scatter_elms(df, FILENAME)
