import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import load_bes_data

from typing import Union, Callable, Dict

def diff_signal(arr: np.array) -> np.array:
    return np.abs(np.diff(arr))


def diff_fil(arr: np.array) -> np.array:
    """ ms-scale smoothing before diff """
    scale = 1
    y = np.convolve(arr, np.ones(scale) / scale, mode='same')
    return diff_signal(y)


def nothing(arr: np.array) -> np.array:
    """ make the array the same length as a diff'd array, take abs """
    # return np.abs(arr[:-1])
    return arr[:-1]


def thresh_signal(arr: np.array, thresh: float) -> np.array:
    sig = arr > thresh
    return sig.astype(bool)


def spread_mask(spread: int = 100) -> Callable:
    def func(mask: np.ndarray):
        return (np.convolve(mask, np.ones(spread), mode='same') > 0).astype(int)
    return func


def get_data(elm_file: Union[str, h5py.Group],
             start_time: float, end_time: float) -> dict:
    if isinstance(elm_file, str):
        out = load_bes_data(elm_file, start_time, end_time)
    else:  # for SMITHSHOTS
        out = {k: v[()] for k, v in elm_file.items()}
    return out


def find_elms(data: dict,
              functions: Dict[str, Callable],
              quantile: float,
              bes_thresh: float) -> (dict, dict):
    masks = {}
    signals = {}
    for name, signal in data.items():
        if name in functions:
            y = functions[name](signal)
            signals[name] = y
            if name.lower() == 'bes':
                thresh = bes_thresh
                # thresh = np.quantile(y, 0.99)
            else:
                thresh = np.quantile(y, quantile)
            masks[name] = thresh_signal(y, thresh)
    return masks, signals


def label_elms(df: pd.DataFrame) -> pd.DataFrame:
    # df['int_elms'] = df['denv2f'] & df['denv3f']
    # df['fil_elms'] = df['FS02'] & df['FS03'] & df['FS04']
    n = 100
    df['int_elms'] = (np.convolve(df['denv2f'],
                                  np.ones(2 * n + 1),
                                  mode='same') > 0).astype(int) | \
                     (np.convolve(df['denv3f'],
                                  np.ones(2 * n + 1),
                                  mode='same') > 0).astype(int)
    # two of of three
    fs02 = (np.convolve(df['FS02'], np.ones(2 * n + 1),
                        mode='same') > 0).astype(int)
    fs03 = (np.convolve(df['FS03'], np.ones(2 * n + 1),
                        mode='same') > 0).astype(int)
    fs04 = (np.convolve(df['FS04'], np.ones(2 * n + 1),
                        mode='same') > 0).astype(int)
    df['fil_elms'] =  ((fs02 + fs03 + fs04) > 1)


    df['elms'] = df['int_elms'] & df['fil_elms'] & df['bes']
    return df


def find_big_elms_easy(elm_file: Union[str, h5py.Group],
                       start_time: float,
                       end_time: float,
                       quantile: float = 0.95,
                       bes_thresh: float = 1.0) \
        -> (np.ndarray, np.ndarray, pd.DataFrame):

    functions = {'bes': nothing, 'denv2f': diff_signal, 'denv3f': diff_signal,
                 'FS02': diff_fil, 'FS03': diff_fil, 'FS04': diff_fil
                 }

    # out = load_bes_data('elm_data_166576.h5', 2470, 2490)
    # out = load_bes_data('elm_data_166576.h5', 2415, 2435)
    # out = load_bes_data('elm_data_166576.h5', 2465, 2485)
    # out = load_bes_data('elm_data_166576.h5', 2400, 2600)
    out = get_data(elm_file=elm_file, start_time=start_time, end_time=end_time)

    masks, signals = find_elms(data=out,
                               functions=functions,
                               quantile=quantile,
                               bes_thresh=bes_thresh)

    # logic for labeling an ELM as an ELM
    df = pd.DataFrame(masks)
    df['time'] = out['time'][:-1]
    df = label_elms(df=df)

    return out, signals, df


def find_and_label_elms(elm_file: Union[str, h5py.Group],
                        start_time: float,
                        end_time: float,
                        quantile: float = 0.95,
                        bes_thresh: float = 1.0,
                        spread: int = 100
                        ) -> (np.ndarray, np.ndarray, pd.DataFrame):

    functions = {'bes': nothing, 'denv2f': diff_signal, 'denv3f': diff_signal,
                 'FS02': diff_fil, 'FS03': diff_fil, 'FS04': diff_fil
                 }

    out = get_data(elm_file=elm_file, start_time=start_time, end_time=end_time)

    masks, signals = find_elms(data=out,
                               functions=functions,
                               quantile=quantile,
                               bes_thresh=bes_thresh)

    # mask_func = spread_mask(spread=spread)
    # for key, value in masks.items():
    #     masks[key] = mask_func(value)

    # logic for labeling an ELM as an ELM
    df = pd.DataFrame(masks)
    df['time'] = out['time'][:-1]
    df = label_elms(df=df)

    return out, signals, df


def plot_elms(elm_file: str,
              start_time: float,
              end_time: float,
              quantile: float = 0.95,
              bes_thresh: float = 1.0,
              plot_masks: bool = False,
              save_fig: bool = False,
              save_name: str = 'save_fig.png') -> None:
    if plot_masks:
        rows = 4
    else:
        rows = 3
    out, signals, df = find_and_label_elms(elm_file, start_time, end_time,
                                           quantile, bes_thresh)
    fig, axs = plt.subplots(rows, 1, sharex='col')
    # interferometer data
    axs[0].set_title(elm_file)
    axs[0].plot(out['time'], out['denv2f'], 'b', label='denv2f')
    axs[0].plot(out['time'], out['denv3f'], 'r', label='denv3f')
    axs[0].set_ylabel('Line Avg Dens (AU)')
    axs[0].legend(loc='upper right')
    axt0 = axs[0].twinx()
    axt0.plot(out['time'][:-1], signals['denv2f'], 'b', alpha=0.3)
    axt0.plot(out['time'][:-1], signals['denv3f'], 'r', alpha=0.3)
    axt0.axhline(np.quantile(signals['denv2f'], quantile), linestyle='dashed',
                 color='k')
    axt0.set_ylabel('time derivative')
    axt0.set_yscale('log')
    # filterscope data
    axs[1].plot(out['time'], out['FS02'], 'b', label='FS02')
    axs[1].plot(out['time'], out['FS03'], 'r', label='FS03')
    axs[1].plot(out['time'], out['FS04'], 'm', label='FS04')
    axs[1].set_ylabel('D alpha')
    axs[1].legend(loc='upper right')
    axt1 = axs[1].twinx()
    axt1.plot(out['time'][:-1], signals['FS02'], 'b', alpha=0.3)
    axt1.plot(out['time'][:-1], signals['FS03'], 'r', alpha=0.3)
    axt1.plot(out['time'][:-1], signals['FS04'], 'm', alpha=0.3)
    axt1.axhline(np.quantile(signals['denv2f'], quantile), linestyle='dashed',
                 color='k')
    axt1.set_ylabel('time derivative')
    axt1.set_yscale('log')
    # BES data
    axs[2].plot(out['time'], out['bes'], label='Avg BES')
    axs[2].set_ylabel('BES')
    axs[2].legend(loc='upper right')
    elm_times = df['time'][df['elms']]
    axs[2].plot(elm_times, np.zeros_like(elm_times), '*r')
    if plot_masks:
        # what times are being flagged as elms
        width = df['time'][1] - df['time'][0]
        axs[3].bar(df['time'], df['int_elms'], color='b', width=width, alpha=0.5)
        axs[3].bar(df['time'], df['fil_elms'], color='r', width=width, alpha=0.5)
        axs[3].bar(df['time'], df['bes'], color='m', width=width, alpha=0.5)
        axs[3].set_ylabel('masks')

    axs[-1].set_xlabel('time (ms)')
    print('save_fig: ', save_fig)
    if save_fig:
        fig.savefig(fname=save_name, dpi=300)
    else:
        fig.show()
    plt.close(fig=fig)


if __name__ == "__main__":
    quantile = 0.997
    plot_elms('/sdf/group/ml/datasets/elm_data/elm_data_166576.h5',
              2400, 2600, quantile, 1.0, False, save_fig=True)
    # plot_elms('/sdf/group/ml/datasets/elm_data/elm_data_179498.h5',
    #           5405, 5430, quantile, 1.0, False)
    # plot_elms('/sdf/group/ml/datasets/elm_data/elm_data_179492.h5',
    #           2180, 2205, quantile, 1.0, False)
    # plot_elms('/sdf/group/ml/datasets/elm_data/elm_data_179492.h5',
    #           2180, 2205, quantile, 1.0, False)
