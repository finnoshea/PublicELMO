from copy import deepcopy
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct, fft, ifft

from typing import Optional, Callable, Dict, List


############################################
# static functions
############################################
def diff_signal(arr: np.array) -> np.array:
    """ first order time difference, take abs"""
    return np.abs(np.diff(arr))


def nothing(arr: np.array) -> np.array:
    """ make the array the same length as a diff'd array """
    return arr[:-1]


def thresh_signal(arr: np.array, thresh: float) -> np.array:
    sig = arr > thresh
    return sig.astype(bool)


def _process_functions(funcs: Dict[str, Callable]) -> Dict[str, Callable]:
    functions = deepcopy(_default_functions)
    for name, func in funcs.items():
        if name.lower() in functions.keys():
            functions[name.lower()] = func
    return functions


def _get_edges(labels: np.array) -> np.ndarray:
    edges = np.hstack(([False], np.abs(np.diff(labels)).astype(bool)))
    # deal with the edges
    if labels[0].astype(bool):  # if the first element is one
        edges[0] = True
    if labels[-1].astype(bool):  # if the last element is one
        edges[-1] = True
    return edges


def add_labeled_periods(ax: plt.axis, times: np.array, labels: np.array):
    edges = _get_edges(labels)
    span_time = times[edges]
    for idx in range(0, span_time.shape[0], 2):
        label = 'labels' if idx == 0 else None
        ax.axvspan(span_time[idx], span_time[idx + 1], color='g', alpha=0.5,
                   linestyle=None, label=label)


def blur_labels(arr: np.ndarray, n: int) -> np.ndarray:
    return (np.convolve(arr, np.ones(2 * n + 1), mode='same') > 0).astype(int)


def _bridge_small_gaps(labels: np.array, n: int) -> np.array:
    out = labels.copy()
    diffs = np.diff(out.astype(int), prepend=[0])
    # correct the edge case of ending in a window
    if out[-1] == 1:
        diffs[-1] = -1
    # get the indexes for the leading edges of the labeled sections (+1)
    starts = np.arange(diffs.shape[0])[diffs > 0]
    # the the indexes for the trailing edges of the labeled sections (-1)
    stops = np.arange(diffs.shape[0])[diffs < 0]
    for sta, sto in zip(starts[1:], stops[:-1]):
        if sta - sto <= n:
            out[sto:sta] = True
    return out


def mark_peaks(arr: np.array, labels: np.array) -> np.array:
    out = np.zeros_like(arr, dtype=bool)
    diffs = np.diff(labels.astype(int), prepend=[0])
    # correct the edge case of ending in a window
    if labels[-1] is True:
        diffs[-1] = -1
    # get the indexes for the leading edges of the labeled sections (+1)
    starts = np.arange(diffs.shape[0])[diffs > 0]
    # the the indexes for the trailing edges of the labeled sections (-1)
    stops = np.arange(diffs.shape[0])[diffs < 0]
    # look for the peaks between each pair of leading and trailing edges
    peaks = []
    for sta, sto in zip(starts, stops):
        peaks.append(sta + np.argmax(arr[sta:sto + 1]) + 1)
    out[peaks] = True
    return out


############################################
# data loading
############################################
def _mask_by_time(t: np.ndarray,
                  start_time: float,
                  end_time: float) -> np.ndarray:
    """ Creates a numpy mask of the t between start_time and end_time """
    return (t >= start_time) & (t < end_time)


def _harvest_bes_data(grp: h5py.Group,
                      start_time: float,
                      end_time: float) -> (np.array, np.array):
    mask = _mask_by_time(grp['times'][()], start_time, end_time)
    data = np.zeros((64, mask.sum()))
    index = 0
    for key, value in grp.items():
        if 'BES' in key:
            value.read_direct(data[index, :], source_sel=mask, dest_sel=None)
            index += 1
    data[32:, :] *= 2  # second half channels have 50% less range
    return grp['times'][mask], data.mean(axis=0)


def lengthen_data(arr: np.array, new_length: int) -> np.array:
    """  Lengthens a data array using DCT/extend with zeros/iDCT.  """
    old_length = arr.shape[0]
    final_arr = np.zeros(new_length)
    y = dct(arr, norm='ortho')
    final_arr[:old_length] = y
    return idct(final_arr, norm='ortho') * np.sqrt(new_length / old_length)


def _count_chunks(L: int, max_chunk_length: int) -> (int, int):
    n = L // max_chunk_length + int(bool(L % max_chunk_length))
    m = L // n
    return n, m


def chunked_lengthen(arr: np.ndarray, new_length: int) -> np.array:
    """ do the lengthening in chunks """
    max_chunk_length = 1000 * 200  # 200 ms long
    n, m = _count_chunks(new_length, max_chunk_length)
    p = arr.shape[0] // n
    output = np.zeros(n * m)
    for i in range(n):
        output[i * m:(i + 1) * m] = lengthen_data(arr[i * p:(i + 1) * p], m)
    return output


def extract_data(hdffile: str,
                 start_time: float,
                 end_time: float
                 ) -> dict:
    """ Loads all the bes-related data """
    data = {}
    with h5py.File(hdffile, 'r') as hdfgrp:
        bes_times, bes_data = _harvest_bes_data(hdfgrp['BESFU'],
                                                start_time, end_time)
        data['time'] = bes_times
        data['bes'] = bes_data
        int_mask = _mask_by_time(hdfgrp['interferometer/times'][()],
                                 start_time, end_time)
        if int_mask.sum() == 0:
            estr = 'interferometer data does not contain the time range'
            estr = estr + '{:7.2f} ms to {:7.2f} ms'.format(start_time,
                                                            end_time)
            raise IndexError(estr)
        for name in ['denv2f', 'denv3f']:
            key = 'interferometer/' + name
            data[name.lower()] = chunked_lengthen(
                hdfgrp[key][int_mask], new_length=bes_times.shape[0])
        fil_mask = _mask_by_time(hdfgrp['filterscope/times'][()],
                                 start_time, end_time)
        if fil_mask.sum() == 0:
            estr = 'filterscope data does not contain the time range'
            estr = estr + '{:7.2f} ms to {:7.2f} ms'.format(start_time,
                                                            end_time)
            raise IndexError(estr)
        for name in ['FS02', 'FS03', 'FS04']:
            key = 'filterscope/' + name
            data[name.lower()] = chunked_lengthen(
                hdfgrp[key][fil_mask], new_length=bes_times.shape[0])

    # account for the chunking making some of the oversampled "data"
    # shorter, only trims off a point or two
    max_length = min(data['denv2f'].shape[0], data['fs02'].shape[0])
    for key, value in data.items():
        data[key] = value[:max_length]

    return data


def get_smith_labels(directory: str,
                     filenames: List[str]) -> dict:
    files = [directory + f for f in filenames]
    elms = {}
    for f in files:
        with h5py.File(f, 'r') as hdf:
            for grp in hdf.values():
                shot = grp.attrs['shot']
                times = grp['time'][()]
                labels = grp['labels'][()].astype(bool)
                if shot in elms:
                    number = len(elms[shot])
                else:
                    elms[shot] = {}
                    number = 0
                elms[shot][number] = {'time': times, 'labels': labels}
    return elms


def low_pass_filter(time_series: np.ndarray,
                    omega_c: float,
                    sample_rate: float = 50000.0) -> np.ndarray:
    u = fft(time_series)
    omegas = np.fft.fftfreq(len(time_series), d=1/sample_rate) * 2 * np.pi
    x = omegas / omega_c
    amp = 1 / np.sqrt(1 + np.power(x, 2))
    phase = np.arctan(-x)
    return ifft(u * amp * np.exp(1j * phase))


_default_functions = {'bes': nothing,
                      'denv2f': diff_signal, 'denv3f': diff_signal,
                      'fs02': diff_signal, 'fs03': diff_signal,
                      'fs04': diff_signal
                      }


class Elmo:
    """
    """

    def __init__(self,
                 filename: str,
                 start_time: float,
                 end_time: float,
                 percentile: float = 0.995,
                 bes_threshold: float = 1.0,
                 functions: Dict[str, Callable] = {}):
        """
        Elmo: ELM Observer Base Class
        """
        self.params = {'filename': filename,
                       'start_time': start_time, 'end_time': end_time,
                       'percentile': percentile, 'bes_threshold': bes_threshold}
        self.functions = _process_functions(functions)

        self.data: Optional[dict[str: np.ndarray]] = None
        self.candidate_masks: Optional[pd.DataFrame] = None
        self.candidate_signals: Optional[Dict[str, np.ndarray]] = None

    def load_data(self):
        """ Load the data into self.data """
        self.data = extract_data(self.params['filename'],
                                 self.params['start_time'],
                                 self.params['end_time'])

    def find_candidates(self):
        """ Find candidates and save into self.candidate_masks and
        self.candidate_signals """
        masks = {}
        signals = {}
        for name, signal in self.data.items():
            if name in self.functions:  # only data with functions is used
                y = self.functions[name](signal)
                signals[name] = y
                if name.lower() == 'bes':
                    thresh = self.params['bes_threshold']
                else:
                    thresh = np.quantile(y, self.params['percentile'])
                masks[name] = thresh_signal(y, thresh)
        self.candidate_masks = pd.DataFrame(masks)
        self.candidate_masks['time'] = nothing(self.data['time'])
        self.candidate_signals = signals

    def label_candidates(self):
        """ Labels candidates as ELMs (or not) and save into
        self.candidate_masks """
        n = 100
        # either
        denv2f = blur_labels(self.candidate_masks['denv2f'], n)
        denv3f = blur_labels(self.candidate_masks['denv3f'], n)
        self.candidate_masks['int_elms'] = (denv2f | denv3f).astype(bool)
        # two of of three
        fs02 = blur_labels(self.candidate_masks['fs02'], n)
        fs03 = blur_labels(self.candidate_masks['fs03'], n)
        fs04 = blur_labels(self.candidate_masks['fs04'], n)
        self.candidate_masks['fil_elms'] = ((fs02 + fs03 + fs04) > 1)

        elms = (self.candidate_masks['int_elms'] &
                self.candidate_masks['fil_elms'] &
                self.candidate_masks['bes']).values

        self.candidate_masks['elms'] = _bridge_small_gaps(labels=elms, n=n)

        self.candidate_masks['peaks'] = mark_peaks(
            nothing(self.data['fs03']), self.candidate_masks['elms'].values)

    def find_elms(self) -> pd.DataFrame:
        """
        Finds the ELMS in the given data.

        Returns
        -------
        Pandas dataframe of ELM labels.
        """
        self.load_data()
        self.find_candidates()
        self.label_candidates()
        return self.candidate_masks

    def plot_data(self,
                  save_fig: bool = False,
                  show_more: bool = True,
                  save_name: str = 'elm_data.png'):
        fa = 'You must run find_elms before there is anything to plot.'
        assert 'elms' in self.candidate_masks, fa
        out = self.data
        df = self.candidate_masks
        signals = self.candidate_signals
        fig, axs = plt.subplots(3, 1, sharex='col', figsize=(6, 8))
        title = self.params['filename']
        # interferometer data
        axs[0].plot(out['time'], out['denv2f'], 'b', label='denv2f')
        axs[0].plot(out['time'], out['denv3f'], 'r', label='denv3f')
        add_labeled_periods(axs[0], df['time'].values, df['int_elms'].values)
        axs[0].set_ylabel('interferometer (arb.)')
        if show_more:
            axs[0].set_title(title)
            axs[0].legend(loc='upper right')
            axt0 = axs[0].twinx()
            axt0.plot(out['time'][:-1], signals['denv2f'], 'b', alpha=0.3)
            axt0.plot(out['time'][:-1], signals['denv3f'], 'r', alpha=0.3)
            axt0.set_ylabel('time derivative')
            axt0.set_yscale('log')
        # filterscope data
        axs[1].plot(out['time'], out['fs02'], 'b', label='FS02')
        axs[1].plot(out['time'], out['fs03'], 'r', label='FS03')
        axs[1].plot(out['time'], out['fs04'], 'm', label='FS04')
        add_labeled_periods(axs[1], df['time'].values, df['fil_elms'].values)
        axs[1].set_ylabel('filterscope (arb.)')
        if show_more:
            axs[1].legend(loc='upper right')
            axt1 = axs[1].twinx()
            axt1.plot(out['time'][:-1], signals['fs02'], 'b', alpha=0.3)
            axt1.plot(out['time'][:-1], signals['fs03'], 'r', alpha=0.3)
            axt1.plot(out['time'][:-1], signals['fs04'], 'm', alpha=0.3)
            axt1.set_ylabel('time derivative')
            axt1.set_yscale('log')
        # BES data
        axs[2].plot(out['time'], out['bes'], label='Avg BES')
        add_labeled_periods(axs[2], df['time'].values, df['elms'].values)
        axs[2].set_ylabel('BES (V)')
        if show_more:
            axs[2].legend(loc='upper right')
        elm_times = df['time'][df['peaks']]
        axs[2].plot(elm_times, np.zeros_like(elm_times), '*r')

        axs[-1].set_xlabel('time (ms)')

        if save_fig:
            fig.savefig(fname=save_name, dpi=300)
            plt.close(fig=fig)
        else:
            fig.show()

    def plot_bes_at_elm(self, peak_time_index: int):
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        dt = 1000  # 1 ms
        time = self.data['time'][peak_time_index - dt:peak_time_index + dt + 1]
        bes = self.data['bes'][peak_time_index - dt:peak_time_index + dt + 1]
        ax.plot(time, bes, 'k', alpha=0.8)
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('mean bes signal (V)')
        time_labels = time[np.arange(0, time.shape[0], dt // 2)]
        ax.set_xticks(time_labels)
        plt.show()


class FilterscopeElmCompiler:
    """
    """
    def __init__(self):
        """
        Grabs all the ELMs labeled by Dave Smith and saves them in an HDF5 file.
        """
        self.smith_elms = None

    def get_elms_labeled_by_smith(self):
        self.smith_elms = get_smith_labels(
            directory='/usr/src/app/elm_data/',
            filenames=['labeled-elm-events.hdf5',
                       'labeled_elm_events_long_windows_20220527.hdf5']
        )

    @staticmethod
    def make_shot_filename(shot: int) -> str:
        filedir = '/usr/src/app/elm_data/'
        fn = 'elm_data_' + str(shot) + '.h5'
        return filedir + fn

    def package_all_elms(self,
                         filename: str = 'smith_elms_filterscope.h5'):
        assert_str = 'run self.get_elms_labeled_by_smith() first'
        assert self.smith_elms is not None, assert_str

        with h5py.File(filename, 'w') as hdf:
            for shot_count, (shot, indexes) in \
                    enumerate(self.smith_elms.items()):
                ptr = '{:03d}/{:03d}: saving filterscope '.format(
                    shot_count + 1, len(self.smith_elms.keys()))
                ptr = ptr + 'for shot {:d}'.format(shot)
                print(ptr)
                # open the hdf5 file with the filterscope data
                fn = self.make_shot_filename(shot)
                with h5py.File(fn, 'r') as source_hdf:
                    source = source_hdf['filterscope']
                    for index, values in indexes.items():
                        # start saving some data
                        sn = str(shot) + '-{:02d}'.format(index)
                        print('Saving ' + sn)
                        grp = hdf.create_group(sn)
                        grp.create_dataset(
                            'labels',  data=values['labels'])  # smith's labels
                        grp.create_dataset(
                            'label_time', data=values['time'])
                        # mask the times from values['time']
                        mask = _mask_by_time(source['times'][()],
                                             values['time'][0],
                                             values['time'][-1])
                        grp.create_dataset('time', data=source['times'][mask])
                        # save the filterscope data
                        for fs in ['FS02', 'FS03', 'FS04']:
                            grp.create_dataset(fs, data=source[fs][mask])


class FilterscopeElmFinder:
    """
    """
    def __init__(self):
        """
        ELM labeler as described in:
        https://iopscience.iop.org/article/10.1088/1741-4326/aa6b16
        """
        self.smith_elms = {}

    @staticmethod
    def eldon_algo(time_series: np.ndarray) \
            -> (np.ndarray, np.ndarray, np.ndarray):
        """

        Parameters
        ----------
        time_series : np.ndarray
            The filterscope data sampled at 50 kHz.

        Returns
        -------
        D-S, SD1, and SD2 from the Eldon paper.
        """
        slow = low_pass_filter(time_series, 1000.0).real
        dsdt = np.diff(slow)
        sd1 = low_pass_filter(dsdt, 1000.0).real
        sd2 = low_pass_filter(dsdt, 200.0).real
        return time_series[:-1] - slow[:-1], sd1, sd2


if __name__ == "__main__":
    # elmo = Elmo('elm_data_166576.h5', start_time=5600, end_time=5800,
    #             percentile=0.997)
    elmo = Elmo('elm_data_166576.h5', start_time=2450, end_time=2550,
                percentile=0.997)
    # elmo = Elmo('elm_data_166576.h5', start_time=2460, end_time=2660,
    #             percentile=0.997)
    # elmo = Elmo('/usr/src/app/elm_data/elm_data_184452.h5',
    #             start_time=4000, end_time=4200, percentile=0.997)
    df = elmo.find_elms()
