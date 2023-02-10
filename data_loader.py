import os
import time
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dct, idct

from typing import Union, List

SHOTDIR = '/sdf/group/ml/datasets/elm_data'
SHOTFILE = lambda x: 'elm_data_{:d}.h5'.format(x)
# ROOT = '/Users/foshea/Documents/Projects/Anomaly Detection/Plasma/'
ROOT = '/sdf/group/ml/edge_plasma/users/foshea/Plasma/'
FILES = ['labeled-elm-events.hdf5',
         'labeled_elm_events_long_windows_20220527.hdf5']
# SMITHSHOTS = '/sdf/group/ml/datasets/elm_data/smith_traces.h5'
SMITHSHOTS = '/usr/src/app/elm_data/smith_traces.h5'

def _mask_by_time(t: np.ndarray,
                  start_time: float,
                  end_time: float) -> np.ndarray:
    """ Creates a numpy mask of the t between start_time and end_time """
    return (t >= start_time) & (t < end_time)


def get_example(shot: str = '164824-00') -> dict:
    out = {}
    with h5py.File(SMITHSHOTS, 'r') as hdf:
        for k, v in hdf[shot].items():
            out[k] = v[()]
    return out


def get_example_keys() -> list:
    with h5py.File(SMITHSHOTS, 'r') as hdf:
        return list(hdf.keys())


def load_data(data_file: str,
              instrument: str,
              channel: int,
              start_time: float,
              end_time: float
              ) -> dict:
    """ Loads the data """
    with h5py.File(data_file, 'r') as hdffile:
        time_name = instrument + '/times'
        times = hdffile[time_name][()]
        ch_name = instrument + '/' + instrument + '{:02d}'.format(channel)
        signals = hdffile[ch_name][()]
    mask = _mask_by_time(times, start_time, end_time)
    tt = times[mask]
    vv = signals[mask]
    return {'time': tt, 'signal': vv}


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


def load_bes_data(data_source: Union[str, h5py.Group],
                  start_time: float,
                  end_time: float
                  ) -> dict:
    if isinstance(data_source, str):
        with h5py.File(data_source, 'r') as hdffile:
            return extract_bes_data(hdffile,
                                    start_time=start_time,
                                    end_time=end_time)
    elif isinstance(data_source, h5py.Group):
        return extract_bes_data(data_source,
                                start_time=start_time,
                                end_time=end_time)


def extract_bes_data(hdfgrp: h5py.Group,
                     start_time: float,
                     end_time: float
                  ) -> dict:
    """ Loads all the bes-related data """
    data = {}
    bes_times, bes_data = _harvest_bes_data(hdfgrp['BESFU'],
                                            start_time, end_time)
    data['time'] = bes_times
    data['bes'] = bes_data
    int_mask = _mask_by_time(hdfgrp['interferometer/times'][()],
                             start_time, end_time)
    for name in ['denv2f', 'denv3f']:
        key = 'interferometer/' + name
        data[name] = lengthen_data(hdfgrp[key][int_mask],
                                   new_length=bes_times.shape[0])
    fil_mask = _mask_by_time(hdfgrp['filterscope/times'][()],
                             start_time, end_time)
    for name in ['FS02', 'FS03', 'FS04']:
        key = 'filterscope/' + name
        data[name] = lengthen_data(hdfgrp[key][fil_mask],
                                   new_length=bes_times.shape[0])
    return data


#########################################################
# Creating a large h5 file with all traces
#########################################################


def copy_between_hdf_files(source: h5py.Group,
                           dest: h5py.Group, dest_name: str,
                           start_time: float, end_time: float) \
        -> Union[h5py.Group, None]:
    try:
        data = extract_bes_data(hdfgrp=source,
                                start_time=start_time, end_time=end_time)
        grp = dest.create_group(name=dest_name)
        for name, ds in data.items():
            grp.create_dataset(name=name, data=ds)
    except:
        return None
    return grp


def create_single_hdf_file(dest: str, smith_shots: dict):
    total = 0
    for value in smith_shots.values():
        total += len(value)
    print('Moving {:d} time series to {:s}'.format(total, dest))
    count = 0
    total_time = 0
    with h5py.File(dest, 'a') as dest_file:
        for shot, value in smith_shots.items():
            for number, labels in value.items():
                full_path = os.path.join(SHOTDIR, SHOTFILE(shot))
                dest_name = '{:d}-{:02d}'.format(shot, number)
                if (count + 1) % 10 == 0:
                    print('Moving {:>3d} / {:3d}: {:s}'
                          .format(count + 1, total, dest_name))
                    etr = (total - count - 1) * total_time / (count + 1)
                    print('Estimated time remaining: {:3.2e} hours'
                          .format(etr / 3600))
                with h5py.File(full_path, 'r') as source_file:
                    t0 = time.time()
                    grp = copy_between_hdf_files(
                        source=source_file, dest=dest_file,
                        dest_name=dest_name, start_time=labels['time'][0],
                        end_time=labels['time'][-1] + 0.001
                    )
                    grp['labels'] = labels['labels']
                    count += 1
                    total_time += time.time() - t0


def get_smith_labels(directory: str = ROOT,
                     filenames: List[str] = FILES) -> dict:
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

#########################################################
# DCT interpolation functions and testing
#########################################################


def interpolate_with_dct(arr: np.array, new_length: int) -> np.array:
    """
    This function tries to get every other point in the extended array to match
    the values from the original array.  It works great but the points are not
    in the same place in time, so the interpolation is worse than just the
    ordinary DCT/extend/iDCT of lengthen_data.
    """
    y = dct(arr, norm='ortho')
    c_matrix = np.zeros((new_length, arr.shape[0] - 1))
    for idx in range(c_matrix.shape[0]):
        coef = np.pi * (2 * idx + 2) / (4 * arr.shape[0])
        c_matrix[idx] = np.sqrt(2) * \
                        np.cos(coef * np.arange(1, arr.shape[0]))
    res = (y[0] + y[1:] @ c_matrix.T) / np.sqrt(arr.shape[0])
    return res


def lengthen_data(arr: np.array, new_length: int) -> np.array:
    """  Lengthens a data array using DCT/extend with zeros/iDCT.  """
    old_length = arr.shape[0]
    final_arr = np.zeros(new_length)
    y = dct(arr, norm='ortho')
    final_arr[:old_length] = y
    return idct(final_arr, norm='ortho') * np.sqrt(new_length / old_length)


def test_interpolations(t_max: float, n: int) -> None:
    def func(x: np.ndarray) -> np.ndarray:
        # return np.sin(x) + np.cos(2 * x)
        return np.cos(3 * x) * np.exp(-0.1 * (x - 5) ** 2)
    t = np.linspace(0, t_max, n)
    t2 = np.linspace(0, t_max, 2 * n)
    f = func(t)
    ft = func(t2)
    f2 = interpolate_with_dct(f, 2 * n)
    f3 = lengthen_data(f, 2 * n)
    fig, ax = plt.subplots(squeeze=True)
    ax.plot(t, f, '-k', label='original')
    ax.plot(t2, f2, '--b', label='Proper DCT')
    ax.plot(t2, f3, '--r', label='Regular DCT')
    axt = ax.twinx()
    # # show the differences at the "same" points (they aren't the same)
    # axt.bar(t, f2[::2] - f, color='b', alpha=0.5, width=t[1])
    # axt.bar(t, f3[::2] - f, color='r', alpha=0.5, width=t[1])
    axt.bar(t2, f2 - ft, color='b', alpha=0.5, width=t2[1])
    axt.bar(t2, f3 - ft, color='r', alpha=0.5, width=t2[1])
    ax.set_ylabel('curve value')
    ax.set_xlabel('x')
    ax.set_ylim([-2, 2])
    axt.set_ylim([-2, 2])
    axt.set_ylabel('error in interpolation')
    ax.legend()
    plt.show()


