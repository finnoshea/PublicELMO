import time
import itertools
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

from typing import Generator, Any

from brainblocks.blocks import ScalarTransformer, SequenceLearner


def _mask_by_time(t: np.ndarray,
                  start_time: float,
                  end_time: float) -> np.ndarray:
    """ Creates a numpy mask of the t between start_time and end_time """
    return (t >= start_time) & (t < end_time)


def load_data(data_params: dict) -> dict:
    """ Loads the data """
    with h5py.File(data_params['input_file'], 'r') as hdffile:
        time_name = data_params['data_name'].split('/')[0] + '/times'
        times = hdffile[time_name][()]
        signals = hdffile[data_params['data_name']][()]
    mask = _mask_by_time(times,
                         data_params['start_time'],
                         data_params['end_time'])
    tt = times[mask]
    vv = signals[mask]
    if data_params['abs']:
        vv = np.abs(vv)
    return {'time': tt, 'signal': vv}


def _update_dict(dd: dict, name: str, value: Any) -> None:
    """
    Updates dictionary dd with the name, value pair.

    Parameters
    ----------
    dd : dict
        Dictionary to update.
    name : str
        Parameter key of the form element__feature, where element is a
        high-level element of the HTM and feature is its feature name.
        For example: encoder__size, tm__activationThreshold
    value : Any
        What to set the parameter to.
    """
    first, second = name.split('__')
    try:
        dd[first][second] = value
    except KeyError:
        dd[first] = {second: value}


def build_param_dict(names: list[str],
                     values: tuple) -> dict:
    """
    Keys and values to build a parameter dictionary with.

    Parameter keys have the form element__feature, where element is a
    high-level element of the HTM and feature is its feature name.
    For example: encoder__size, tm__activationThreshold

    Parameters
    ----------
    names : list[str]
        Parameter keys for the dictionary.
    values : tuple
        Parameter values.

    Returns
    -------
    Dictionary of parameter values.
    """
    dd = {}
    for name, value in zip(names, values):
        _update_dict(dd, name, value)
    return dd


def yield_permutations(params: dict) -> Generator[dict, None, None]:
    """
    Yield all the different permutations from params.

    Parameters
    ----------
    params : dict
        Dictionary of key: list pairs of values to be scanned over.

    Yields
    ------
    Parameter dictionaries.
    """
    names = []
    values = []
    for name, value in params.items():
        names.append(name)
        values.append(value)
    for p in itertools.product(*values):
        yield build_param_dict(names, p)


def train_permutation(params: dict, data: dict) \
        -> (float, np.ndarray, np.ndarray):
    """

    Parameters
    ----------
    params
    data

    Returns
    -------

    """
    st = ScalarTransformer(**params['st'])
    sl = SequenceLearner(**params['sl'])

    sl.input.add_child(st.output)

    scores = np.zeros_like(data['signal'])
    run_time = np.zeros_like(data['signal'])

    # dummy run to cut down on first run time
    st.set_value(0)
    st.feedforward()
    sl.feedforward(learn=False)
    # Loop through data
    t_start = time.time()
    for index, value in enumerate(data['signal']):
        t0 = time.time()
        # Set scalar transformer value and compute
        st.set_value(value)
        st.feedforward()

        # Compute the sequence learner
        sl.feedforward(learn=True)

        # Get anomaly score
        scores[index] = sl.get_anomaly_score()
        run_time[index] = time.time() - t0
    t_end = time.time()
    return t_end - t_start, scores, run_time


def init_block(params: dict) -> (ScalarTransformer, SequenceLearner):
    # Setup blocks
    st = ScalarTransformer(**params['st'])
    sl = SequenceLearner(**params['sl'])

    # Connect blocks
    # connect sequence_learner input to scalar_transformer output
    sl.input.add_child(st.output)
    return st, sl


def reset_train_permutation(params: dict) -> (float, pd.DataFrame):
    data_start = params['data']['start_time']
    data_end = params['data']['end_time']
    data_dict = load_data(params['data'])

    # compute some basic stuff that is hopefully informative
    # elms are signals greater than 1 V
    elms = np.abs(data_dict['signal']) > 1.0
    fudge_factor = 100
    win = np.ones(fudge_factor)
    elms = (np.convolve(elms, win) > 0).astype(bool)[:-(fudge_factor - 1)]
    # this is slightly slower, but more compact
    elm_starts_mask = np.hstack(([0, np.diff(elms.astype(int))])) > 0
    elm_starts = data_dict['time'][elm_starts_mask]

    t_run = 0
    scores = []
    run_time = []

    intervals = np.hstack(
        (np.array(data_start),
         np.repeat(elm_starts, 2),
         np.array(data_end))
    ).reshape(-1, 2)

    for (start_time, end_time) in intervals:
        params['data']['start_time'] = start_time
        params['data']['end_time'] = end_time
        dd = load_data(params['data'])
        seg_scores = np.zeros_like(dd['signal'])
        seg_run_time = np.zeros_like(dd['signal'])
        # reset the HTM elements
        st, sl = init_block(params)
        # dummy run to cut down on first run time
        st.set_value(0)
        st.feedforward()
        sl.feedforward(learn=False)
        # Loop through data
        t_start = time.time()
        for index, value in enumerate(dd['signal']):
            t0 = time.time()
            # Set scalar transformer value and compute
            st.set_value(value)
            st.feedforward()

            # Compute the sequence learner
            sl.feedforward(learn=True)

            # Get anomaly score
            seg_scores[index] = sl.get_anomaly_score()
            seg_run_time[index] = time.time() - t0
        scores.append(seg_scores)
        run_time.append(seg_run_time)
        t_run += time.time() - t_start
    params['data']['start_time'] = data_start
    params['data']['end_time'] = data_end

    scores = np.hstack(scores)
    run_time = np.hstack(run_time)

    elm_scores = scores[elm_starts_mask]

    # anomaly candidates have score greater than 0.01 - permissive
    anomaly_candidates = scores > 0.01
    anomalies = np.logical_and(anomaly_candidates, np.logical_not(elms))
    anom_times = data_dict['time'][anomalies]
    anom_scores = scores[anomalies]
    elm_not_anom = np.hstack((np.ones_like(elm_scores).astype(bool),
                              np.zeros_like(anom_scores).astype(bool))
                             )

    df = pd.DataFrame.from_dict({'time': np.hstack((elm_starts, anom_times)),
                                 'score': np.hstack((elm_scores, anom_scores)),
                                 'elm_not_anom': elm_not_anom
                                 }).sort_values('time', ignore_index=True)
    return t_run, df


def make_plot(location: str,
              figure_name: str,
              data_dict: dict,
              scores: np.array,
              run_time: np.array) -> None:
    fig, ax = plt.subplots(3, 1, sharex='col', squeeze=True, figsize=(6, 8))
    ax[0].plot(data_dict['time'], data_dict['signal'])

    ax[1].plot(data_dict['time'], scores)
    ax[1].set_yscale('log')
    ax[1].set_ylim([1e-2, 1.0])
    yticks = [0.1, 0.5, 1.0]
    ax[1].set_yticks(yticks, labels=['{:2.1f}'.format(x) for x in yticks])

    ax[2].scatter(data_dict['time'], run_time * 1e6, s=1, marker='.')
    ax[2].set_yscale('log')

    ax[0].set_title('Run : {:s}'.format(figure_name))
    ax[0].set_ylabel('signal (V)')
    ax[1].set_ylabel('anomaly score')
    ax[2].set_ylabel(r'run time ($\mu$s)')
    ax[2].set_xlabel('shot time (ms)')

    plt.savefig(location + '/' + figure_name, dpi=300)
    plt.close(fig)


def make_dataframe(location: str,
                   frame_name: str,
                   data_dict: dict,
                   scores: np.array
                   ) -> None:
    elms = np.abs(data_dict['signal']) > 1.0
    fudge_factor = 100
    win = np.ones(fudge_factor)
    elms = (np.convolve(elms, win) > 0).astype(bool)[:-(fudge_factor - 1)]

    elm_starts_mask = np.hstack(([0, np.diff(elms.astype(int))])) > 0
    elm_starts = data_dict['time'][elm_starts_mask]
    elm_scores = scores[elm_starts_mask]

    # anomaly candidates have score greater than 0.01 - permissive
    anomaly_candidates = scores > 0.01
    anomalies = np.logical_and(anomaly_candidates, np.logical_not(elms))
    anom_times = data_dict['time'][anomalies]
    anom_scores = scores[anomalies]
    elm_not_anom = np.hstack((np.ones_like(elm_scores).astype(bool),
                              np.zeros_like(anom_scores).astype(bool))
                             )

    df = pd.DataFrame.from_dict({'time': np.hstack((elm_starts, anom_times)),
                                 'score': np.hstack((elm_scores, anom_scores)),
                                 'elm_not_anom': elm_not_anom
                                 }).sort_values('time', ignore_index=True)

    df.to_csv(location + '/' + frame_name + '.csv', index=False)
