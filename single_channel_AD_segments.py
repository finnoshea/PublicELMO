import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import load_data
from brainblocks.blocks import ScalarTransformer, SequenceLearner

FILENAME = '/usr/src/app/shared_volume/ecebes_166434.h5'


data_start = 2670  # 3050
data_end = 3040  # 3420
data_dict = load_data(FILENAME, 'BESFU', 27, data_start, data_end)
# data_dict = load_data(FILENAME, 'BESFU', 27, 5450, 5550)


# compute some basic stuff that is hopefully informative
# elms are signals greater than 1 V
elms = np.abs(data_dict['signal']) > 1.0
fudge_factor = 100
win = np.ones(fudge_factor)
elms = (np.convolve(elms, win) > 0).astype(bool)[:-(fudge_factor - 1)]
# this is slightly slower, but more compact
elm_starts_mask = np.hstack(([0, np.diff(elms.astype(int))])) > 0
elm_starts = data_dict['time'][elm_starts_mask]

def init_blocks():
    # Setup blocks
    st = ScalarTransformer(
        min_val=-11.0,  # minimum input value
        max_val=11.0,   # maximum input value
        num_s=256,       # number of statelets
        num_as=8)       # number of active statelets

    sl = SequenceLearner(
        num_spc=16,   # number of statelets per column      10
        num_dps=8,   # number of dendrites per statelet    10
        num_rpd=8,   # number of receptors per dendrite    12
        d_thresh=2,   # dendrite threshold                  6
        perm_thr=16,  # receptor permanence threshold       20
        perm_inc=3,   # receptor permanence increment       2
        perm_dec=1)   # receptor permanence decrement       1

    # Connect blocks
    # connect sequence_learner input to scalar_transformer output
    sl.input.add_child(st.output)
    return st, sl


t_run = 0
scores = []
run_time = []

intervals = np.hstack(
    (np.array(data_start),
     np.repeat(elm_starts, 2),
     np.array(data_end))
).reshape(-1, 2)


for (start_time, end_time) in intervals:
    dd = load_data(FILENAME, 'BESFU', 27, start_time, end_time)
    seg_scores = np.zeros_like(dd['signal'])
    seg_run_time = np.zeros_like(dd['signal'])
    # reset the HTM elements
    st, sl = init_blocks()
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

scores = np.hstack(scores)
run_time = np.hstack(run_time)
print('Time to run {:d} steps: {:2.1e} seconds'.format(scores.shape[0], t_run))

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



plt.plot(data_dict['time'], data_dict['signal'])
ax = plt.gca()
for at in anom_times:
    ax.axvline(at, linestyle='dashed', color='k', linewidth=1)
for el in elm_starts:
    ax.axvline(el, linestyle='dashed', color='r', linewidth=1)
plt.savefig('test.png', dpi=300)
plt.close()


# if False:
#     fig, ax = plt.subplots(3, 1, sharex='col', squeeze=True, figsize=(6, 8))
#     ax[0].plot(data_dict['time'], data_dict['signal'])
#
#     ax[1].plot(data_dict['time'], scores)
#     ax[1].set_yscale('log')
#     ax[1].set_ylim([1e-2, 1.0])
#     yticks = [0.1, 0.5, 1.0]
#     ax[1].set_yticks(yticks, labels=['{:2.1f}'.format(x) for x in yticks])
#
#     ax[2].scatter(data_dict['time'], run_time * 1e6, s=1, marker='.')
#     ax[2].set_yscale('log')
#
#     shot_time = (data_dict['time'][-1] - data_dict['time'][0]) * 1e-3
#     run_ratio = (t_end - t_start) / shot_time
#     title = ('run time ratio: {:2.1e}'.format(run_ratio))
#     ax[0].set_title(title)
#     ax[0].set_ylabel('signal (V)')
#     ax[1].set_ylabel('anomaly score')
#     ax[2].set_ylabel(r'run time ($\mu$s)')
#     ax[2].set_xlabel('shot time (ms)')
#
#     plt.savefig('bb_anomaly_detection.png', dpi=300)






