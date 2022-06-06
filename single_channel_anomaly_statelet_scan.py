import numpy as np
import time
import matplotlib.pyplot as plt

from data_loader import load_data
from brainblocks.blocks import ScalarTransformer, SequenceLearner

FILENAME = '/usr/src/app/shared_volume/ecebes_166434.h5'


def makeSequenceLearner() -> SequenceLearner:
    sl = SequenceLearner(
        num_spc=10,  # number of statelets per column
        num_dps=10,  # number of dendrites per statelet
        num_rpd=12,  # number of receptors per dendrite
        d_thresh=6,  # dendrite threshold
        perm_thr=20,  # receptor permanence threshold
        perm_inc=2,  # receptor permanence increment
        perm_dec=1)  # receptor permanence decrement
    return sl


def makeScalarTransformer(num_statelets: int,
                          num_active: int
                          ) -> ScalarTransformer:
    st = ScalarTransformer(
        min_val=-11.0,          # minimum input value
        max_val=11.0,           # maximum input value
        num_s=num_statelets,    # number of statelets
        num_as=num_active)      # number of active statelets
    return st


def makeRun(st: ScalarTransformer,
            sl: SequenceLearner,
            data_dict: dict) \
        -> (float, np.ndarray, np.ndarray):
    # Connect blocks
    # connect sequence_learner input to scalar_transformer output
    sl.input.add_child(st.output)

    scores = np.zeros_like(data_dict['signal'])
    run_time = np.zeros_like(data_dict['signal'])

    # dummy run to cut down on first run time
    st.set_value(0)
    st.feedforward()
    sl.feedforward(learn=False)
    # Loop through data
    t_start = time.time()
    for index, value in enumerate(data_dict['signal']):
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
    # return {'total_time': t_end - t_start,
    #         'scores': scores,
    #         'run_time': run_time
    #         }
    return t_end - t_start, scores, run_time


def makePlot(num_statelets: int,
             num_active: int,
             total_run_time: float,
             location: str,
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

    title = ('num_statelets: {:>4d}, '.format(num_statelets),
             'num_active: {:>4d}, '.format(num_active),
             'run time: {:3.2e}'.format(total_run_time)
             )
    ax[0].set_title(''.join(title))
    ax[0].set_ylabel('signal (V)')
    ax[1].set_ylabel('anomaly score')
    ax[2].set_ylabel(r'run time ($\mu$s)')
    ax[2].set_xlabel('shot time (ms)')

    name = 's_{:04d}_a_{:04d}_BESFU27.png'.format(num_statelets, num_active)
    plt.savefig(location + '/' + name, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    pos_dict = load_data(FILENAME, 'BESFU', 27, 2830, 2930)
    neg_dict = load_data(FILENAME, 'BESFU', 27, 5450, 5550)
    save_here = '/usr/src/app/shared_volume/images'

    exponents = [n for n in range(2, 10)]
    for exponent in exponents:
        ns = 2 ** exponent
        for na in [2 ** n for n in range(2, exponent)]:
            print('Running {:>4d} / {:>4d}'.format(ns, na))
            # data with "ELMs"
            st = makeScalarTransformer(ns, na)
            sl = makeSequenceLearner()
            pos_run_time, pos_scores, pos_time = makeRun(st, sl, pos_dict)
            # data without "ELMs"
            st = makeScalarTransformer(ns, na)
            sl = makeSequenceLearner()
            neg_run_time, neg_scores, neg_time = makeRun(st, sl, neg_dict)

            fig, ax = plt.subplots(3, 2, sharex='col',
                                   squeeze=True, figsize=(12, 8))
            # positive data
            ax[0, 0].plot(pos_dict['time'], pos_dict['signal'], '-b')
            ax[1, 0].plot(pos_dict['time'], pos_scores, 'b')
            ax[1, 0].set_yscale('log')
            ax[1, 0].set_ylim([1e-2, 1.0])
            yticks = [0.1, 0.5, 1.0]
            ax[1, 0].set_yticks(yticks,
                                labels=['{:2.1f}'.format(x) for x in yticks])
            ax[2, 0].scatter(pos_dict['time'], pos_time * 1e6,
                             s=1, marker='.', color='b')
            ax[2, 0].set_yscale('log')

            title = ('Pos. statelets: {:>3d}, '.format(ns),
                     'num_active: {:>2d}, '.format(na),
                     'run time: {:3.2e}'.format(pos_run_time)
                     )
            ax[0, 0].set_title(''.join(title))
            ax[0, 0].set_ylabel('signal (V)')
            ax[1, 0].set_ylabel('anomaly score')
            ax[2, 0].set_ylabel(r'run time ($\mu$s)')
            ax[2, 0].set_xlabel('shot time (ms)')

            # negative data
            ax[0, 1].plot(neg_dict['time'], neg_dict['signal'], '-r')
            ax[1, 1].plot(neg_dict['time'], neg_scores, 'r')
            ax[1, 1].set_yscale('log')
            ax[1, 1].set_ylim([1e-2, 1.0])
            ax[1, 1].set_yticks(yticks,
                                labels=['{:2.1f}'.format(x) for x in yticks])
            ax[2, 1].scatter(neg_dict['time'], neg_time * 1e6,
                             s=1, marker='.', color='r')
            ax[2, 1].set_yscale('log')

            title = ('Neg. statelets: {:>3d}, '.format(ns),
                     'num_active: {:>2d}, '.format(na),
                     'run time: {:3.2e}'.format(neg_run_time)
                     )
            ax[0, 1].set_title(''.join(title))
            ax[2, 1].set_xlabel('shot time (ms)')

            name = 's_{:04d}_a_{:04d}_BESFU27.png'.format(ns, na)
            plt.savefig(save_here + '/' + name, dpi=300)
            plt.close(fig)
