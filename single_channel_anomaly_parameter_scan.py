import os
import shutil
import json

from parameter_scan_utils import (load_data,
                                  yield_permutations,
                                  train_permutation,
                                  reset_train_permutation,
                                  make_plot,
                                  make_dataframe)


if __name__ == "__main__":
    from datetime import datetime
    import logging
    import argparse

    parser = argparse.ArgumentParser(description='Run an HTM simulation.')
    parser.add_argument('-n', '--name', type=str, dest='name', required=True,
                        help='name of the run, used in directory naming')
    parser.add_argument('-st', '--start-time', type=int, dest='start_time',
                        help='Start time of the data to learn on, ms'
                        )
    parser.add_argument('-et', '--end-time', type=int, dest='end_time',
                        help='End time of the data to learn on, ms'
                        )
    parser.add_argument('-s', '--statelets', type=int, dest='num_statelets',
                        help='The number of statelets in the ScalarTransformer'
                        )
    help_str = 'The number of active statelets in the ScalarTransformer'
    parser.add_argument('-a', '--active', type=int, dest='num_active',
                        help=help_str
                        )
    parser.add_argument('-r', '--reset', dest='reset',
                        action='store_true', default=False,
                        help='Whether to run in reset mode'
                        )
    args = parser.parse_args()

    dest = os.path.join(os.getcwd(), 'results', args.name)
    os.makedirs(dest, exist_ok=True)
    shutil.copy(__file__,
                os.path.join(dest, __file__.split(os.path.sep)[-1]))

    t = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
    log_str = 'bb_param_scan_{:s}_{:s}.log'.format(args.name, t)
    log_name = os.path.join(dest, log_str)
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                        filename=log_name,
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')

    logging.info('Run name: {:s}'.format(args.name))

    scan_dict = {
        'data__input_file': ['/usr/src/app/shared_volume/ecebes_166434.h5'],
        'data__data_name': ['BESFU/BESFU27'],
        'data__start_time': [2700],
        'data__end_time': [3000],
        'data__abs': [False],
        'st__min_val': [-11.0],
        'st__max_val': [11.0],
        'st__num_s': [256],  # [64, 128, 256]
        'st__num_as': [16],  # [4, 8, 16, 32]
        'sl__num_spc': [8, 16],
        'sl__num_dps': [8, 16, 32],
        'sl__num_rpd': [8, 16],
        'sl__d_thresh': [2, 4, 8, 16],
        'sl__perm_thr': [8, 16, 32],
        'sl__perm_inc': [1, 2, 3],
        'sl__perm_dec': [1, 2, 3]
    }

    adjust_these = {'num_statelets': 'st__num_s',
                    'num_active': 'st__num_as',
                    'start_time': 'data__start_time',
                    'end_time': 'data__end_time'
                    }
    for k, v in adjust_these.items():
        scan_dict[v] = [getattr(args, k)]

    logging.info('Parameters given:')
    for key, value in scan_dict.items():
        logging.info('{:s} : {:s}'.format(key, str(value)))

    count = 1
    for v in scan_dict.values():
        count *= len(v)
    logging.info('Number of runs to perform: {:d}'.format(count))

    for idx, p in enumerate(yield_permutations(scan_dict)):
        run_str = 'bb_run_{:s}_{:04d}'.format(args.name, idx)
        with open(os.path.join(dest, run_str + '.json'), 'w') as fp:
            json.dump(p, fp, sort_keys=True, indent=3)
        try:
            data_dict = load_data(p['data'])
            if args.reset:
                t_run, df = reset_train_permutation(params=p)
                df.to_csv(dest + '/' + run_str + '.csv', index=False)
            else:
                t_run, scores, run_time = train_permutation(params=p,
                                                            data=data_dict)
                # make_plot(location=dest, figure_name=run_str,
                #           data_dict=data_dict, scores=scores,
                #           run_time=run_time)
                make_dataframe(location=dest, frame_name=run_str,
                               data_dict=data_dict, scores=scores)
        except RuntimeError as e:
            logging.warning('failed to create {:s}'.format(run_str))
            logging.warning(e)
        else:
            logging.info('trained {:s} in {:2.1f} seconds'.format(run_str,
                                                                  t_run))



