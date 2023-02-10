import time
import json
import h5py
import numpy as np
from elm_finder import find_big_elms_easy, find_and_label_elms
from data_loader import SMITHSHOTS


def compare_finder_and_smith(shot: h5py.Group,
                             quantile: float = 0.95,
                             bes_thresh: float = 1.0) -> dict:
    # _, _, df = find_big_elms_easy(elm_file=shot,
    #                               start_time=shot['time'][0],
    #                               end_time=shot['time'][-1],
    #                               quantile=quantile,
    #                               bes_thresh=bes_thresh
    #                               )
    _, _, df = find_and_label_elms(elm_file=shot,
                                   start_time=shot['time'][0],
                                   end_time=shot['time'][-1],
                                   quantile=quantile,
                                   bes_thresh=bes_thresh
                                   )
    # df comes after an np.diff, so it is one spot shorter than labels
    labels = shot['labels'][:-1]
    tp = int(np.logical_and(df['elms'], labels).any())
    fn = int(np.logical_not(df['elms'])[labels].all())
    # this is the classical way
    # fp = int(np.logical_and(df['elms'], np.logical_not(labels)).any())

    # smooth out the spikes that don't tell us anything
    smooth = 100  # 100 microseconds
    pf = df['elms'].values.copy()
    # starts = np.arange(pf.shape[0])[pf]
    # for s in starts:
    #     if s + smooth >= len(pf):
    #         end = len(pf)
    #     else:
    #         end = s + smooth
    #     pf[s:end] = np.ones(end - s)

    pfp = np.logical_or(pf, labels).astype(int)  # possible fps
    # count the number of rising edges and subtract the expected 1
    fp = int((np.diff(pfp) > 0).sum() - 1)

    return {'tp': tp, 'fp': fp, 'fn': fn}


def compute_stats(quantile: float = 0.95,
                  bes_thresh: float = 1.0) -> dict:
    stats = {'number': 0, 'tp': 0, 'fp': 0, 'fn': 0}
    with h5py.File(SMITHSHOTS, 'r') as hdf:
        for shot_value in hdf.values():
            rd = compare_finder_and_smith(shot=shot_value,
                                          quantile=quantile,
                                          bes_thresh=bes_thresh)
            stats['number'] += 1
            for kk, vv in rd.items():
                stats[kk] += vv
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate an ELM finder.')
    parser.add_argument('-q', '--quantile', type=float, dest='quantile',
                        default=0.95,
                        help='Quantile above which to declare an ELM.'
                        )
    parser.add_argument('-b', '--bes-thresh', type=float, dest='bes_thresh',
                        default=1.0,
                        help='Threshold above which to declare an ELM in BES.'
                        )
    args = parser.parse_args()

    run_times = {}

    t0 = time.time()
    cat_stats = compute_stats(quantile=args.quantile,
                              bes_thresh=args.bes_thresh
                              )
    cat_stats['quantile'] = args.quantile
    cat_stats['bes_thresh'] = args.bes_thresh
    run_times['get_stats'] = (time.time() - t0) / cat_stats['number']
    with open('quantile_stats.json', 'r') as jf:
        all_stats = json.load(jf)
    all_stats[len(all_stats)] = cat_stats
    with open('quantile_stats.json', 'w') as jf:
        json.dump(all_stats, jf, indent=3)

    print('Run times:')
    total = 0
    for k, v in run_times.items():
        print('{0:15s} : {1:>2.1e} sec'.format(k, v))
        total += v
    print('Total run time: {0:>2.1e} sec'
          .format(run_times['get_stats'] * cat_stats['number']))
