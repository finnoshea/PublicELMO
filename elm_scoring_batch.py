import json
import time
from datetime import datetime
import logging
import numpy as np
from elm_scoring import compute_stats


t = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
log_str = 'quantile_computation_{:s}.log'.format(t)
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                    filename=log_str,
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')


fn = 'quantile_stats.json'
thresholds = [0.5, 1.0, 2.0]
quantiles = np.hstack((np.arange(0.2, 0.96, 0.05),
                       np.arange(0.96, 0.98, 0.01),
                       np.arange(0.99, 0.999, 0.001)
                       ))


def append_to_file(stats: dict, filename: str = fn) -> None:
    with open(filename, 'r') as jf:
        all_stats = json.load(jf)
    all_stats[len(all_stats)] = stats
    with open(filename, 'w') as jf:
        json.dump(all_stats, jf, indent=3)


num = 1
tot_num = len(thresholds) * len(quantiles)
for t in thresholds:  # bes_threshold
    for q in quantiles:  # quantiles
        t0 = time.time()
        cat_stats = compute_stats(quantile=q, bes_thresh=t)
        cat_stats['quantile'] = q
        cat_stats['bes_thresh'] = t
        append_to_file(cat_stats, fn)
        t_run = time.time() - t0
        logging.info('Just finished {0:>3d} / {1:>3d} in {2:3.2e} seconds'
                     .format(num, tot_num, t_run))
        num += 1
