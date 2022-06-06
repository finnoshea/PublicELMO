import os
import json
import pandas as pd
import numpy as np


RESULTSDIR = '/sdf/home/f/foshea/Plasma/containers/' \
             'brainblocks/shared_volume/results'

CONFIGS = []


def format_parameters(json_file: str) -> dict:
    with open(json_file, 'r') as jf:
        dd = json.load(jf)
    params = {}
    for key, value in dd.items():
        for k, v in value.items():
            name = key + '__' + k
            params[name] = v
    return params


for direc in os.listdir(RESULTSDIR):
    if 'reset' in direc:
        full_path = os.path.join(RESULTSDIR, direc)
        for f in os.listdir(full_path):
            if 'csv' in f:
                fn = os.path.join(full_path, f)
                p = format_parameters(fn.replace('csv', 'json'))
                df = pd.read_csv(fn)
                p['elm_count'] = df['elm_not_anom'].sum()
                p['anom_count'] = df.shape[0] - p['elm_count']
                CONFIGS.append(p)

df = pd.DataFrame(CONFIGS)
df.to_csv('reset_results.csv', index=False)