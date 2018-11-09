import numpy as np
from copy import deepcopy
from fine_grain_model import FineGrained
from bellman_utilities import BellmanUtil
from observers import ObserverSim
from data_and_likelihood import DataLikelihoods

size = 500
model_params = {'T': 10,
                'dt': 0.05,
                't_w': 0.5,
                'size': size,
                'lapse': 1e-6,
                'N_values': (8, 12, 16),
                'g_values': np.linspace(1e-4, 1 - 1e-4, size),
                'subject_num': 1,
                'fine_sigma': 0.7,
                'fine_model': 'const',
                'reward_scheme': 'sym',
                }

finegr = FineGrained(**model_params)
model_params['coarse_stats'] = finegr.coarse_stats

N_values = model_params['N_values']
dist_computed_params = []
for i, N in enumerate(N_values):
    curr_params = deepcopy(model_params)
    curr_params['mu'] = model_params['coarse_stats'][i, :, 0]
    curr_params['sigma'] = model_params['coarse_stats'][i, :, 1]
    bellutil = BellmanUtil(**curr_params)
    curr_params['decisions'] = bellutil.decisions
    obs = ObserverSim(**curr_params)
    curr_params['fractions'] = obs.fractions
    dist_computed_params.append(curr_params)

data_eval = DataLikelihoods(**model_params)
for single_N_params in dist_computed_params:
    data_eval.increment_likelihood(**single_N_params)

print(data_eval.likelihood)
