import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
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
                'fine_sigma': 0.83,
                'reward': 0.60,
                'punishment': -3.,
                'fine_model': 'const',
                'reward_scheme': 'asym_reward',
                }

finegr = FineGrained(**model_params)
model_params['coarse_stats'] = finegr.coarse_stats

N_values = model_params['N_values']
dist_computed_params = []
for i, N in enumerate(N_values):
    curr_params = deepcopy(model_params)
    curr_params['N'] = N
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

T = model_params['T']
dt = model_params['dt']
fig, axes = plt.subplots(3, 1)
t_values = np.arange(0, T, dt) + (dt / 2)

for i, N in enumerate(N_values):
    currfracs = dist_computed_params[i]['fractions']
    curr_abs_normed = currfracs[0][0, :] / np.sum(currfracs[0][0, :] * dt)
    curr_incorr_during_abs = np.sum(currfracs[0][1, :])
    curr_pres_normed = currfracs[1][1, :] / np.sum(currfracs[1][1, :] * dt)
    curr_incorr_during_pres = np.sum(currfracs[1][0, :])
    axes[i].plot(t_values, curr_abs_normed, lw=2, color='purple', label='Correct Abs')
    axes[i].plot(t_values, curr_pres_normed, lw=2, color='orange', label='Correct pres')
    axes[i].set_title('N = {}, when C = 0, {:.2%} incorr, when C = 1, {:.2%} incorr'.format(
        N, curr_incorr_during_abs, curr_incorr_during_pres))
