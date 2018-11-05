'''
October 2018

Tests to ensure that at a fixed reward of 1  the sig_asym_reward model produces
the same likelihood values for given data as the sig_sym model
'''

import numpy as np
import sys
from pathlib import Path
import multiprocess as mulpro
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
from fine_grain_model import FineGrained
from bellman_utilities import BellmanUtil
from observers import ObserverSim
from data_and_likelihood import DataLikelihoods

try:
    subject_num = sys.argv[1]
    if not subject_num.isnumeric():
        subject_num = 1
        print('Invalid subject number passed at prompt. Setting subject to 1')
except IndexError:
    subject_num = 1
    print('No subject number passed at prompt. Setting subject to 1')

datapath = str(Path('~/Documents/fit_data/').expanduser())
filepath = datapath + '/subject_{}_sig_reward_asym_reward_const_modelfit.p'.format(subject_num)
fitdata = np.load(filepath)
optlikelihoods = fitdata['likelihoods_returned']
testparams = fitdata['tested_params']
bestsig, bestreward = testparams[np.argmin(optlikelihoods)]
size = 100
model_params = {'T': 10,
                'dt': 0.05,
                't_w': 0.5,
                'size': size,
                'lapse': 1e-6,
                'N_values': (8, 12, 16),
                'g_values': np.linspace(1e-4, 1 - 1e-4, size),
                'subject_num': subject_num,
                'fine_sigma': np.exp(bestsig),
                'reward': np.exp(bestreward),
                'punishment': 0,
                'reward_scheme': 'asym_reward',
                'numsims': 45000}


def get_dists_single_N(curr_params):
    bellutil = BellmanUtil(**curr_params)
    curr_params['decisions'] = bellutil.decisions

    obs = ObserverSim(**curr_params)
    curr_params['dist_matrix'] = obs.dist_matrix
    curr_params['rts_matrix'] = obs.rts_matrix
    return curr_params


def single_run(model_params):
    curr_model = deepcopy(model_params)
    finegr = FineGrained(**curr_model)
    coarse_stats = finegr.coarse_stats

    N_blocked_params = []
    for i in range(len(curr_model['N_values'])):
        curr_params = deepcopy(curr_model)
        curr_params['mu'] = coarse_stats[i, :, 0]
        curr_params['sigma'] = coarse_stats[i, :, 1]
        curr_params['N'] = curr_params['N_values'][i]
        N_blocked_params.append(curr_params)

    pool = mulpro.Pool(processes=3)
    dist_computed_params = pool.map(get_dists_single_N, N_blocked_params)
    pool.close()
    pool.join()

    likelihood_data = DataLikelihoods(**curr_model)
    for single_N_params in dist_computed_params:
        likelihood_data.increment_likelihood(**single_N_params)

    return likelihood_data.likelihood


model_runs = []
for iter in range(200):
    model_runs.append(single_run(model_params))

savepath = str(Path('~/Documents/fit_data/fit_tests/').expanduser())
sns.kdeplot(model_runs, shade=True, label='Optimal param evaluations')
plt.legend()
plt.savefig(savepath + '/subject_{}_opt_param_run_var.png'.format(subject_num))
