import numpy as np
from fine_grain_model import FineGrained
from bellman_utilities import BellmanUtil
from observers import ObserverSim
from data_and_likelihood import DataLikelihoods
from copy import deepcopy
from pathlib import Path
import pickle
import itertools as it
from multiprocess import Pool

subject_num = 1
size = 100
model_params = {'T': 10,
                'dt': 0.05,
                't_w': 0.5,
                'size': size,
                'lapse': 1e-6,
                'N_values': (8, 12, 16),
                'g_values': np.linspace(1e-4, 1 - 1e-4, size),
                'subject_num': subject_num,
                'fine_model': 'const',
                'reward_scheme': 'asym_reward',
                'punishment': 0,
                'numsims': 55000}


def comparisons(arglist):
    fine_sigma, i = arglist[0]
    reward, j = arglist[1]
    print(i, j)
    model_params['fine_sigma'] = fine_sigma
    model_params['reward'] = reward
    finegr = FineGrained(**model_params)
    coarse_stats = finegr.coarse_stats
    model_params['mu'] = coarse_stats[1, :, 0]
    model_params['sigma'] = coarse_stats[1, :, 1]
    model_params['N'] = model_params['N_values'][1]
    bellutil = BellmanUtil(**model_params)
    model_params['decisions'] = bellutil.decisions

    sim_params = deepcopy(model_params)
    sim_params['simulate'] = True
    fp_params = deepcopy(model_params)

    fpobs = ObserverSim(**fp_params)
    simobs = ObserverSim(**sim_params)

    sim_likelihood = DataLikelihoods(subject_num)
    fp_likelihood = DataLikelihoods(subject_num)

    fp_likelihood.increment_likelihood(fpobs.fractions, **fp_params)
    sim_likelihood.increment_likelihood_legacy(simobs.dist_matrix,
                                               simobs.rts_matrix,
                                               **sim_params)

    fp_value = fp_likelihood.likelihood
    sim_value = sim_likelihood.likelihood
    return (fp_value, sim_value, i, j)


fp_likelihoods = np.zeros((100, 100))
sim_likelihoods = np.zeros((100, 100))
sigmas = np.linspace(0.2, 3, 100)
rewards = np.linspace(0.5, 1.5, 100)

arglists = it.product(zip(sigmas, range(100)), zip(rewards, range(100)))

pool = Pool(processes=22)
output_likelihoods = pool.map(comparisons, arglists)

for output in output_likelihoods:
    i, j = output[2:]
    fp_likelihoods[i, j] = output[0]
    sim_likelihoods[i, j] = output[1]

fw = open('/home/berk/Documents/fp_likelihoods_test.p', 'wb')
outdict = {'fp_likelihoods': fp_likelihoods, 'sim_likelihoods': sim_likelihoods}
pickle.dump(outdict, fw)
fw.close()
