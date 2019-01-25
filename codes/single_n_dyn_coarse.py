'''
December 2018

Optimization of the coarse-grained model directly for sigma and mu,
bypassing the fine-grained model
'''

import numpy as np
import sys
from pathlib import Path
from gauss_opt import bayesian_optimisation
from bellman_utilities import BellmanUtil
from observers import ObserverSim
from copy import deepcopy
from data_and_likelihood import DataLikelihoods
import pickle

presamp = 20
num_samples = 980
savepath = Path("~/Documents/fit_data/")  # Where to save figures
savepath = str(savepath.expanduser())
N_index = 1
N_value = 12


def subject_likelihood(likelihood_arglist):
    """
    First handle the different cases of what we need to fit and save parameters. They just set
    the appropriate values of fine_sigma, reward, and punishment depending on the model type
    and print what is being evaluated by the optimization algorithm."""
    log_parameters, model_params = likelihood_arglist

    curr_params = deepcopy(model_params)
    # fine_sigma and punishment are fit, reward fixed at 1
    sigma = np.array((np.exp(log_parameters[0]), np.exp(log_parameters[1])))
    reward = np.exp(log_parameters[2])
    punishment = np.exp(log_parameters[3])
    alpha = np.exp(log_parameters[4])
    print('sigmas = {:.2f}, {:.2f}'.format(*sigma), '; reward = {:.2f}'.format(reward),
          '; punishment = {:.2f}'.format(punishment), '; alpha = {:.2f}'.format(alpha))

    curr_params['sigma'] = sigma
    curr_params['reward'] = reward
    curr_params['punishment'] = -punishment
    curr_params['alpha'] = alpha

    bellutil = BellmanUtil(**curr_params)
    curr_params['rho'] = bellutil.rho
    curr_params['decisions'] = bellutil.decisions

    obs = ObserverSim(**curr_params)
    curr_params['fractions'] = obs.fractions

    likelihood_data = DataLikelihoods(**curr_params)
    likelihood_data.increment_likelihood(**curr_params)
    print(likelihood_data.likelihood)
    return likelihood_data.likelihood


def modelfit(model_params):
    def likelihood_for_opt(log_parameters):
        likelihood_arglist = log_parameters, model_params
        return subject_likelihood(likelihood_arglist)

    bounds = np.array(((0.5, 2.5),
                       (1.5, 3.5),
                       (0.4, 1.1),
                       (2.5, 20.),
                       (0.8, 4.5)))
    log_bounds = np.log(bounds)
    x_opt = bayesian_optimisation(n_iters=num_samples, sample_loss=likelihood_for_opt,
                                  bounds=log_bounds, n_pre_samples=presamp)

    model_params['tested_params'], model_params['likelihoods_returned'] = x_opt
    fw = open(savepath + '/subject_{}_single_N_{}_modelfit.p'.format(subject_num, N_value), 'wb')
    pickle.dump(model_params, fw)
    fw.close()
    return


if __name__ == '__main__':
    try:
        subject_num = sys.argv[1]
        if not subject_num.isnumeric():
            subject_num = 1
            print('Invalid subject number passed at prompt. Setting subject to 1')
    except IndexError:
        subject_num = 1
        print('No subject number passed at prompt. Setting subject to 1')

    size = 600
    model_params = {'T': 10,
                    'dt': 0.05,
                    't_w': 0.5,
                    't_delay': 0.2,
                    't_max': 5.,
                    'size': size,
                    'lapse': 1e-6,
                    'mu': np.array((0, 1)),
                    'N': N_value,
                    'N_values': (8, 12, 16),
                    'g_values': np.linspace(1e-4, 1 - 1e-4, size),
                    'subject_num': subject_num,
                    'reward_scheme': 'asym_reward'}

    print('Subject number {}'.format(subject_num))
    modelfit(model_params)
