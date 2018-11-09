'''
August 2018

Class based implementation of the dynamic model

September 2018

You can thank the multiprocessing module for how fucking ugly this is. --Berk
'''

import numpy as np
import sys
from pathlib import Path
from copy import deepcopy
import multiprocess as mulpro
import itertools as it
from gauss_opt import bayesian_optimisation
from fine_grain_model import FineGrained
from bellman_utilities import BellmanUtil
from observers import ObserverSim
from data_and_likelihood import DataLikelihoods
import pickle

num_samples = 100
savepath = Path("~/Documents/")  # Where to save figures
savepath = str(savepath.expanduser())
gridsearch = True


def likelihood_inner_loop(curr_params):
    bellutil = BellmanUtil(**curr_params)
    curr_params['decisions'] = bellutil.decisions

    obs = ObserverSim(**curr_params)
    curr_params['fractions'] = obs.fractions
    return curr_params


def subject_likelihood(likelihood_arglist):
    """
    First handle the different cases of what we need to fit and save parameters. They just set
    the appropriate values of fine_sigma, reward, and punishment depending on the model type
    and print what is being evaluated by the optimization algorithm."""
    model_type, log_parameters, model_params = likelihood_arglist

    # Handling what to optimize and what to hold constant
    if model_type[0] == 'sig':
        # Only fine_sigma is fit
        fine_sigma = np.exp(log_parameters[0])
        print('fine_sigma =', fine_sigma, '|| model_type = ', model_type)

        model_params['fine_sigma'] = fine_sigma
        model_params['reward'] = 1
        model_params['punishment'] = 0
    elif model_type[0] == 'sig_reward':
        # fine_sigma and reward are fit, punishment fixed at 0
        fine_sigma = np.exp(log_parameters[0])
        reward = np.exp(log_parameters[1])
        print('fine_sigma =', fine_sigma, '| reward =', reward, '|| model_type = ', model_type)

        model_params['fine_sigma'] = fine_sigma
        model_params['reward'] = reward
        model_params['punishment'] = 0
    elif model_type[0] == 'sig_punish':
        # fine_sigma and punishment are fit, reward fixed at 1
        fine_sigma = np.exp(log_parameters[0])
        punishment = np.exp(log_parameters[1])
        print('fine_sigma =', fine_sigma, '| punishment =', punishment,
              '|| model_type = ', model_type)

        model_params['fine_sigma'] = fine_sigma
        model_params['reward'] = 1
        model_params['punishment'] = punishment

    finegr = FineGrained(**model_params)
    coarse_stats = finegr.coarse_stats

    N_blocked_model_params = []
    for i in range(len(model_params['N_values'])):
        curr_params = deepcopy(model_params)
        curr_params['mu'] = coarse_stats[i, :, 0]
        curr_params['sigma'] = coarse_stats[i, :, 1]
        curr_params['N'] = curr_params['N_values'][i]
        N_blocked_model_params.append(curr_params)

    pool = mulpro.Pool(processes=3)
    dist_computed_params = pool.map(likelihood_inner_loop, N_blocked_model_params)
    pool.close()
    pool.join()

    likelihood_data = DataLikelihoods(**model_params)
    for single_N_params in dist_computed_params:
        likelihood_data.increment_likelihood(**single_N_params)

    return likelihood_data.likelihood


def modelfit(arglist):
    model_type, model_params = arglist
    model_params['fine_model'] = model_type[2]
    model_params['reward_scheme'] = model_type[1]
    model_params['opt_type'] = model_type[0]

    if not gridsearch:
        def likelihood_for_opt(log_parameters):
            likelihood_arglist = model_type, log_parameters, model_params
            return subject_likelihood(likelihood_arglist)

        if model_type[0] == 'sig':
            bnds = np.array(((-1.7, 2.5),))  # [n_variables, 2] shaped array with bounds
            x_opt = bayesian_optimisation(n_iters=num_samples, sample_loss=likelihood_for_opt,
                                          bounds=bnds, n_pre_samples=50)

        if model_type[0] == 'sig_reward':
            bnds = np.array(((-1.7, 2.5), (-1., 0.5)))  # [n_variables, 2] shaped array with bounds
            x_opt = bayesian_optimisation(n_iters=num_samples, sample_loss=likelihood_for_opt,
                                          bounds=bnds, n_pre_samples=50)

        if model_type[0] == 'sig_punish':
            bnds = np.array(((-1.7, 2.5), (-5., 1.)))  # [n_variables, 2] shaped array with bounds
            x_opt = bayesian_optimisation(n_iters=num_samples, sample_loss=likelihood_for_opt,
                                          bounds=bnds, n_pre_samples=50)
    else:
        dim_size = int(np.sqrt(num_samples))

        def likelihood_for_opt(log_parameters):
            likelihood_arglist = model_type, log_parameters, model_params
            return subject_likelihood(likelihood_arglist)

        if model_type[0] == 'sig':
            bnds = np.array(((1, 5),))
            log_param_list = np.log(np.linspace(bnds[0][0], bnds[0][1], num_samples))
            likelihoods_returned = np.zeros_like(log_param_list)
            for idx, log_param in enumerate(log_param_list):
                likelihoods_returned[idx] = subject_likelihood([model_type, (log_param, ),
                                                                model_params])
            x_opt = (log_param_list, likelihoods_returned)

        if model_type[0] == 'sig_reward':
            bnds = np.array(((0.1, 1.5), (0.6, 1.5)))  # [n_variables, 2] shaped array with bounds
            log_sigma_list = np.log(np.linspace(bnds[0][0], bnds[0][1], dim_size))
            log_reward_list = np.log(np.linspace(bnds[1][0], bnds[1][1], dim_size))
            log_param_pairs = np.array(list(it.product(log_sigma_list, log_reward_list)))
            likelihoods_returned = np.zeros(log_param_pairs.shape[0])
            for idx, log_params in enumerate(log_param_pairs):
                likelihoods_returned[idx] = subject_likelihood([model_type, log_params,
                                                                model_params])
            x_opt = (log_param_pairs, likelihoods_returned)

        if model_type[0] == 'sig_punish':
            bnds = np.array(((0.1, 1.5), (0.05, 0.8)))  # [n_variables, 2] shaped array with bounds
            log_sigma_list = np.log(np.linspace(bnds[0][0], bnds[0][1], dim_size))
            log_punish_list = np.log(np.linspace(bnds[1][0], bnds[1][1], dim_size))
            log_param_pairs = np.array(list(it.product(log_sigma_list, log_punish_list)))
            likelihoods_returned = np.zeros(log_param_pairs.shape[0])
            for idx, log_params in enumerate(log_param_pairs):
                likelihoods_returned[idx] = subject_likelihood([model_type, log_params,
                                                                model_params])
            x_opt = (log_param_pairs, likelihoods_returned)

    model_params['tested_params'], model_params['likelihoods_returned'] = x_opt
    fw = open(savepath + '/subject_{}_{}_{}_{}_modelfit.p'.format(subject_num, *model_type), 'wb')
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

    size = 100
    model_params = {'T': 10,
                    'dt': 0.05,
                    't_w': 0.5,
                    'size': size,
                    'lapse': 1e-6,
                    'N_values': (8, 12, 16),
                    'g_values': np.linspace(1e-4, 1 - 1e-4, size),
                    'subject_num': subject_num}

    print('Subject number {}'.format(subject_num))

    model_list = [

        ('sig', 'sym', 'const'),
        # ('sig_reward', 'asym_reward', 'const'),
        # ('sig_punish', 'epsilon_punish', 'const'),
        ('sig', 'sym', 'sqrt'),
        # ('sig_reward', 'asym_reward', 'sqrt'),
        # ('sig_punish', 'epsilon_punish', 'sqrt'),
        ]

    master_arglists = [(model, model_params) for model in model_list]
    processes = [None] * len(model_list)
    for proc_num in range(len(model_list)):
        processes[proc_num] = mulpro.Process(target=modelfit, args=(master_arglists[proc_num],))
        processes[proc_num].start()

    for proc_num in range(len(model_list)):
        processes[proc_num].join()
