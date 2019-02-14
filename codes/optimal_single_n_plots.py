import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from gauss_opt import bayesian_optimisation
from bellman_utilities import BellmanUtil
from observers import ObserverSim
from copy import deepcopy
from data_and_likelihood import DataLikelihoods
import pickle

presamp = 20
num_samples = 980
savepath = Path("~/Documents/fit_data/")  # Where to save figures
loadpath = Path("~/Documents/fit_data/single_N/")
savepath = str(savepath.expanduser())
experiment = 'exp2abs'
if experiment[3] == '2':
    subjects = range(1, 13)
else:
    subjects = range(1, 12)

for N in (8, 12, 16):
    for subject in subjects:

        N_value = N
        subject_num = subject

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
                        'reward_scheme': 'asym_reward',
                        'experiment': experiment}
        dt = model_params['dt']
        T = model_params['T']
        t_delay = model_params['t_delay']
        t_max = model_params['t_max']
        maxind = int(t_max / dt) - 1
        t_values = np.arange(0, model_params['T'], model_params['dt'])

        filename = '/home/berk/Documents/fit_data/single_N/subject_{}_{}_single_N_{}_modelfit.p'.format(
            subject_num, experiment, N_value)
        testdata = np.load(filename)
        print('fit returned min of', np.amin(testdata['likelihoods_returned']))
        numiters = len(testdata['likelihoods_returned'])
        log_parameters = testdata['tested_params'][np.argmin(
            testdata['likelihoods_returned'])]
        curr_params = deepcopy(model_params)
        # fine_sigma and punishment are fit, reward fixed at 1
        sigma = np.array(
            (np.exp(log_parameters[0]), np.exp(log_parameters[1])))
        reward = np.exp(log_parameters[2])
        punishment = np.exp(log_parameters[3])
        alpha = np.exp(log_parameters[4])
        finalparams = np.array([*sigma, reward, punishment, alpha])
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
        presmean = np.sum(obs.fractions[1][1, :maxind] * (
            t_values[:maxind] + t_delay)) / np.sum(obs.fractions[1][1, :maxind]) + t_delay
        incpresmean = np.sum(obs.fractions[0][1, :maxind] * (
            t_values[:maxind] + t_delay)) / np.sum(obs.fractions[0][1, :maxind]) + t_delay
        absmean = np.sum(obs.fractions[0][0, :maxind] * (
            t_values[:maxind] + t_delay)) / np.sum(obs.fractions[0][0, :maxind]) + t_delay
        incabsmean = np.sum(obs.fractions[1][0, :maxind] * (
            t_values[:maxind] + t_delay)) / np.sum(obs.fractions[1][0, :maxind]) + t_delay
        testdata['meanresps'] = np.array(
            [[absmean, incabsmean], [incpresmean, presmean]])

        likelihood_data = DataLikelihoods(**curr_params)
        likelihood_data.increment_likelihood(**curr_params)
        print(likelihood_data.likelihood)
        N_data = likelihood_data.sub_data.query(
            'setsize == {}'.format(N_value))
        subj_rts = np.zeros((2, 3), dtype=object)
        subj_rts[0, 0] = N_data.query(
            'resp == 2 & target == \'Absent\'').rt.values
        subj_rts[0, 1] = N_data.query(
            'resp == 1 & target == \'Absent\'').rt.values
        subabsmean = np.mean(subj_rts[0, 0])
        abs_timeouts = len(N_data.query(
            'resp == -1 & target == \'Absent\'').rt.values)
        tot_abs = len(subj_rts[0, 0]) + len(subj_rts[0, 1]) + abs_timeouts

        subj_rts[1, 0] = N_data.query(
            'resp == 2 & target == \'Present\'').rt.values
        subj_rts[1, 1] = N_data.query(
            'resp == 1 & target == \'Present\'').rt.values
        subpresmean = np.mean(subj_rts[1, 1])
        pres_timeouts = len(N_data.query(
            'resp == -1 & target == \'Present\'').rt.values)
        tot_pres = len(subj_rts[1, 1]) + len(subj_rts[1, 0]) + pres_timeouts
        grandtotal = tot_pres + tot_abs

        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        ax[0].fill_between(t_values[:maxind] + 0.2, obs.fractions[0][0, :maxind] / np.sum(obs.fractions[0][0, :maxind]) / dt,
                           color='purple', alpha=0.5)
        ax[0].fill_between(t_values[:maxind] + 0.2, obs.fractions[1][1, :maxind] / np.sum(obs.fractions[1][1, :maxind]) / dt,
                           color='orange', alpha=0.5)
        sns.kdeplot(subj_rts[0, 0], bw=0.1, alpha=0.5,
                    shade=True, color='blue', ax=ax[0])
        sns.kdeplot(subj_rts[1, 1], bw=0.1, alpha=0.5,
                    shade=True, color='red', ax=ax[0])

        ymin, ymax = ax[0].get_ylim()
        ax[0].vlines(presmean, ymin, ymax, colors='orange', lw=2)
        ax[0].vlines(absmean, ymin, ymax, colors='purple', lw=2)
        ax[0].vlines(np.mean(subj_rts[0, 0]), ymin, ymax, color='blue', lw=2)
        ax[0].vlines(np.mean(subj_rts[1, 1]), ymin, ymax, color='red', lw=2)

        ax[0].set_xlabel('RT (s)', size=18)
        ax[0].set_xlim([0, 6])
        ax[0].set_ylim([ymin, ymax])
        ax[0].set_title('Optimal fit Sub {} Correct\n'.format(subject_num) +
                        'sig_abs = {:.2f}, sig_pres = {:.2f}, reward = {:.2f}\n'.format(*finalparams[:3]) +
                        'punishment = {:.2f}, alpha = {:.2f}'.format(*finalparams[3:]), size=18)

        ax[1].fill_between(t_values[:maxind] + 0.2, obs.fractions[0][1, :maxind] / np.sum(obs.fractions[0][1, :maxind]) / dt,
                           color='purple', alpha=0.5, label=r'Sim $\hat{C} = 1 | C = 0$')
        ax[1].fill_between(t_values[:maxind] + 0.2, obs.fractions[1][0, :maxind] / np.sum(obs.fractions[1][0, :maxind]) / dt,
                           color='orange', alpha=0.5, label=r'Sim $\hat{C} = 0 | C = 1$')
        sns.kdeplot(subj_rts[0, 1], bw=0.1, alpha=0.5, shade=True, color='blue', ax=ax[1],
                    label=r'Subj $\hat{C} = 1 | C = 0$')
        sns.kdeplot(subj_rts[1, 0], bw=0.1, alpha=0.5, shade=True, color='red', ax=ax[1],
                    label=r'Subj $\hat{C} = 0 | C = 1$')
        ax[1].set_title('Optimal fit sub {} Incorrect\n'.format(subject_num) +
                        'Resp pres | abs: Subj {:.2%} Sim {:.2%}\n'.format(len(subj_rts[0, 1]) / tot_abs,
                                                                           np.sum(obs.fractions[0][1, :maxind])) +
                        'Resp abs | pres: Subj {:.2%} Sim {:.2%}\n'.format(len(subj_rts[1, 0]) / tot_pres,
                                                                           np.sum(obs.fractions[1][0, :maxind])) +
                        'Timeout: Subj {:.2%} Sim {:.2%}'.format((abs_timeouts + pres_timeouts) / grandtotal,
                                                                 obs.fractions[0][2, maxind] * 0.5 + obs.fractions[1][2, maxind] * 0.5),
                        size=18)

        plt.tight_layout()
        plt.savefig('/home/berk/Documents/single_N_subject_{}_N_{}_{}_1000_iter_optfit.png'.format(
            subject_num, N_value, experiment), DPI=500)
        plt.close()
        meanrho = 1 / presmean * np.sum(obs.fractions[1][1, :]) * 0.5 +\
            (-punishment / incpresmean) * np.sum(obs.fractions[0][1, :]) * 0.5 +\
            reward / absmean * np.sum(obs.fractions[0][0, :]) * 0.5 +\
            (-punishment / incabsmean) * \
            np.sum(obs.fractions[1][0, :]) * 0.5
        print(meanrho)
        print(curr_params['rho'])
