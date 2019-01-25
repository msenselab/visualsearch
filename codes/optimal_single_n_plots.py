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
N_index = 0
N_value = 8


subject_num = 4

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

t_values = np.arange(0, model_params['T'], model_params['dt'])

testdata = np.load('/home/berk/Documents/fit_data/single_N/subject_{}_single_N_12_modelfit.p'.format(subject_num))
np.amin(testdata['likelihoods_returned'])
log_parameters = testdata['tested_params'][np.argmin(testdata['likelihoods_returned'])]
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
presmean = np.sum(obs.fractions[1][1, :] * t_values) / np.sum(obs.fractions[1][1, :])
absmean = np.sum(obs.fractions[0][0, :] * t_values) / np.sum(obs.fractions[0][0, :])


likelihood_data = DataLikelihoods(**curr_params)
likelihood_data.increment_likelihood(**curr_params)
print(likelihood_data.likelihood)
N_data = likelihood_data.sub_data.query('setsize == 12')
subj_rts = np.zeros((2, 3), dtype=object)
subj_rts[0, 0] = N_data.query('resp == 2 & target == \'Absent\'').rt.values
subj_rts[0, 1] = N_data.query('resp == 1 & target == \'Absent\'').rt.values
subabsmean = np.mean(subj_rts[0, 0])
abs_timeouts = len(N_data.query('resp == -1 & target == \'Absent\'').rt.values)
abs_timeouts = np.array([abs_timeouts])

subj_rts[1, 0] = N_data.query('resp == 2 & target == \'Present\'').rt.values
subj_rts[1, 1] = N_data.query('resp == 1 & target == \'Present\'').rt.values
subpresmean = np.mean(subj_rts[1, 1])
pres_timeouts = len(N_data.query('resp == -1 & target == \'Present\'').rt.values)
pres_timeouts = np.array([pres_timeouts])
plt.fill_between(t_values + 0.2, obs.fractions[0][0, :] / np.sum(obs.fractions[0][0, :]) / dt, color='purple', alpha=0.5)
plt.fill_between(t_values + 0.2, obs.fractions[1][1, :] / np.sum(obs.fractions[1][1, :]) / dt, color='orange', alpha=0.5)
sns.kdeplot(subj_rts[0, 0], bw=0.1, alpha=0.5, shade=True, color='blue')
sns.kdeplot(subj_rts[1, 1], bw=0.1, alpha=0.5, shade=True, color='red')
ax = plt.gca()
ymin, ymax = ax.get_ylim()
plt.vlines(presmean, ymin, ymax, colors='orange', lw=2)
plt.vlines(absmean, ymin, ymax, colors='purple', lw=2)
plt.vlines(np.mean(subj_rts[0, 0]), ymin, ymax, color='blue', lw=2)
plt.vlines(np.mean(subj_rts[1, 1]), ymin, ymax, color='red', lw=2)

plt.xlabel('RT (s)', size=18)
ax.set_xlim([0, 6])
ax.set_ylim([ymin, ymax])
plt.title('Optimal fit Sub {}\n'.format(subject_num) + 'sig_abs = {:.2f}, sig_pres = {:.2f}, reward = {:.2f}\n punishment = {:.2f}, alpha = {:.2f}'.format(*np.exp(log_parameters)) , size=18)
plt.tight_layout()
plt.savefig('/home/berk/Documents/single_N_subject_{}_N_12_1000_iter_optfit.png'.format(subject_num), DPI=500)

meanrho = presmean * np.sum(obs.fractions[1][1, :]) * 1 +\
          np.sum(obs.fractions[1][0, :] * t_values) * punishment +\
          absmean * np.sum(obs.fractions[0][0, :]) * reward +\
          np.sum(obs.fractions[0][1, :] * t_values) * punishment
