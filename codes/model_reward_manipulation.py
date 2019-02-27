import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from bellman_utilities import BellmanUtil
from observers import ObserverSim
from copy import deepcopy

savepath = Path("~/Documents/fit_data/")  # Where to save figures
loadpath = Path("~/Documents/fit_data/single_N/")
savepath = str(savepath.expanduser())

subject_num = 1
manipulation_amount = 0.15
manip_vals = np.array([1 - manipulation_amount, 1, 1 + manipulation_amount])

size = 600
model_params = {'T': 10,
                'dt': 0.05,
                't_w': 0.5,
                't_delay': 0.2,
                't_max': 5.,
                'size': size,
                'lapse': 1e-6,
                'mu': np.array((0, 1)),
                'N_values': (8, 12, 16),
                'g_values': np.linspace(1e-4, 1 - 1e-4, size),
                'subject_num': subject_num,
                'reward_scheme': 'asym_reward',
                'experiment': 'null'}

dt = model_params['dt']
T = model_params['T']
t_delay = model_params['t_delay']
t_max = model_params['t_max']
maxind = int(t_max / dt) - 1
t_values = np.arange(0, model_params['T'], model_params['dt'])

filename = str(loadpath.expanduser()) + '/subject_{}_single_N_12_modelfit.p'.format(subject_num)
testdata = np.load(filename)
print('fit returned min of', np.amin(testdata['likelihoods_returned']))
log_parameters = testdata['tested_params'][np.argmin(testdata['likelihoods_returned'])]


sim_mrts = np.zeros((3, 3, 2, 2))

sim_prop_incorrect = np.zeros((3, 3, 2))
fig, ax = plt.subplots(3, 3, sharex=True)
for i, rewardfactor in enumerate(manip_vals):
    for j, N in enumerate(model_params['N_values']):
        curr_params = deepcopy(model_params)
        curr_params['N'] = N
        sigma = np.array((np.exp(log_parameters[0]), np.exp(log_parameters[1])))
        reward = np.exp(log_parameters[2]) * rewardfactor
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

        presmean = np.sum(obs.fractions[1][1, :maxind] * (t_values[:maxind] + t_delay))\
            / np.sum(obs.fractions[1][1, :maxind]) + t_delay
        incpresmean = np.sum(obs.fractions[0][1, :maxind] * (t_values[:maxind] + t_delay))\
            / np.sum(obs.fractions[0][1, :maxind]) + t_delay
        absmean = np.sum(obs.fractions[0][0, :maxind] * (t_values[:maxind] + t_delay))\
            / np.sum(obs.fractions[0][0, :maxind]) + t_delay
        incabsmean = np.sum(obs.fractions[1][0, :maxind] * (t_values[:maxind] + t_delay))\
            / np.sum(obs.fractions[1][0, :maxind]) + t_delay

        sim_mrts[i, j, :] = np.array([[absmean, incabsmean],
                                      [incpresmean, presmean]])
        sim_prop_incorrect[i, j, :] = np.array([np.sum(obs.fractions[0][1, :maxind]),
                                                np.sum(obs.fractions[1][1, :maxind])])
        ax[i, j].fill_between(t_values[:maxind],
                              obs.fractions[0][0, :maxind] / np.sum(obs.fractions[0][0, :maxind] * dt),
                              color='purple', alpha=0.5, label=r'$\hat{C} = 0 | C = 0')
        ax[i, j].fill_between(t_values[:maxind],
                              obs.fractions[1][1, :maxind] / np.sum(obs.fractions[1][1, :maxind] * dt),
                              color='orange', alpha=0.5, label=r'$\hat{C} = 1 | C = 1')
        ax[i, j].set_title('N = {}, reward = {:.2f}'.format(N, reward))
        ymin, ymax = ax[i, j].get_ylim()
        ax[i, j].vlines(absmean, ymin, ymax, color='purple', lw=3)
        ax[i, j].vlines(presmean, ymin, ymax, color='orange', lw=3)
    ax[i, -1].set_xlabel('RT (s)')
    ax[i, 0].set_ylabel('Reward manipulation = {:.2f}'.format(rewardfactor))

exp1 = pd.read_csv('../data/exp1.csv')
exp2 = pd.read_csv('../data/exp2.csv')

exp1 = exp1[(exp1['dyn'] == 'Dynamic') & (exp1['sub'] != 666) & (exp1['setsize'] == 12)]
exp2 = exp2[(exp2['dyn'] == 'Dynamic') & (exp2['sub'] != 666) & (exp2['setsize'] == 12)]
exp1['reward'] = 'None'
newdf = exp1.append(exp1, sort=False)

g = sns.catplot(x='reward', y='rt', hue='target', data=newdf, kind='point',
                order=['Absent', 'None', 'Present'])
g.set_xticklabels(size=18)
g.set_yticklabels(size=18)
g.set_ylabels('RT (s)', size=20)
g.set_xlabels('Reward Condition', size=20)

ax = plt.gca()
ax.plot([0, 1, 2], sim_mrts[::-1, 1, 1, 1], label='Sim Present', c='lightgreen', lw=2)
ax.plot([0, 1, 2], sim_mrts[::-1, 1, 0, 0], label='Sim Absent', c='purple', lw=2)
