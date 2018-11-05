import numpy as np
from copy import deepcopy
import multiprocessing as mulpro
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from fine_grain_model import FineGrained
from bellman_utilities import BellmanUtil
from observers import ObserverSim
from data_and_likelihood import DataLikelihoods


def likelihood_inner_loop(curr_params):
    bellutil = BellmanUtil(**curr_params)
    curr_params['decisions'] = bellutil.decisions

    obs = ObserverSim(**curr_params)
    curr_params['fractions'] = obs.fractions
    return curr_params


fine_sigma = 0.622
reward_vals = np.linspace(0.9, 1.1, 1000)

size = 100
model_params = {'T': 10,
                'dt': 0.05,
                't_w': 0.5,
                'size': size,
                'lapse': 1e-6,
                'N_values': (8, 12, 16),
                'g_values': np.linspace(1e-4, 1 - 1e-4, size),
                'subject_num': 1,
                'fine_sigma': fine_sigma,
                'punishment': 0,
                'fine_model': 'const',
                'reward_scheme': 'asym_reward',
                'opt_type': 'sig_reward'}

colorvals = sns.cubehelix_palette(reward_vals.shape[0], start=2, rot=0, dark=0, light=.85,
                                  reverse=True)
cmap = sns.cubehelix_palette(as_cmap=True, start=2, rot=0, dark=0, light=.85,
                             reverse=True)
maxlen = int(model_params['T'] / model_params['dt'])

allfracs = np.zeros((reward_vals.shape[0],  # Store all fractions for each reward value
                     len(model_params['N_values']),  # Store fractions for each N
                     2,  # Store fractions across state C in (Abs, Pres)
                     3,  # Store fractions across choice identity (abs, pres, wait)
                     maxlen))  # Store across all time steps
for i, reward in enumerate(reward_vals):
    curr_params = deepcopy(model_params)
    curr_params['reward'] = reward
    print(i)

    finegr = FineGrained(**curr_params)
    coarse_stats = finegr.coarse_stats

    N_blocked_model_params = []
    for j in range(len(model_params['N_values'])):
        N_params = deepcopy(curr_params)
        N_params['mu'] = coarse_stats[j, :, 0]
        N_params['sigma'] = coarse_stats[j, :, 1]
        N_params['N'] = curr_params['N_values'][j]
        N_blocked_model_params.append(N_params)

    pool = mulpro.Pool(processes=3)
    dist_computed_params = pool.map(likelihood_inner_loop, N_blocked_model_params)
    pool.close()
    pool.join()

    for j in range(len(model_params['N_values'])):
        allfracs[i, j, 0, :, :] = dist_computed_params[j]['fractions'][0]
        allfracs[i, j, 1, :, :] = dist_computed_params[j]['fractions'][1]

T = model_params['T']
dt = model_params['dt']
t_values = np.arange(0, T, dt)
fig, axes = plt.subplots(2, 2)
for i in range(reward_vals.shape[0]):
    axes[0, 0].plot(t_values, allfracs[i, 0, 0, 0, :] / np.sum(allfracs[i, 0, 0, 0, :] * dt),
                    color=colorvals[i], lw=3, alpha=0.25)
    axes[1, 0].plot(t_values, allfracs[i, 0, 1, 1, :] / np.sum(allfracs[i, 0, 1, 1, :] * dt),
                    color=colorvals[i], lw=3, alpha=0.25)
norm = mpl.colors.Normalize(vmin=reward_vals[0], vmax=reward_vals[-1])
cb1 = mpl.colorbar.ColorbarBase(axes[0, 1], cmap=cmap, norm=norm, orientation='vertical')
axes[1, 1].remove()
