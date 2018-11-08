import numpy as np
import os
import pickle
from copy import deepcopy
import multiprocessing as mulpro
import matplotlib.pyplot as plt
import seaborn as sns
from fine_grain_model import FineGrained
from bellman_utilities import BellmanUtil
from observers import ObserverSim


def likelihood_inner_loop(curr_params):
    bellutil = BellmanUtil(**curr_params)
    curr_params['decisions'] = bellutil.decisions

    obs = ObserverSim(**curr_params)
    curr_params['fractions'] = obs.fractions
    return curr_params


fine_sigma = 0.669
reward_vals = np.linspace(0.815, 1., 50)

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
                'num_samples': 50000,
                'opt_type': 'sig_reward'}

prescolorvals = sns.color_palette('Blues_d', reward_vals.shape[0])
abscolorvals = sns.color_palette('Greens_d', reward_vals.shape[0])

pres_cmap = sns.choose_colorbrewer_palette('Blues_d', as_cmap=True)
abs_cmap = sns.choose_colorbrewer_palette('Greens_d', as_cmap=True)

maxlen = int(model_params['T'] / model_params['dt'])


def rewardeval(reward):
    curr_params = deepcopy(model_params)
    curr_params['reward'] = reward
    print(np.where(reward_vals == reward)[0][0])

    finegr = FineGrained(**curr_params)
    coarse_stats = finegr.coarse_stats

    N_blocked_model_params = []
    for j in range(len(model_params['N_values'])):
        N_params = deepcopy(curr_params)
        N_params['mu'] = coarse_stats[j, :, 0]
        N_params['sigma'] = coarse_stats[j, :, 1]
        N_params['N'] = curr_params['N_values'][j]
        N_blocked_model_params.append(N_params)

    dist_computed_params = list(map(likelihood_inner_loop, N_blocked_model_params))

    outarr = np.zeros((3, 2, 3, maxlen))
    boundsarr = np.zeros((3, model_params['size'], maxlen))
    for j in range(len(model_params['N_values'])):
        outarr[j, 0, :, :] = dist_computed_params[j]['fractions'][0]
        outarr[j, 1, :, :] = dist_computed_params[j]['fractions'][1]
        boundsarr[j, :, :] = dist_computed_params[j]['decisions']

    return outarr, boundsarr


if __name__ == '__main__':
    allfracs = np.zeros((reward_vals.shape[0],  # Store all fractions for each reward value
                         len(model_params['N_values']),  # Store fractions for each N
                         2,  # Store fractions across state C in (Abs, Pres)
                         3,  # Store fractions across choice identity (abs, pres, wait)
                         maxlen))  # Store across all time steps
    allbounds = np.zeros((reward_vals.shape[0],
                          len(model_params['N_values']),
                          model_params['size'],
                          maxlen))
    max_processes = 18
    for firstproc in range(0, reward_vals.shape[0], max_processes):
        indices = np.arange(firstproc, firstproc + max_processes)
        indices = indices[indices < reward_vals.shape[0]]
        pool = mulpro.Pool(processes=max_processes)
        currevals = pool.map(rewardeval, reward_vals[indices])
        pool.close()
        pool.join()
        for i, (eval, bounds) in enumerate(currevals):
            allfracs[indices[i], :] = eval
            allbounds[indices[i], :] = bounds

    fpath = os.path.expanduser('~/Documents/allfracs_{}_finesig_widereward.p'.format(fine_sigma))
    fw = open(fpath, 'wb')
    outdict = {'allfracs': allfracs, 'allbounds': allbounds, 'fine_sigma': fine_sigma,
               'reward_vals': reward_vals, 'model_params': model_params}
    pickle.dump(outdict, fw)
    fw.close()

    T = model_params['T']
    dt = model_params['dt']
    t_values = np.arange(0, T, dt)
    t_centers = t_values + (dt / 2)

    upperbounds = np.zeros((reward_vals.shape[0], 3))
    lowerbounds = np.zeros((reward_vals.shape[0], 3))
    g_values = model_params['g_values']
    for i in range(reward_vals.shape[0]):
        for j in range(len(model_params['N_values'])):
            currdecs = allbounds[i, j, :, -1]
            upperbounds[i, j] = g_values[np.where(currdecs == 2)[0][0]]
            lowerbounds[i, j] = g_values[np.amax(np.where(currdecs == 1)[0]) + 1]

    N_values = model_params['N_values']
    plt.figure()
    for i in range(reward_vals.shape[0]):
        plt.plot(N_values, upperbounds[i, :], lw=2, color=prescolorvals[i])
        plt.plot(N_values, lowerbounds[i, :], lw=2, color=abscolorvals[i])
    plt.xlabel('N stimuli', size=22)
    plt.xticks(N_values)
    plt.ylabel(r'$g_t = P(C = 1 | x_{1 \dots t})$', size=22)

    fig, axes = plt.subplots(2, 10, figsize=(28, 8))
    interval = int(reward_vals.shape[0] / 10)
    midpoint = int(reward_vals.shape[0] / 2)

    for i, j in enumerate(range(0, reward_vals.shape[0], interval)):
        axes[0, i].plot(t_values, allfracs[j, 0, 0, 0, :] / np.sum(allfracs[j, 0, 0, 0, :] * dt),
                        color=abscolorvals[0], lw=2)
        axes[0, i].plot(t_values, allfracs[j, 1, 0, 0, :] / np.sum(allfracs[j, 1, 0, 0, :] * dt),
                        color=abscolorvals[midpoint], lw=2)
        axes[0, i].plot(t_values, allfracs[j, 2, 0, 0, :] / np.sum(allfracs[j, 2, 0, 0, :] * dt),
                        color=abscolorvals[-1], lw=2)
        axes[0, i].set_xlim([0, 5])
        axes[0, i].set_title('reward = {:.3f}'.format(reward_vals[j]))

        axes[1, i].plot(t_values, allfracs[j, 0, 1, 1, :] / np.sum(allfracs[j, 0, 1, 1, :] * dt),
                        color=abscolorvals[0], lw=2)
        axes[1, i].plot(t_values, allfracs[j, 1, 1, 1, :] / np.sum(allfracs[j, 1, 1, 1, :] * dt),
                        color=abscolorvals[midpoint], lw=2)
        axes[1, i].plot(t_values, allfracs[j, 2, 1, 1, :] / np.sum(allfracs[j, 2, 1, 1, :] * dt),
                        color=abscolorvals[-1], lw=2)
        axes[1, i].set_xlim([0, 5])
        # axes[1, i].set_ylim([0, 1])
        axes[1, i].set_xlabel('Reaction time (s)')

    plt.tight_layout()
    # norm = mpl.colors.Normalize(vmin=reward_vals[0], vmax=reward_vals[-1])
    # cb1 = mpl.colorbar.ColorbarBase(axes[0, 1], cmap=abs_cmap, norm=norm, orientation='vertical')
    # axes[0, 1].set_title('Absent colormap')
    # cb2 = mpl.colorbar.ColorbarBase(axes[1, 1], cmap=pres_cmap, norm=norm, orientation='vertical')
    # axes[1, 1].set_title('Present colormap')

    sim_rt_means = np.zeros(allfracs.shape[:-1])
    norm_fracs = np.zeros_like(allfracs)
    for i in range(reward_vals.shape[0]):
        for j in range(3):
            for k in range(2):
                for l in range(3):
                    currarr = allfracs[i, j, k, l, :]
                    norm_fracs[i, j, k, l, :] = currarr / (np.sum(currarr) * dt)
    sim_rt_means = np.sum(norm_fracs * t_centers * dt, axis=-1)
