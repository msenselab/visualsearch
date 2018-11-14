import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from fine_grain_model import FineGrained
from bellman_utilities import BellmanUtil
from observers import ObserverSim
from data_and_likelihood import DataLikelihoods
from example_model_implementation import run_model

datapath = Path('../data/')
exp1 = pd.read_csv(str(datapath.joinpath('exp1.csv').resolve()), index_col=None)
exp1.rename(columns={'sub': 'subno'}, inplace=True)
exp2 = pd.read_csv(str(datapath.joinpath('exp2.csv').resolve()), index_col=None)
exp2.rename(columns={'sub': 'subno'}, inplace=True)
exp3 = pd.read_csv(str(datapath.joinpath('exp3.csv').resolve()), index_col=None)
exp3.rename(columns={'sub': 'subno'}, inplace=True)


# Run the reference model and plot to compare against exp1
size = 250
model_params = {'T': 30,
                'dt': 0.05,
                't_w': 0.5,
                'size': size,
                'lapse': 1e-6,
                'N_values': (8, 12, 16),
                'g_values': np.linspace(1e-4, 1 - 1e-4, size),
                'subject_num': 1,
                'fine_sigma': 0.8,
                'reward': 0.567,
                'punishment': -2.29,
                'fine_model': 'sqrt',
                'reward_scheme': 'asym_reward',
                }
T = model_params['T']
dt = model_params['dt']
t_values = np.arange(0, T, dt)
N_values = model_params['N_values']

dist_computed_params = run_model(model_params)

ref_sim_mean_rts = np.zeros((3, 2))
ref_sim_error_rates = np.zeros((3, 2))
for i, N in enumerate(N_values):
    currfracs = dist_computed_params[i]['fractions']
    curr_abs_normed = currfracs[0][0, :] / np.sum(currfracs[0][0, :] * dt)
    curr_abs_mean = np.sum(curr_abs_normed * t_values * dt)
    curr_incorr_during_abs = np.sum(currfracs[0][1, :])
    curr_pres_normed = currfracs[1][1, :] / np.sum(currfracs[1][1, :] * dt)
    curr_pres_mean = np.sum(curr_pres_normed * t_values * dt)
    curr_incorr_during_pres = np.sum(currfracs[1][0, :])
    ref_sim_mean_rts[i, 0] = curr_abs_mean
    ref_sim_mean_rts[i, 1] = curr_pres_mean
    ref_sim_error_rates[i, 0] = curr_incorr_during_abs
    ref_sim_error_rates[i, 1] = curr_incorr_during_pres

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey='row')
axes[0, 0].plot(N_values, ref_sim_mean_rts[:, 0], lw=2, color='blue', marker='o', ms=10)
axes[0, 0].plot(N_values, ref_sim_mean_rts[:, 1], lw=2, ls='--', color='green', marker='o', ms=10)
axes[0, 0].set_xlim([7, 17])
axes[0, 0].set_xlabel('N stimuli')
axes[0, 0].set_ylabel('Mean reaction time (s)')
axes[0, 0].set_title('Simulation')

axes[1, 0].bar((8, 12, 16), ref_sim_error_rates[:, 0], edgecolor='blue', color='white', width=-1.8,
               linewidth=2)
axes[1, 0].bar((8, 12, 16), ref_sim_error_rates[:, 1], edgecolor='green', color='white', width=1.8,
               linewidth=2)
axes[1, 0].set_xticks([8, 12, 16])
axes[1, 0].set_xlabel('N Stimuli')
axes[1, 0].set_ylabel('Mean error rate')

exp1mrt = exp1.query('correct == 1 & dyn == \'Dynamic\'')\
    .groupby(['subno', 'dyn', 'setsize', 'target']).agg({'rt': 'mean'}).reset_index()
sns.catplot(x='setsize', y='rt', hue='target', data=exp1mrt, kind='point', ax=axes[0, 1])
plt.close()
axes[0, 1].set_ylim(axes[0, 0].get_ylim())
axes[0, 1].set_title('Aggregated subject data')

exp1arr = np.array(exp1.query('dyn == \'Dynamic\''))
sub_error_rates = np.zeros((3, 2, 11))
for subject in range(1, 12):
    for i, N in enumerate((8, 12, 16)):
        for j, C in enumerate(('Absent', 'Present')):
            filt = (exp1arr[:, 0] == C) & (exp1arr[:, 1] == N) & (exp1arr[:, 5] == subject)
            sub_error_rates[i, j, subject - 1] = (np.sum((exp1arr[filt][:, -1] == 0)) /
                                                  len(exp1arr[filt]))

mean_sub_error_rates = np.mean(sub_error_rates, axis=-1)
sem_sub_error_rates = np.std(sub_error_rates, axis=-1) / np.sqrt(sub_error_rates.shape[-1])
axes[1, 1].bar((8, 12, 16), mean_sub_error_rates[:, 0], yerr=sem_sub_error_rates[:, 0],
               edgecolor='blue', color='white', width=-1.8, linewidth=2, capsize=10)
axes[1, 1].bar((8, 12, 16), mean_sub_error_rates[:, 1], yerr=sem_sub_error_rates[:, 0],
               edgecolor='green', color='white', width=1.8, linewidth=2, capsize=10)
axes[1, 1].set_xticks([8, 12, 16])
axes[1, 1].set_xlabel('N Stimuli')
axes[1, 1].set_ylabel('Mean error rate')

plt.suptitle('Experiment 1: No manipulations, No explicit reward')


#############################
#           EXP2            #
# Increased abs reward case #
#############################
size = 250
increase_factor = (4 / 3)
model_params = {'T': 30,
                'dt': 0.05,
                't_w': 0.5,
                'size': size,
                'lapse': 1e-6,
                'N_values': (8, 12, 16),
                'g_values': np.linspace(1e-4, 1 - 1e-4, size),
                'subject_num': 1,
                'fine_sigma': 0.8,
                'reward': 0.567 * increase_factor,
                'punishment': -2.29,
                'fine_model': 'sqrt',
                'reward_scheme': 'asym_reward',
                }
T = model_params['T']
dt = model_params['dt']
t_values = np.arange(0, T, dt)
N_values = model_params['N_values']

dist_computed_params = run_model(model_params)

sim_mean_rts = np.zeros((3, 2))
sim_error_rates = np.zeros((3, 2))
for i, N in enumerate(N_values):
    currfracs = dist_computed_params[i]['fractions']
    curr_abs_normed = currfracs[0][0, :] / np.sum(currfracs[0][0, :] * dt)
    curr_abs_mean = np.sum(curr_abs_normed * t_values * dt)
    curr_incorr_during_abs = np.sum(currfracs[0][1, :])
    curr_pres_normed = currfracs[1][1, :] / np.sum(currfracs[1][1, :] * dt)
    curr_pres_mean = np.sum(curr_pres_normed * t_values * dt)
    curr_incorr_during_pres = np.sum(currfracs[1][0, :])
    sim_mean_rts[i, 0] = curr_abs_mean
    sim_mean_rts[i, 1] = curr_pres_mean
    sim_error_rates[i, 0] = curr_incorr_during_abs
    sim_error_rates[i, 1] = curr_incorr_during_pres

fig, axes = plt.subplots(2, 4, figsize=(14, 8), sharey='row')

axes[0, 0].plot(N_values, ref_sim_mean_rts[:, 0], lw=2, color='blue', marker='o', ms=10)
axes[0, 0].plot(N_values, ref_sim_mean_rts[:, 1], lw=2, ls='--', color='green', marker='o', ms=10)
axes[0, 0].set_xlim([7, 17])
axes[0, 0].set_xlabel('N stimuli')
axes[0, 0].set_ylabel('Mean reaction time (s)')
axes[0, 0].set_title('Original Simulation')

axes[1, 0].bar((8, 12, 16), ref_sim_error_rates[:, 0], edgecolor='blue', color='white', width=-1.8,
               linewidth=2)
axes[1, 0].bar((8, 12, 16), ref_sim_error_rates[:, 1], edgecolor='green', color='white', width=1.8,
               linewidth=2)
axes[1, 0].set_xticks([8, 12, 16])
axes[1, 0].set_xlabel('N Stimuli')
axes[1, 0].set_ylabel('Mean error rate')

axes[0, 1].plot(N_values, sim_mean_rts[:, 0], lw=2, color='blue', marker='o', ms=10)
axes[0, 1].plot(N_values, sim_mean_rts[:, 1], lw=2, ls='--', color='green', marker='o', ms=10)
axes[0, 1].set_xlim([7, 17])
axes[0, 1].set_xlabel('N stimuli')
axes[0, 1].set_ylabel('Mean reaction time (s)')
axes[0, 1].set_title('Abs-reward Simulation')

axes[1, 1].bar((8, 12, 16), sim_error_rates[:, 0], edgecolor='blue', color='white', width=-1.8,
               linewidth=2)
axes[1, 1].bar((8, 12, 16), sim_error_rates[:, 1], edgecolor='green', color='white', width=1.8,
               linewidth=2)
axes[1, 1].set_xticks([8, 12, 16])
axes[1, 1].set_xlabel('N Stimuli')
axes[1, 1].set_ylabel('Mean error rate')

exp2mrt = exp2.query('correct == 1 & dyn == \'Dynamic\' & reward == \'Absent\'')\
    .groupby(['subno', 'dyn', 'setsize', 'target']).agg({'rt': 'mean'}).reset_index()
sns.catplot(x='setsize', y='rt', hue='target', data=exp2mrt, kind='point', ax=axes[0, 2])
plt.close()
axes[0, 2].set_title('Absent-rewarded subject data')

exp2arr = np.array(exp2.query('dyn == \'Dynamic\' & reward == \'Absent\''))
sub_error_rates = np.zeros((3, 2, 11))
for subject in range(1, 12):
    for i, N in enumerate((8, 12, 16)):
        for j, C in enumerate(('Absent', 'Present')):
            filt = (exp2arr[:, 0] == C) & (exp2arr[:, 1] == N) & (exp2arr[:, 5] == subject)
            sub_error_rates[i, j, subject - 1] = (np.sum((exp2arr[filt][:, -1] == 0)) /
                                                  len(exp2arr[filt]))

mean_sub_error_rates = np.mean(sub_error_rates, axis=-1)
sem_sub_error_rates = np.std(sub_error_rates, axis=-1) / np.sqrt(sub_error_rates.shape[-1])
axes[1, 2].bar((8, 12, 16), mean_sub_error_rates[:, 0], yerr=sem_sub_error_rates[:, 0],
               edgecolor='blue', color='white', width=-1.8, linewidth=2, capsize=10)
axes[1, 2].bar((8, 12, 16), mean_sub_error_rates[:, 1], yerr=sem_sub_error_rates[:, 0],
               edgecolor='green', color='white', width=1.8, linewidth=2, capsize=10)
axes[1, 2].set_xticks([8, 12, 16])
axes[1, 2].set_xlabel('N Stimuli')
axes[1, 2].set_ylabel('Mean error rate')

exp1mrt = exp1.query('correct == 1 & dyn == \'Dynamic\'')\
    .groupby(['subno', 'dyn', 'setsize', 'target']).agg({'rt': 'mean'}).reset_index()
sns.catplot(x='setsize', y='rt', hue='target', data=exp1mrt, kind='point', ax=axes[0, 3])
plt.close()
axes[0, 3].set_ylim(axes[0, 0].get_ylim())
axes[0, 3].set_title('Reference subject data')

exp1arr = np.array(exp1.query('dyn == \'Dynamic\''))
sub_error_rates = np.zeros((3, 2, 11))
for subject in range(1, 12):
    for i, N in enumerate((8, 12, 16)):
        for j, C in enumerate(('Absent', 'Present')):
            filt = (exp1arr[:, 0] == C) & (exp1arr[:, 1] == N) & (exp1arr[:, 5] == subject)
            sub_error_rates[i, j, subject - 1] = (np.sum((exp1arr[filt][:, -1] == 0)) /
                                                  len(exp1arr[filt]))

mean_sub_error_rates = np.mean(sub_error_rates, axis=-1)
sem_sub_error_rates = np.std(sub_error_rates, axis=-1) / np.sqrt(sub_error_rates.shape[-1])
axes[1, 3].bar((8, 12, 16), mean_sub_error_rates[:, 0], yerr=sem_sub_error_rates[:, 0],
               edgecolor='blue', color='white', width=-1.8, linewidth=2, capsize=10)
axes[1, 3].bar((8, 12, 16), mean_sub_error_rates[:, 1], yerr=sem_sub_error_rates[:, 0],
               edgecolor='green', color='white', width=1.8, linewidth=2, capsize=10)
axes[1, 3].set_xticks([8, 12, 16])
axes[1, 3].set_xlabel('N Stimuli')
axes[1, 3].set_ylabel('Mean error rate')
plt.suptitle('Experiment 2: Explicit reward on Absent correct')

##############################
#           EXP2             #
# Increased pres reward case #
##############################
size = 250
increase_factor = (4 / 3)
model_params = {'T': 30,
                'dt': 0.05,
                't_w': 0.5,
                'size': size,
                'lapse': 1e-6,
                'N_values': (8, 12, 16),
                'g_values': np.linspace(1e-4, 1 - 1e-4, size),
                'subject_num': 1,
                'fine_sigma': 0.8,
                'reward': 0.567 * (1 / increase_factor),
                'punishment': -2.29 * (1 / increase_factor),
                'fine_model': 'sqrt',
                'reward_scheme': 'asym_reward',
                }
T = model_params['T']
dt = model_params['dt']
t_values = np.arange(0, T, dt)
N_values = model_params['N_values']

dist_computed_params = run_model(model_params)

sim_mean_rts = np.zeros((3, 2))
sim_error_rates = np.zeros((3, 2))
for i, N in enumerate(N_values):
    currfracs = dist_computed_params[i]['fractions']
    curr_abs_normed = currfracs[0][0, :] / np.sum(currfracs[0][0, :] * dt)
    curr_abs_mean = np.sum(curr_abs_normed * t_values * dt)
    curr_incorr_during_abs = np.sum(currfracs[0][1, :])
    curr_pres_normed = currfracs[1][1, :] / np.sum(currfracs[1][1, :] * dt)
    curr_pres_mean = np.sum(curr_pres_normed * t_values * dt)
    curr_incorr_during_pres = np.sum(currfracs[1][0, :])
    sim_mean_rts[i, 0] = curr_abs_mean
    sim_mean_rts[i, 1] = curr_pres_mean
    sim_error_rates[i, 0] = curr_incorr_during_abs
    sim_error_rates[i, 1] = curr_incorr_during_pres

fig, axes = plt.subplots(2, 4, figsize=(14, 8), sharey='row')

axes[0, 0].plot(N_values, ref_sim_mean_rts[:, 0], lw=2, color='blue', marker='o', ms=10)
axes[0, 0].plot(N_values, ref_sim_mean_rts[:, 1], lw=2, ls='--', color='green', marker='o', ms=10)
axes[0, 0].set_xlim([7, 17])
axes[0, 0].set_xlabel('N stimuli')
axes[0, 0].set_ylabel('Mean reaction time (s)')
axes[0, 0].set_title('Original Simulation')

axes[1, 0].bar((8, 12, 16), ref_sim_error_rates[:, 0], edgecolor='blue', color='white', width=-1.8,
               linewidth=2)
axes[1, 0].bar((8, 12, 16), ref_sim_error_rates[:, 1], edgecolor='green', color='white', width=1.8,
               linewidth=2)
axes[1, 0].set_xticks([8, 12, 16])
axes[1, 0].set_xlabel('N Stimuli')
axes[1, 0].set_ylabel('Mean error rate')

axes[0, 1].plot(N_values, sim_mean_rts[:, 0], lw=2, color='blue', marker='o', ms=10)
axes[0, 1].plot(N_values, sim_mean_rts[:, 1], lw=2, ls='--', color='green', marker='o', ms=10)
axes[0, 1].set_xlim([7, 17])
axes[0, 1].set_xlabel('N stimuli')
axes[0, 1].set_ylabel('Mean reaction time (s)')
axes[0, 1].set_title('Pres-reward Simulation')

axes[1, 1].bar((8, 12, 16), sim_error_rates[:, 0], edgecolor='blue', color='white', width=-1.8,
               linewidth=2)
axes[1, 1].bar((8, 12, 16), sim_error_rates[:, 1], edgecolor='green', color='white', width=1.8,
               linewidth=2)
axes[1, 1].set_xticks([8, 12, 16])
axes[1, 1].set_xlabel('N Stimuli')
axes[1, 1].set_ylabel('Mean error rate')

exp2mrt = exp2.query('correct == 1 & dyn == \'Dynamic\' & reward == \'Present\'')\
    .groupby(['subno', 'dyn', 'setsize', 'target']).agg({'rt': 'mean'}).reset_index()
sns.catplot(x='setsize', y='rt', hue='target', data=exp2mrt, kind='point', ax=axes[0, 2])
plt.close()
axes[0, 2].set_title('Present-rewarded subject data')

exp2arr = np.array(exp2.query('dyn == \'Dynamic\' & reward == \'Present\''))
sub_error_rates = np.zeros((3, 2, 11))
for subject in range(1, 12):
    for i, N in enumerate((8, 12, 16)):
        for j, C in enumerate(('Absent', 'Present')):
            filt = (exp2arr[:, 0] == C) & (exp2arr[:, 1] == N) & (exp2arr[:, 5] == subject)
            sub_error_rates[i, j, subject - 1] = (np.sum((exp2arr[filt][:, -1] == 0)) /
                                                  len(exp2arr[filt]))

mean_sub_error_rates = np.mean(sub_error_rates, axis=-1)
sem_sub_error_rates = np.std(sub_error_rates, axis=-1) / np.sqrt(sub_error_rates.shape[-1])
axes[1, 2].bar((8, 12, 16), mean_sub_error_rates[:, 0], yerr=sem_sub_error_rates[:, 0],
               edgecolor='blue', color='white', width=-1.8, linewidth=2, capsize=10)
axes[1, 2].bar((8, 12, 16), mean_sub_error_rates[:, 1], yerr=sem_sub_error_rates[:, 0],
               edgecolor='green', color='white', width=1.8, linewidth=2, capsize=10)
axes[1, 2].set_xticks([8, 12, 16])
axes[1, 2].set_xlabel('N Stimuli')
axes[1, 2].set_ylabel('Mean error rate')

exp1mrt = exp1.query('correct == 1 & dyn == \'Dynamic\'')\
    .groupby(['subno', 'dyn', 'setsize', 'target']).agg({'rt': 'mean'}).reset_index()
sns.catplot(x='setsize', y='rt', hue='target', data=exp1mrt, kind='point', ax=axes[0, 3])
plt.close()
axes[0, 3].set_ylim(axes[0, 0].get_ylim())
axes[0, 3].set_title('Reference subject data')

exp1arr = np.array(exp1.query('dyn == \'Dynamic\''))
sub_error_rates = np.zeros((3, 2, 11))
for subject in range(1, 12):
    for i, N in enumerate((8, 12, 16)):
        for j, C in enumerate(('Absent', 'Present')):
            filt = (exp1arr[:, 0] == C) & (exp1arr[:, 1] == N) & (exp1arr[:, 5] == subject)
            sub_error_rates[i, j, subject - 1] = (np.sum((exp1arr[filt][:, -1] == 0)) /
                                                  len(exp1arr[filt]))

mean_sub_error_rates = np.mean(sub_error_rates, axis=-1)
sem_sub_error_rates = np.std(sub_error_rates, axis=-1) / np.sqrt(sub_error_rates.shape[-1])
axes[1, 3].bar((8, 12, 16), mean_sub_error_rates[:, 0], yerr=sem_sub_error_rates[:, 0],
               edgecolor='blue', color='white', width=-1.8, linewidth=2, capsize=10)
axes[1, 3].bar((8, 12, 16), mean_sub_error_rates[:, 1], yerr=sem_sub_error_rates[:, 0],
               edgecolor='green', color='white', width=1.8, linewidth=2, capsize=10)
axes[1, 3].set_xticks([8, 12, 16])
axes[1, 3].set_xlabel('N Stimuli')
axes[1, 3].set_ylabel('Mean error rate')
plt.suptitle('Experiment 2: Explicit reward on Present correct')


##############################
#           EXP2             #
# Increased pres reward case #
##############################
size = 250
increase_factor = (4 / 3)
model_params = {'T': 30,
                'dt': 0.05,
                't_w': 0.5,
                'size': size,
                'lapse': 1e-6,
                'N_values': (8, 12, 16),
                'g_values': np.linspace(1e-4, 1 - 1e-4, size),
                'subject_num': 1,
                'fine_sigma': 0.8 * increase_factor,
                'reward': 0.567,
                'punishment': -2.29,
                'fine_model': 'sqrt',
                'reward_scheme': 'asym_reward',
                }
T = model_params['T']
dt = model_params['dt']
t_values = np.arange(0, T, dt)
N_values = model_params['N_values']

dist_computed_params = run_model(model_params)

sim_mean_rts = np.zeros((3, 2))
sim_error_rates = np.zeros((3, 2))
for i, N in enumerate(N_values):
    currfracs = dist_computed_params[i]['fractions']
    curr_abs_normed = currfracs[0][0, :] / np.sum(currfracs[0][0, :] * dt)
    curr_abs_mean = np.sum(curr_abs_normed * t_values * dt)
    curr_incorr_during_abs = np.sum(currfracs[0][1, :])
    curr_pres_normed = currfracs[1][1, :] / np.sum(currfracs[1][1, :] * dt)
    curr_pres_mean = np.sum(curr_pres_normed * t_values * dt)
    curr_incorr_during_pres = np.sum(currfracs[1][0, :])
    sim_mean_rts[i, 0] = curr_abs_mean
    sim_mean_rts[i, 1] = curr_pres_mean
    sim_error_rates[i, 0] = curr_incorr_during_abs
    sim_error_rates[i, 1] = curr_incorr_during_pres

fig, axes = plt.subplots(2, 4, figsize=(14, 8), sharey='row')

axes[0, 0].plot(N_values, ref_sim_mean_rts[:, 0], lw=2, color='blue', marker='o', ms=10)
axes[0, 0].plot(N_values, ref_sim_mean_rts[:, 1], lw=2, ls='--', color='green', marker='o', ms=10)
axes[0, 0].set_xlim([7, 17])
axes[0, 0].set_xlabel('N stimuli')
axes[0, 0].set_ylabel('Mean reaction time (s)')
axes[0, 0].set_title('Original Simulation')

axes[1, 0].bar((8, 12, 16), ref_sim_error_rates[:, 0], edgecolor='blue', color='white', width=-1.8,
               linewidth=2)
axes[1, 0].bar((8, 12, 16), ref_sim_error_rates[:, 1], edgecolor='green', color='white', width=1.8,
               linewidth=2)
axes[1, 0].set_xticks([8, 12, 16])
axes[1, 0].set_xlabel('N Stimuli')
axes[1, 0].set_ylabel('Mean error rate')

axes[0, 1].plot(N_values, sim_mean_rts[:, 0], lw=2, color='blue', marker='o', ms=10)
axes[0, 1].plot(N_values, sim_mean_rts[:, 1], lw=2, ls='--', color='green', marker='o', ms=10)
axes[0, 1].set_xlim([7, 17])
axes[0, 1].set_xlabel('N stimuli')
axes[0, 1].set_ylabel('Mean reaction time (s)')
axes[0, 1].set_title('Pres-reward Simulation')

axes[1, 1].bar((8, 12, 16), sim_error_rates[:, 0], edgecolor='blue', color='white', width=-1.8,
               linewidth=2)
axes[1, 1].bar((8, 12, 16), sim_error_rates[:, 1], edgecolor='green', color='white', width=1.8,
               linewidth=2)
axes[1, 1].set_xticks([8, 12, 16])
axes[1, 1].set_xlabel('N Stimuli')
axes[1, 1].set_ylabel('Mean error rate')

exp3mrt = exp3.query('correct == 1 & dyn == \'Dynamic\'')\
    .groupby(['subno', 'dyn', 'setsize', 'target']).agg({'rt': 'mean'}).reset_index()
sns.catplot(x='setsize', y='rt', hue='target', data=exp3mrt, kind='point', ax=axes[0, 2])
plt.close()
axes[0, 2].set_title('Present-rewarded subject data')

exp3arr = np.array(exp3.query('dyn == \'Dynamic\''))
sub_error_rates = np.zeros((3, 2, 11))
for subject in range(1, 12):
    for i, N in enumerate((8, 12, 16)):
        for j, C in enumerate(('Absent', 'Present')):
            filt = (exp3arr[:, 0] == C) & (exp3arr[:, 1] == N) & (exp3arr[:, 5] == subject)
            sub_error_rates[i, j, subject - 1] = (np.sum((exp3arr[filt][:, -1] == 0)) /
                                                  len(exp3arr[filt]))

mean_sub_error_rates = np.mean(sub_error_rates, axis=-1)
sem_sub_error_rates = np.std(sub_error_rates, axis=-1) / np.sqrt(sub_error_rates.shape[-1])
axes[1, 2].bar((8, 12, 16), mean_sub_error_rates[:, 0], yerr=sem_sub_error_rates[:, 0],
               edgecolor='blue', color='white', width=-1.8, linewidth=2, capsize=10)
axes[1, 2].bar((8, 12, 16), mean_sub_error_rates[:, 1], yerr=sem_sub_error_rates[:, 0],
               edgecolor='green', color='white', width=1.8, linewidth=2, capsize=10)
axes[1, 2].set_xticks([8, 12, 16])
axes[1, 2].set_xlabel('N Stimuli')
axes[1, 2].set_ylabel('Mean error rate')

exp1mrt = exp1.query('correct == 1 & dyn == \'Dynamic\'')\
    .groupby(['subno', 'dyn', 'setsize', 'target']).agg({'rt': 'mean'}).reset_index()
sns.catplot(x='setsize', y='rt', hue='target', data=exp1mrt, kind='point', ax=axes[0, 3])
plt.close()
axes[0, 3].set_ylim(axes[0, 0].get_ylim())
axes[0, 3].set_title('Reference subject data')

exp1arr = np.array(exp1.query('dyn == \'Dynamic\''))
sub_error_rates = np.zeros((3, 2, 11))
for subject in range(1, 12):
    for i, N in enumerate((8, 12, 16)):
        for j, C in enumerate(('Absent', 'Present')):
            filt = (exp1arr[:, 0] == C) & (exp1arr[:, 1] == N) & (exp1arr[:, 5] == subject)
            sub_error_rates[i, j, subject - 1] = (np.sum((exp1arr[filt][:, -1] == 0)) /
                                                  len(exp1arr[filt]))

mean_sub_error_rates = np.mean(sub_error_rates, axis=-1)
sem_sub_error_rates = np.std(sub_error_rates, axis=-1) / np.sqrt(sub_error_rates.shape[-1])
axes[1, 3].bar((8, 12, 16), mean_sub_error_rates[:, 0], yerr=sem_sub_error_rates[:, 0],
               edgecolor='blue', color='white', width=-1.8, linewidth=2, capsize=10)
axes[1, 3].bar((8, 12, 16), mean_sub_error_rates[:, 1], yerr=sem_sub_error_rates[:, 0],
               edgecolor='green', color='white', width=1.8, linewidth=2, capsize=10)
axes[1, 3].set_xticks([8, 12, 16])
axes[1, 3].set_xlabel('N Stimuli')
axes[1, 3].set_ylabel('Mean error rate')
plt.suptitle('Experiment 2: Explicit reward on Present correct')
