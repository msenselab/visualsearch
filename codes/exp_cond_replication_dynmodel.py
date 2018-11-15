import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from example_model_implementation import run_model

datapath = Path('../data/')
exp1 = pd.read_csv(str(datapath.joinpath('exp1.csv').resolve()), index_col=None)
exp1.rename(columns={'sub': 'subno'}, inplace=True)
exp2 = pd.read_csv(str(datapath.joinpath('exp2.csv').resolve()), index_col=None)
exp2.rename(columns={'sub': 'subno'}, inplace=True)
exp3 = pd.read_csv(str(datapath.joinpath('exp3.csv').resolve()), index_col=None)
exp3.rename(columns={'sub': 'subno'}, inplace=True)
exp5 = pd.read_csv(str(datapath.joinpath('exp5.csv').resolve()), index_col=None)
exp5.rename(columns={'sub': 'subno'}, inplace=True)


# Run the reference model and plot to compare against exp1
size = 250
model_params = {'T': 30,
                'dt': 0.05,
                't_w': 0.5,
                'size': size,
                'lapse': 1e-6,
                'N_values': [8, 12, 16],
                'g_values': np.linspace(1e-4, 1 - 1e-4, size),
                'subject_num': 1,
                'fine_sigma': 0.8,
                'reward': 0.72,
                'punishment': -2.75,
                'fine_model': 'sqrt',
                'reward_scheme': 'asym_reward',
                'num_samples': 100000,
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

axes[1, 0].bar([8, 12, 16], ref_sim_error_rates[:, 0], edgecolor='blue', color='white', width=-1.8,
               linewidth=2)
axes[1, 0].bar([8, 12, 16], ref_sim_error_rates[:, 1], edgecolor='green', color='white', width=1.8,
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

exp1mrt_allsubs = np.array(exp1mrt.groupby(['dyn', 'setsize', 'target']).agg({'rt': 'mean'}))
exp1mrt_allsubs = exp1mrt_allsubs.reshape(3, -1)
exp1sem_allsubs = np.array(exp1mrt.groupby(['dyn', 'setsize', 'target']).agg({'rt': 'sem'}))
exp1sem_allsubs = exp1sem_allsubs.reshape(3, -1)

exp1arr = np.array(exp1.query('dyn == \'Dynamic\''))
sub_error_rates = np.zeros((3, 2, 11))
for subject in range(1, 12):
    for i, N in enumerate([8, 12, 16]):
        for j, C in enumerate(('Absent', 'Present')):
            filt = (exp1arr[:, 0] == C) & (exp1arr[:, 1] == N) & (exp1arr[:, 5] == subject)
            sub_error_rates[i, j, subject - 1] = (np.sum((exp1arr[filt][:, -1] == 0)) /
                                                  len(exp1arr[filt]))

mean_sub_error_rates = np.mean(sub_error_rates, axis=-1)
sem_sub_error_rates = np.std(sub_error_rates, axis=-1) / np.sqrt(sub_error_rates.shape[-1])
axes[1, 1].bar([8, 12, 16], mean_sub_error_rates[:, 0], yerr=sem_sub_error_rates[:, 0],
               edgecolor='blue', color='white', width=-1.8, linewidth=2, capsize=10)
axes[1, 1].bar([8, 12, 16], mean_sub_error_rates[:, 1], yerr=sem_sub_error_rates[:, 0],
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
                'N_values': [8, 12, 16],
                'g_values': np.linspace(1e-4, 1 - 1e-4, size),
                'subject_num': 1,
                'fine_sigma': 0.8,
                'reward': 0.72 * increase_factor,
                'punishment': -2.75,
                'fine_model': 'sqrt',
                'reward_scheme': 'asym_reward',
                'num_samples': 100000,
                }
T = model_params['T']
dt = model_params['dt']
t_values = np.arange(0, T, dt)
N_values = model_params['N_values']

dist_computed_params = run_model(model_params)

absR_sim_mean_rts = np.zeros((3, 2))
sim_error_rates = np.zeros((3, 2))
for i, N in enumerate(N_values):
    currfracs = dist_computed_params[i]['fractions']
    curr_abs_normed = currfracs[0][0, :] / np.sum(currfracs[0][0, :] * dt)
    curr_abs_mean = np.sum(curr_abs_normed * t_values * dt)
    curr_incorr_during_abs = np.sum(currfracs[0][1, :])
    curr_pres_normed = currfracs[1][1, :] / np.sum(currfracs[1][1, :] * dt)
    curr_pres_mean = np.sum(curr_pres_normed * t_values * dt)
    curr_incorr_during_pres = np.sum(currfracs[1][0, :])
    absR_sim_mean_rts[i, 0] = curr_abs_mean
    absR_sim_mean_rts[i, 1] = curr_pres_mean
    sim_error_rates[i, 0] = curr_incorr_during_abs
    sim_error_rates[i, 1] = curr_incorr_during_pres

fig, axes = plt.subplots(2, 4, figsize=(14, 8), sharey='row')

axes[0, 0].plot(N_values, ref_sim_mean_rts[:, 0], lw=2, color='blue', marker='o', ms=10)
axes[0, 0].plot(N_values, ref_sim_mean_rts[:, 1], lw=2, ls='--', color='green', marker='o', ms=10)
axes[0, 0].set_xlim([7, 17])
axes[0, 0].set_xlabel('N stimuli')
axes[0, 0].set_ylabel('Mean reaction time (s)')
axes[0, 0].set_title('Original Simulation')

axes[1, 0].bar([8, 12, 16], ref_sim_error_rates[:, 0], edgecolor='blue', color='white', width=-1.8,
               linewidth=2)
axes[1, 0].bar([8, 12, 16], ref_sim_error_rates[:, 1], edgecolor='green', color='white', width=1.8,
               linewidth=2)
axes[1, 0].set_xticks([8, 12, 16])
axes[1, 0].set_xlabel('N Stimuli')
axes[1, 0].set_ylabel('Mean error rate')

axes[0, 1].plot(N_values, absR_sim_mean_rts[:, 0], lw=2, color='blue', marker='o', ms=10)
axes[0, 1].plot(N_values, absR_sim_mean_rts[:, 1], lw=2, ls='--', color='green', marker='o', ms=10)
axes[0, 1].set_xlim([7, 17])
axes[0, 1].set_xlabel('N stimuli')
axes[0, 1].set_ylabel('Mean reaction time (s)')
axes[0, 1].set_title('Abs-reward Simulation')

axes[1, 1].bar([8, 12, 16], sim_error_rates[:, 0], edgecolor='blue', color='white', width=-1.8,
               linewidth=2)
axes[1, 1].bar([8, 12, 16], sim_error_rates[:, 1], edgecolor='green', color='white', width=1.8,
               linewidth=2)
axes[1, 1].set_xticks([8, 12, 16])
axes[1, 1].set_xlabel('N Stimuli')
axes[1, 1].set_ylabel('Mean error rate')

exp2mrt_abs = exp2.query('correct == 1 & dyn == \'Dynamic\' & reward == \'Absent\'')\
    .groupby(['subno', 'dyn', 'setsize', 'target']).agg({'rt': 'mean'}).reset_index()
sns.catplot(x='setsize', y='rt', hue='target', data=exp2mrt_abs, kind='point', ax=axes[0, 2])
plt.close()
axes[0, 2].set_title('Absent-rewarded subject data')

exp2arr = np.array(exp2.query('dyn == \'Dynamic\' & reward == \'Absent\''))
sub_error_rates = np.zeros((3, 2, 11))
for subject in range(1, 12):
    for i, N in enumerate([8, 12, 16]):
        for j, C in enumerate(('Absent', 'Present')):
            filt = (exp2arr[:, 0] == C) & (exp2arr[:, 1] == N) & (exp2arr[:, 5] == subject)
            sub_error_rates[i, j, subject - 1] = (np.sum((exp2arr[filt][:, -1] == 0)) /
                                                  len(exp2arr[filt]))

mean_sub_error_rates = np.mean(sub_error_rates, axis=-1)
sem_sub_error_rates = np.std(sub_error_rates, axis=-1) / np.sqrt(sub_error_rates.shape[-1])
axes[1, 2].bar([8, 12, 16], mean_sub_error_rates[:, 0], yerr=sem_sub_error_rates[:, 0],
               edgecolor='blue', color='white', width=-1.8, linewidth=2, capsize=10)
axes[1, 2].bar([8, 12, 16], mean_sub_error_rates[:, 1], yerr=sem_sub_error_rates[:, 0],
               edgecolor='green', color='white', width=1.8, linewidth=2, capsize=10)
axes[1, 2].set_xticks([8, 12, 16])
axes[1, 2].set_xlabel('N Stimuli')
axes[1, 2].set_ylabel('Mean error rate')

sns.catplot(x='setsize', y='rt', hue='target', data=exp1mrt, kind='point', ax=axes[0, 3])
plt.close()
axes[0, 3].set_ylim(axes[0, 0].get_ylim())
axes[0, 3].set_title('Reference subject data')

exp1arr = np.array(exp1.query('dyn == \'Dynamic\''))
sub_error_rates = np.zeros((3, 2, 11))
for subject in range(1, 12):
    for i, N in enumerate([8, 12, 16]):
        for j, C in enumerate(('Absent', 'Present')):
            filt = (exp1arr[:, 0] == C) & (exp1arr[:, 1] == N) & (exp1arr[:, 5] == subject)
            sub_error_rates[i, j, subject - 1] = (np.sum((exp1arr[filt][:, -1] == 0)) /
                                                  len(exp1arr[filt]))

mean_sub_error_rates = np.mean(sub_error_rates, axis=-1)
sem_sub_error_rates = np.std(sub_error_rates, axis=-1) / np.sqrt(sub_error_rates.shape[-1])
axes[1, 3].bar([8, 12, 16], mean_sub_error_rates[:, 0], yerr=sem_sub_error_rates[:, 0],
               edgecolor='blue', color='white', width=-1.8, linewidth=2, capsize=10)
axes[1, 3].bar([8, 12, 16], mean_sub_error_rates[:, 1], yerr=sem_sub_error_rates[:, 0],
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
                'N_values': [8, 12, 16],
                'g_values': np.linspace(1e-4, 1 - 1e-4, size),
                'subject_num': 1,
                'fine_sigma': 0.8,
                'reward': 0.72 * (1 / increase_factor),
                'punishment': -2.75 * (1 / increase_factor),
                'fine_model': 'sqrt',
                'reward_scheme': 'asym_reward',
                'num_samples': 100000,
                }
T = model_params['T']
dt = model_params['dt']
t_values = np.arange(0, T, dt)
N_values = model_params['N_values']

dist_computed_params = run_model(model_params)

presR_sim_mean_rts = np.zeros((3, 2))
sim_error_rates = np.zeros((3, 2))
for i, N in enumerate(N_values):
    currfracs = dist_computed_params[i]['fractions']
    curr_abs_normed = currfracs[0][0, :] / np.sum(currfracs[0][0, :] * dt)
    curr_abs_mean = np.sum(curr_abs_normed * t_values * dt)
    curr_incorr_during_abs = np.sum(currfracs[0][1, :])
    curr_pres_normed = currfracs[1][1, :] / np.sum(currfracs[1][1, :] * dt)
    curr_pres_mean = np.sum(curr_pres_normed * t_values * dt)
    curr_incorr_during_pres = np.sum(currfracs[1][0, :])
    presR_sim_mean_rts[i, 0] = curr_abs_mean
    presR_sim_mean_rts[i, 1] = curr_pres_mean
    sim_error_rates[i, 0] = curr_incorr_during_abs
    sim_error_rates[i, 1] = curr_incorr_during_pres

fig, axes = plt.subplots(2, 4, figsize=(14, 8), sharey='row')

axes[0, 0].plot(N_values, ref_sim_mean_rts[:, 0], lw=2, color='blue', marker='o', ms=10)
axes[0, 0].plot(N_values, ref_sim_mean_rts[:, 1], lw=2, ls='--', color='green', marker='o', ms=10)
axes[0, 0].set_xlim([7, 17])
axes[0, 0].set_xlabel('N stimuli')
axes[0, 0].set_ylabel('Mean reaction time (s)')
axes[0, 0].set_title('Original Simulation')

axes[1, 0].bar([8, 12, 16], ref_sim_error_rates[:, 0], edgecolor='blue', color='white', width=-1.8,
               linewidth=2)
axes[1, 0].bar([8, 12, 16], ref_sim_error_rates[:, 1], edgecolor='green', color='white', width=1.8,
               linewidth=2)
axes[1, 0].set_xticks([8, 12, 16])
axes[1, 0].set_xlabel('N Stimuli')
axes[1, 0].set_ylabel('Mean error rate')

axes[0, 1].plot(N_values, presR_sim_mean_rts[:, 0], lw=2, color='blue', marker='o', ms=10)
axes[0, 1].plot(N_values, presR_sim_mean_rts[:, 1], lw=2, ls='--', color='green', marker='o', ms=10)
axes[0, 1].set_xlim([7, 17])
axes[0, 1].set_xlabel('N stimuli')
axes[0, 1].set_ylabel('Mean reaction time (s)')
axes[0, 1].set_title('Pres-reward Simulation')

axes[1, 1].bar([8, 12, 16], sim_error_rates[:, 0], edgecolor='blue', color='white', width=-1.8,
               linewidth=2)
axes[1, 1].bar([8, 12, 16], sim_error_rates[:, 1], edgecolor='green', color='white', width=1.8,
               linewidth=2)
axes[1, 1].set_xticks([8, 12, 16])
axes[1, 1].set_xlabel('N Stimuli')
axes[1, 1].set_ylabel('Mean error rate')

exp2mrt_pres = exp2.query('correct == 1 & dyn == \'Dynamic\' & reward == \'Present\'')\
    .groupby(['subno', 'dyn', 'setsize', 'target']).agg({'rt': 'mean'}).reset_index()

sns.catplot(x='setsize', y='rt', hue='target', data=exp2mrt_pres, kind='point', ax=axes[0, 2])
plt.close()
axes[0, 2].set_title('Present-rewarded subject data')

exp2arr = np.array(exp2.query('dyn == \'Dynamic\' & reward == \'Present\''))
sub_error_rates = np.zeros((3, 2, 11))
for subject in range(1, 12):
    for i, N in enumerate([8, 12, 16]):
        for j, C in enumerate(('Absent', 'Present')):
            filt = (exp2arr[:, 0] == C) & (exp2arr[:, 1] == N) & (exp2arr[:, 5] == subject)
            sub_error_rates[i, j, subject - 1] = (np.sum((exp2arr[filt][:, -1] == 0)) /
                                                  len(exp2arr[filt]))

mean_sub_error_rates = np.mean(sub_error_rates, axis=-1)
sem_sub_error_rates = np.std(sub_error_rates, axis=-1) / np.sqrt(sub_error_rates.shape[-1])
axes[1, 2].bar([8, 12, 16], mean_sub_error_rates[:, 0], yerr=sem_sub_error_rates[:, 0],
               edgecolor='blue', color='white', width=-1.8, linewidth=2, capsize=10)
axes[1, 2].bar([8, 12, 16], mean_sub_error_rates[:, 1], yerr=sem_sub_error_rates[:, 0],
               edgecolor='green', color='white', width=1.8, linewidth=2, capsize=10)
axes[1, 2].set_xticks([8, 12, 16])
axes[1, 2].set_xlabel('N Stimuli')
axes[1, 2].set_ylabel('Mean error rate')

sns.catplot(x='setsize', y='rt', hue='target', data=exp1mrt, kind='point', ax=axes[0, 3])
plt.close()
axes[0, 3].set_ylim(axes[0, 0].get_ylim())
axes[0, 3].set_title('Reference subject data')

exp1arr = np.array(exp1.query('dyn == \'Dynamic\''))
sub_error_rates = np.zeros((3, 2, 11))
for subject in range(1, 12):
    for i, N in enumerate([8, 12, 16]):
        for j, C in enumerate(('Absent', 'Present')):
            filt = (exp1arr[:, 0] == C) & (exp1arr[:, 1] == N) & (exp1arr[:, 5] == subject)
            sub_error_rates[i, j, subject - 1] = (np.sum((exp1arr[filt][:, -1] == 0)) /
                                                  len(exp1arr[filt]))

mean_sub_error_rates = np.mean(sub_error_rates, axis=-1)
sem_sub_error_rates = np.std(sub_error_rates, axis=-1) / np.sqrt(sub_error_rates.shape[-1])
axes[1, 3].bar([8, 12, 16], mean_sub_error_rates[:, 0], yerr=sem_sub_error_rates[:, 0],
               edgecolor='blue', color='white', width=-1.8, linewidth=2, capsize=10)
axes[1, 3].bar([8, 12, 16], mean_sub_error_rates[:, 1], yerr=sem_sub_error_rates[:, 0],
               edgecolor='green', color='white', width=1.8, linewidth=2, capsize=10)
axes[1, 3].set_xticks([8, 12, 16])
axes[1, 3].set_xlabel('N Stimuli')
axes[1, 3].set_ylabel('Mean error rate')
plt.suptitle('Experiment 2: Explicit reward on Present correct')

####
# Summary of differences in mean RT
####
exp1arr_mrts = np.array(exp1mrt.rt).reshape(11, 3, 2)
exp1arr_diffs = exp1arr_mrts[:, :, 0] - exp1arr_mrts[:, :, 1]
exp1arr_diffmean = np.mean(exp1arr_diffs)
exp1arr_diffSEMs = np.std(exp1arr_diffs, axis=0) / np.sqrt(exp1arr_diffs.shape[0])
exp1arr_overallSEM = np.sqrt(np.sum(exp1arr_diffSEMs ** 2))

exp2arr_abs_mrts = np.array(exp2mrt_abs.rt).reshape(12, 3, 2)
exp2arr_abs_diffs = exp2arr_abs_mrts[:, :, 0] - exp2arr_abs_mrts[:, :, 1]
exp2arr_abs_diffmean = np.mean(exp2arr_abs_diffs)
exp2arr_abs_diffSEMs = np.std(exp2arr_abs_diffs, axis=0) / np.sqrt(exp2arr_abs_diffs.shape[0])
exp2arr_abs_overallSEM = np.sqrt(np.sum(exp2arr_abs_diffSEMs ** 2))

exp2arr_pres_mrts = np.array(exp2mrt_pres.rt).reshape(12, 3, 2)
exp2arr_pres_diffs = exp2arr_pres_mrts[:, :, 0] - exp2arr_pres_mrts[:, :, 1]
exp2arr_pres_diffmean = np.mean(exp2arr_pres_diffs)
exp2arr_pres_diffSEMs = np.std(exp2arr_pres_diffs, axis=0) / np.sqrt(exp2arr_pres_diffs.shape[0])
exp2arr_pres_overallSEM = np.sqrt(np.sum(exp2arr_pres_diffSEMs ** 2))

fig, axes = plt.subplots(1, 1, figsize=(5.5, 5.5),
                         sharex=True, sharey=True)  # columns: rewarded C, rows: disp cond C
sim_absR_curvediff_delta = (-ref_sim_mean_rts[:, 0] + absR_sim_mean_rts[:, 0]
                            + ref_sim_mean_rts[:, 1] - absR_sim_mean_rts[:, 1])
sim_absR_mean_delta = np.mean(sim_absR_curvediff_delta)
data_absR_mean_delta = exp2arr_abs_diffmean - exp1arr_diffmean
data_absR_delta_SEM = np.sqrt(exp1arr_overallSEM ** 2 + exp2arr_abs_overallSEM ** 2)
axes.bar(1, sim_absR_mean_delta, width=-0.8,
         edgecolor='blue', color='white', linestyle='--', linewidth=2, label='Sim')
axes.bar(1, data_absR_mean_delta, width=0.8,
         yerr=data_absR_delta_SEM, edgecolor='blue', color='white', linewidth=2,
         label='Data', error_kw=dict(ecolor='blue', lw=2, capsize=10, capthick=2))
axes.set_title('Change in difference between Abs, Pres mean')
axes.legend(loc='upper left')
axes.set_ylabel('Change in RT difference $\Delta(\mu_{abs} - \mu_{pres})$ (s)', size=12)

sim_presR_curvediff_delta = (-ref_sim_mean_rts[:, 0] + presR_sim_mean_rts[:, 0]
                             + ref_sim_mean_rts[:, 1] - presR_sim_mean_rts[:, 1])
sim_presR_mean_delta = np.mean(sim_presR_curvediff_delta)
data_presR_mean_delta = exp2arr_pres_diffmean - exp1arr_diffmean
data_presR_delta_SEM = np.sqrt(exp1arr_overallSEM ** 2 + exp2arr_pres_overallSEM ** 2)
axes.bar(3, sim_presR_mean_delta, width=-0.8,
         edgecolor='green', color='white', linestyle='--', linewidth=2)
axes.bar(3, data_presR_mean_delta, width=0.8,
         yerr=data_presR_delta_SEM, edgecolor='green', color='white', linewidth=2,
         error_kw=dict(ecolor='green', lw=2, capsize=10, capthick=2))
axes.set_xticks([1, 3])
axes.set_xticklabels(['Absent Reward', 'Present Reward'], size=12)
axes.set_title('Change in difference between Abs, Pres mean')
axes.legend(loc='upper left')
axes.set_ylabel('Change in RT difference $\Delta(\mu_{abs} - \mu_{pres})$ (s)', size=12)


##############################
#           EXP5             #
# Increased task difficulty  #
##############################
size = 250
increase_factor = (10 / 9)
model_params = {'T': 30,
                'dt': 0.05,
                't_w': 0.5,
                'size': size,
                'lapse': 1e-6,
                'N_values': [8, 12, 16],
                'g_values': np.linspace(1e-4, 1 - 1e-4, size),
                'subject_num': 1,
                'fine_sigma': 0.8 * increase_factor,
                'reward': 0.72,
                'punishment': -2.75,
                'fine_model': 'sqrt',
                'reward_scheme': 'asym_reward',
                'num_samples': 100000,
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

axes[1, 0].bar([8, 12, 16], ref_sim_error_rates[:, 0], edgecolor='blue', color='white', width=-1.8,
               linewidth=2)
axes[1, 0].bar([8, 12, 16], ref_sim_error_rates[:, 1], edgecolor='green', color='white', width=1.8,
               linewidth=2)
axes[1, 0].set_xticks([8, 12, 16])
axes[1, 0].set_xlabel('N Stimuli')
axes[1, 0].set_ylabel('Mean error rate')

axes[0, 1].plot(N_values, sim_mean_rts[:, 0], lw=2, color='blue', marker='o', ms=10)
axes[0, 1].plot(N_values, sim_mean_rts[:, 1], lw=2, ls='--', color='green', marker='o', ms=10)
axes[0, 1].set_xlim([7, 17])
axes[0, 1].set_xlabel('N stimuli')
axes[0, 1].set_ylabel('Mean reaction time (s)')
axes[0, 1].set_title('Increased noise simulation')

axes[1, 1].bar([8, 12, 16], sim_error_rates[:, 0], edgecolor='blue', color='white', width=-1.8,
               linewidth=2)
axes[1, 1].bar([8, 12, 16], sim_error_rates[:, 1], edgecolor='green', color='white', width=1.8,
               linewidth=2)
axes[1, 1].set_xticks([8, 12, 16])
axes[1, 1].set_xlabel('N Stimuli')
axes[1, 1].set_ylabel('Mean error rate')

exp5mrt = exp5.query('correct == 1 & dyn == \'Dynamic\'')\
    .groupby(['subno', 'dyn', 'setsize', 'target']).agg({'rt': 'mean'}).reset_index()
sns.catplot(x='setsize', y='rt', hue='target', data=exp5mrt, kind='point', ax=axes[0, 2])
plt.close()
axes[0, 2].set_title('Increased-difficulty subject data')

exp5mrt_allsubs = np.array(exp5mrt.groupby(['dyn', 'setsize', 'target']).agg({'rt': 'mean'}))
exp5mrt_allsubs = exp5mrt_allsubs.reshape(3, -1)
exp5sem_allsubs = np.array(exp5mrt.groupby(['dyn', 'setsize', 'target']).agg({'rt': 'sem'}))
exp5sem_allsubs = exp5sem_allsubs.reshape(3, -1)


exp5arr = np.array(exp5.query('dyn == \'Dynamic\''))
sub_error_rates = np.zeros((3, 2, 11))
for subject in range(1, 12):
    for i, N in enumerate([8, 12, 16]):
        for j, C in enumerate(('Absent', 'Present')):
            filt = (exp5arr[:, 1] == C) & (exp5arr[:, 2] == N) & (exp5arr[:, 7] == subject)
            sub_error_rates[i, j, subject - 1] = (np.sum((exp5arr[filt][:, -1] == 0)) /
                                                  len(exp5arr[filt]))

mean_sub_error_rates = np.mean(sub_error_rates, axis=-1)
sem_sub_error_rates = np.std(sub_error_rates, axis=-1) / np.sqrt(sub_error_rates.shape[-1])
axes[1, 2].bar([8, 12, 16], mean_sub_error_rates[:, 0], yerr=sem_sub_error_rates[:, 0],
               edgecolor='blue', color='white', width=-1.8, linewidth=2, capsize=10)
axes[1, 2].bar([8, 12, 16], mean_sub_error_rates[:, 1], yerr=sem_sub_error_rates[:, 0],
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
    for i, N in enumerate([8, 12, 16]):
        for j, C in enumerate(('Absent', 'Present')):
            filt = (exp1arr[:, 0] == C) & (exp1arr[:, 1] == N) & (exp1arr[:, 5] == subject)
            sub_error_rates[i, j, subject - 1] = (np.sum((exp1arr[filt][:, -1] == 0)) /
                                                  len(exp1arr[filt]))

mean_sub_error_rates = np.mean(sub_error_rates, axis=-1)
sem_sub_error_rates = np.std(sub_error_rates, axis=-1) / np.sqrt(sub_error_rates.shape[-1])
axes[1, 3].bar([8, 12, 16], mean_sub_error_rates[:, 0], yerr=sem_sub_error_rates[:, 0],
               edgecolor='blue', color='white', width=-1.8, linewidth=2, capsize=10)
axes[1, 3].bar([8, 12, 16], mean_sub_error_rates[:, 1], yerr=sem_sub_error_rates[:, 0],
               edgecolor='green', color='white', width=1.8, linewidth=2, capsize=10)
axes[1, 3].set_xticks([8, 12, 16])
axes[1, 3].set_xlabel('N Stimuli')
axes[1, 3].set_ylabel('Mean error rate')
plt.suptitle('Experiment 5: Increased task difficulty')

incnoise_diff_SEMs = np.sqrt(exp1sem_allsubs ** 2 + exp5sem_allsubs ** 2)
incnoise_full_SEM = np.sqrt(np.sum(incnoise_diff_SEMs ** 2, axis=0))
fig, axes = plt.subplots(1, 1, figsize=(5.5, 5.5),
                         sharex=True, sharey=True)  # columns: rewarded C, rows: disp cond C
axes.bar(1, np.mean(-ref_sim_mean_rts[:, 0] + sim_mean_rts[:, 0]), width=-.8,
         edgecolor='blue', color='white', linestyle='--', linewidth=2, label='Sim')
axes.bar(1, np.mean(-exp1mrt_allsubs[:, 0] + exp5mrt_allsubs[:, 0]), width=.8,
         yerr=incnoise_full_SEM[0], edgecolor='blue', color='white', linewidth=2,
         label='Data', error_kw=dict(ecolor='blue', lw=2, capsize=15, capthick=2))
axes.bar(3, np.mean(-ref_sim_mean_rts[:, 1] + sim_mean_rts[:, 1]), width=-.8,
         edgecolor='green', color='white', linestyle='--', linewidth=2)
axes.bar(3, np.mean(-exp1mrt_allsubs[:, 1] + exp5mrt_allsubs[:, 1]), width=.8,
         yerr=incnoise_full_SEM[1], edgecolor='green', color='white', linewidth=2,
         error_kw=dict(ecolor='green', lw=2, capsize=15, capthick=2))
axes.set_title('Mean RT change\n' + r'$\mu_{\sigma_{large}} - \mu_{\sigma_{small}}$', size=15)
axes.set_xticks([1, 3])
axes.set_xticklabels(['Target Absent', 'Target Present'], size=15)
axes.set_xlabel('')
axes.set_ylabel('$\Delta \mu_{RT}$', size=15)
plt.tight_layout()
