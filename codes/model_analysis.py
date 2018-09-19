import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from fine_grain_model import FineGrained
from bellman_utilities import BellmanUtil
from observers import ObserverSim
from data_and_likelihood import DataLikelihoods


class OptAnalysis:
    def __init__(self, subject_num, T, dt, t_w, size, lapse, N_values, g_values, fine_model,
                 reward_scheme, opt_type, tested_params, likelihoods_returned, **kwargs):
        self.model_params = {'T': T,
                             'dt': dt,
                             't_w': t_w,
                             'size': size,
                             'lapse': lapse,
                             'N_values': N_values,
                             'g_values': g_values,
                             'subject_num': subject_num,
                             'fine_model': fine_model,
                             'reward_scheme': reward_scheme,
                             }
        self.tested_params = tested_params
        self.likelihoods_returned = likelihoods_returned
        self.opt_type = opt_type
        self.subject_num = subject_num
        self.fine_model = fine_model
        self.reward_scheme = reward_scheme

        if opt_type == 'sig':
            self.model_params['reward'] = 1
            self.model_params['punishment'] = 0
        elif opt_type == 'sig_reward':
            self.model_params['punishment'] = 0
        elif opt_type == 'sig_punish':
            self.model_params['reward'] = 1
        else:
            raise ValueError('opt_type not in supported set')

    def plot_likelihoods(self):
        if self.opt_type in ('sig_reward', 'sig_punish'):
            fig = plt.figure(figsize=(10, 10))
            ax = Axes3D(fig)
            x, y = np.split(self.tested_params, 2, axis=1)
            x = np.exp(x.reshape(-1))  # parameters are stored as log values. Get back orig via exp
            y = np.exp(y.reshape(-1))
            z = self.likelihoods_returned
            ax.scatter(x, y, z, c=z, cmap='viridis', s=20, lw=0.2)
            ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')
            if self.opt_type == 'sig_reward':
                ax.set_xlabel(r'$\sigma_{fine}$', size=20)
                ax.set_ylabel('Reward', size=20)
            elif self.opt_type == 'sig_punish':
                ax.set_xlabel(r'$\sigma_{fine}$', size=20)
                ax.set_ylabel('Punishment', size=20)
            ax.set_zlabel('log(likelihood)')
            ax.set_title("Subject {} {} optimization, {} fine model, {} reward scheme".format(
                         self.subject_num, self.opt_type, self.fine_model, self.reward_scheme))
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            x = np.exp(self.tested_params)
            y = self.likelihoods_returned
            ax.plot(x, y, c=y, cmap='viridis', size=20)
            ax.set_xlabel(r'$\sigma_{fine}$')
            ax.set_ylabel('log(likelihood)')
            ax.set_title("Subject {} {} optimization, {} fine model, {} reward scheme".format(
                         self.subject_num, self.opt_type, self.fine_model, self.reward_scheme))
        return fig, ax

    def plot_opt_fits(self):
        opt_index = np.argmin(self.likelihoods_returned)
        optparams = self.tested_params[opt_index]

        if self.opt_type == 'sig':
            self.model_params['fine_sigma'] = np.exp(optparams[0])
        elif self.opt_type == 'sig_reward':
            self.model_params['fine_sigma'] = np.exp(optparams[0])
            self.model_params['reward'] = np.exp(optparams[1])
        elif self.opt_type == 'sig_punish':
            self.model_params['fine_sigma'] = np.exp(optparams[0])
            self.model_params['punishment'] = np.exp(optparams[1])
        finegr = FineGrained(**self.model_params)
        coarse_stats = finegr.coarse_stats
        subject_data = DataLikelihoods(**self.model_params)

        fig, axes = plt.subplots(len(self.model_params['N_values']), 1,
                                 sharex=True, figsize=(10, 10))
        for i in range(len(self.model_params['N_values'])):
            curr_params = deepcopy(self.model_params)
            N = curr_params['N_values'][i]
            curr_params['mu'] = coarse_stats[i, :, 0]
            curr_params['sigma'] = coarse_stats[i, :, 1]
            curr_params['N'] = N

            bellutil = BellmanUtil(**curr_params)
            curr_params['decisions'] = bellutil.decisions
            obs = ObserverSim(**curr_params)

            N_data = subject_data.sub_data.query('setsize == {}'.format(N))
            # Only plotting the correct responses, but the data should fit all incl. incorr
            corr_abs_rts = N_data.query('resp == 2 & target == \'Absent\'').rt.values
            corr_pres_rts = N_data.query('resp == 1 & target == \'Present\'').rt.values
            sim_abs_rts = obs.rt_abs[obs.rt_abs[:, 0] == 0, 1]
            sim_pres_rts = obs.rt_pres[obs.rt_pres[:, 0] == 1, 1]

            sns.kdeplot(corr_abs_rts, bw=0.1, shade=True, c='blue', ax=axes[i],
                        label="Data correct absent")
            sns.kdeplot(corr_pres_rts, bw=0.1, shade=True, c='red', ax=axes[i],
                        label="Data correct present")
            sns.kdeplot(sim_abs_rts, bw=0.1, shade=True, c='purple', ax=axes[i],
                        label="Sim correct absent")
            sns.kdeplot(sim_pres_rts, bw=0.1, shade=True, c='orange', ax=axes[i],
                        label="Data correct present")
            axes[i].legend(loc="upper right")
            axes[i].set_title(str(N))

        axes[-1].set_xlabel('RT (s)')
        axes[-1].set_xlim([0, self.model_params['T']])
        fig.suptitle("Subject {} {} optimization, {} fine model, {} reward scheme".format(
                     self.subject_num, self.opt_type, self.fine_model, self.reward_scheme))
        return fig, axes


if __name__ == '__main__':
    from pathlib import Path
    datapath = Path('~/Documents/fit_data/')

    subjects = list(range(1, 12))
    models = [('sig', 'sym', 'const'),
              ('sig_reward', 'asym_reward', 'const'),
              ('sig_punish', 'epsilon_punish', 'const'),
              ('sig', 'sym', 'sqrt'),
              ('sig_reward', 'asym_reward', 'sqrt'),
              ('sig_punish', 'epsilon_punish', 'sqrt')]

    for model in models:
        opt_type, reward_scheme, fine_model = model
        savepath = datapath.joinpath('./{}_{}_{}'.format(opt_type, reward_scheme, fine_model))
        for subject in subjects:
            filename = 'subject_{}_{}_{}_{}_modelfit.p'.format(subject, opt_type,
                                                               reward_scheme, fine_model)
