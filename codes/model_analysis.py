import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, writers
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
            ax.set_zlabel('log(likelihood)', size=20)
            ax.set_title("Subject {} {} optimization, {} fine model, {} reward scheme\n".format(
                         self.subject_num, self.opt_type, self.fine_model, self.reward_scheme) +
                         "Optimum at {:.3f} {:.3f}".format(x[np.argmin(z)], y[np.argmin(z)]))
        else:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
            x = np.exp(self.tested_params)
            y = self.likelihoods_returned
            ax.scatter(x, y, c=y, cmap='viridis', s=12, edgecolor='none')
            ax.set_xlabel(r'$\sigma_{fine}$', size=20)
            ax.set_ylabel('log(likelihood)', size=20)
            ax.set_title("Subject {} {} optimization, {} fine model, {} reward scheme".format(
                         self.subject_num, self.opt_type, self.fine_model, self.reward_scheme),
                         size=24)
        return fig, ax

    def plot_opt_fits(self):
        opt_index = np.argmin(self.likelihoods_returned)
        optparams = self.tested_params[opt_index]

        if self.opt_type == 'sig':
            self.model_params['fine_sigma'] = np.exp(optparams[0])
            optstring = r'$\sigma_{fine}$ =' + str(np.exp(optparams[0]))[:5]
        elif self.opt_type == 'sig_reward':
            self.model_params['fine_sigma'] = np.exp(optparams[0])
            self.model_params['reward'] = np.exp(optparams[1])
            optstring = r'$\sigma_{fine}$ =' + str(np.exp(optparams[0]))[:5] +\
                ' reward =' + str(np.exp(optparams[1]))[:5]
        elif self.opt_type == 'sig_punish':
            self.model_params['fine_sigma'] = np.exp(optparams[0])
            self.model_params['punishment'] = np.exp(optparams[1])
            optstring = r'$\sigma_{fine}$ =' + str(np.exp(optparams[0]))[:5] +\
                ' punishment =' + str(np.exp(optparams[1]))[:5]
        finegr = FineGrained(**self.model_params)
        coarse_stats = finegr.coarse_stats
        subject_data = DataLikelihoods(**self.model_params)

        fig, axes = plt.subplots(len(self.model_params['N_values']), 3,
                                 figsize=(22, 14))
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
            inc_abs_rts = N_data.query('resp == 2 & target == \'Present\'').rt.values
            inc_pres_rts = N_data.query('resp == 1 & target == \'Absent\'').rt.values

            t_values = np.arange(0, curr_params['T'], curr_params['dt'])
            sim_abs_rts = obs.fractions[0][0, :] / np.sum(obs.fractions[0][0, :])
            sim_inc_abs_rts = obs.fractions[0][1, :] / np.sum(obs.fractions[0][1, :])
            sim_pres_rts = obs.fractions[1][1, :] / np.sum(obs.fractions[1][1, :])
            sim_inc_pres_rts = obs.fractions[1][0, :] / np.sum(obs.fractions[1][0, :])

            sns.kdeplot(corr_abs_rts, bw=0.1, shade=True, c='blue', ax=axes[i, 0],
                        label="Data correct absent")
            sns.kdeplot(corr_pres_rts, bw=0.1, shade=True, c='red', ax=axes[i, 0],
                        label="Data correct present")
            sns.kdeplot(sim_abs_rts, bw=0.1, shade=True, c='purple', ax=axes[i, 0],
                        label="Sim correct absent")
            axes[i, 0].fill_between(t_values, sim_abs_rts, color='purple',
                                    label='Sim correct absent')
            axes[i, 0].fill_between(t_values, sim_pres_rts, color='orange',
                                    label='Sim correct present')

            axes[i, 0].legend(loc="upper right")
            axes[i, 0].set_title(str(N) + ' correct')
            axes[i, 0].set_xlim([0, self.model_params['T']])

            sns.kdeplot(inc_abs_rts, bw=0.1, shade=True, c='blue', ax=axes[i, 1],
                        label="Data incorrect absent")
            sns.kdeplot(inc_pres_rts, bw=0.1, shade=True, c='red', ax=axes[i, 1],
                        label="Data incorrect present")
            axes[i, 1].fill_between(t_values, sim_inc_abs_rts, color='purple',
                                    label='Sim incorrect absent')
            axes[i, 1].fill_between(t_values, sim_inc_pres_rts, color='orange',
                                    label='Sim incorrect present')

            axes[i, 1].legend(loc="upper right")
            axes[i, 1].set_title(str(N) + ' incorrect')
            axes[i, 1].set_xlim([0, self.model_params['T']])

            totresp_data = len(corr_abs_rts) + len(corr_pres_rts) +\
                len(inc_abs_rts) + len(inc_pres_rts)

            bars = axes[i, 2].bar([0, 1, 2, 3, 5, 6, 7, 8],
                                  [len(corr_abs_rts) / totresp_data,
                                   np.sum(obs.fractions[0][0, :]),
                                   len(inc_abs_rts) / totresp_data,
                                   np.sum(obs.fractions[0][1, :]),
                                   len(corr_pres_rts) / totresp_data,
                                   np.sum(obs.fractions[1][1, :]),
                                   len(inc_pres_rts) / totresp_data,
                                   np.sum(obs.fractions[1][0, :])],
                                  width=1)
            colors = ['blue', 'purple', 'blue', 'purple', 'red', 'orange', 'red', 'orange']
            for k in range(len(colors)):
                bars[k].set_color(colors[k])
            axes[i, 2].set_xticks([1, 3, 6, 8])
            axes[i, 2].set_xticklabels([])
            axes[i, 2].set_title('Proportion Responses')
        for j in range(2):
            axes[-1, j].set_xlabel('RT (s)')
        axes[2, 2].set_xticklabels(['Correct Absent', 'Incorrect Absent',
                                    'Correct Present', 'Incorrect Present'], rotation=40)
        fig.suptitle("Subject {} {} optimization, {} fine model, {} reward scheme\n {}".format(
                     self.subject_num, self.opt_type, self.fine_model,
                     self.reward_scheme, optstring), size=24)
        return fig, axes

    def save_anim_likelihoods(self, savepath, numframes=420):
        """ Assumes savepath is a Path object from pathlib, not a string """
        if self.opt_type not in ('sig_reward', 'sig_punish'):
            raise ValueError('Only two-parameter optimization data can be shown in 3D')
        ffmpeg_writer = writers['ffmpeg']
        curr_writer = ffmpeg_writer(fps=60, bitrate=5200)

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
        ax.set_zlabel('log(likelihood)', size=20)
        ax.set_title("Subject {} {} optimization, {} fine model, {} reward scheme\n".format(
                     self.subject_num, self.opt_type, self.fine_model, self.reward_scheme) +
                     "Optimum at {:.3f} {:.3f}".format(x[np.argmin(z)], y[np.argmin(z)]))

        def anim_update(i):
            ax.view_init(azim=(i / numframes) * 360, elev=30)
            plt.draw()
            return

        anim = FuncAnimation(fig, anim_update, frames=numframes)
        anim.save(str(savepath.expanduser()), writer=curr_writer)
        return


if __name__ == '__main__':
    from pathlib import Path
    import os
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
        if not os.path.exists(str(savepath.expanduser())):
            os.mkdir(str(savepath.expanduser()))

        for subject in subjects:
            filename = './subject_{}_{}_{}_{}_modelfit.p'.format(subject, opt_type,
                                                                 reward_scheme, fine_model)
            try:
                sub_fit = np.load(datapath.joinpath(filename).expanduser())
            except FileNotFoundError:
                continue
            if 'opt_type' not in sub_fit:
                sub_fit['opt_type'] = opt_type

            curranalysis = OptAnalysis(**sub_fit)
            # First plot tested likelihoods
            if sub_fit['opt_type'] == 'sig':
                fig, ax = curranalysis.plot_likelihoods()
                figurepath = savepath.joinpath('subject_{}_{}_{}_{}_likelihoods_tested.png'.format(
                                               subject, opt_type, reward_scheme, fine_model))
                plt.savefig(str(figurepath.expanduser()), DPI=500)
                plt.close()
            else:
                moviepath = savepath.joinpath('subject_{}_{}_{}_{}_likelihoods_tested.mp4'.format(
                                              subject, opt_type, reward_scheme, fine_model))
                curranalysis.save_anim_likelihoods(moviepath)
                plt.close()

            # Now plot distributions of best fit
            distfigurepath = savepath.joinpath('subject_{}_{}_{}_{}_bestfit_dist.png'.format(
                                               subject, opt_type, reward_scheme, fine_model))
            fig, ax = curranalysis.plot_opt_fits()
            plt.savefig(str(distfigurepath.expanduser()), DPI=500)
            plt.close()
