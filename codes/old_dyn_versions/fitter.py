from OptDDM import OptDDM
from gauss_opt import bayesian_optimisation
import numpy as np
import itertools as it


class Fitter:
    def __init__(self, data, sub_num, model_type, opt_iter, T, t_w, dt, size,
                 lapse, n_pre_samples=15):
        '''
        Class to perform fitting of subject data on a given model type with set parameters
        '''
        self.model_type = model_type
        self.num_samples = opt_iter
        self.sub_num = sub_num
        self.sub_data = data.query('subno == {} & dyn == \'Dynamic\''.format(sub_num))
        self.N_array = np.array([8, 12, 16])

        def subject_likelihood(params):
            ddm = OptDDM(params, self.model_type, T, t_w, dt, size, lapse)
            return ddm.get_data_likelihood(self.sub_data)

        if model_type[0] == 'sig':
            bnds = np.array(((-1.7, 1.),))  # [n_variables, 2] shaped array with bounds
            x_opt = bayesian_optimisation(n_iters=self.num_samples, sample_loss=subject_likelihood,
                                          bounds=bnds, n_pre_samples=5)
        if model_type[0] == 'sig_reward':
            bnds = np.array(((-1.7, 1.), (-1., 0.5)))  # [n_variables, 2] shaped array with bounds
            x_opt = bayesian_optimisation(n_iters=self.num_samples, sample_loss=subject_likelihood,
                                          bounds=bnds, n_pre_samples=15)

        if model_type[0] == 'sig_punish':
            bnds = np.array(((-1.7, 1.), (-5., -0.5)))  # [n_variables, 2] shaped array with bounds
            x_opt = bayesian_optimisation(n_iters=self.num_samples, sample_loss=subject_likelihood,
                                          bounds=bnds, n_pre_samples=5)

        xp, yp = x_opt

        self.xp = xp
        self.yp = yp
        self.opt_likelihood = np.amin(yp)
        self.opt_params = xp[np.argmin(yp)]
        self.opt_DDM = OptDDM(xp, self.model_type, T, t_w, dt, size, lapse)
        # Pull out each of the log(sigma) that the optimizer tested and put them in an array together
        # with the associated log(likelihood). datarr is (N x 2) where N is the number of optimize samps

    # def plot_gp(self):
    #     # Plot test points and likelihoods
    #     fig = plt.figure()
    #     ax = Axes3D(fig)
    #     if self.model_type[0] == 'sig':
    #         ax.scatter(self.xp[:, 0], self.yp, s=100)
    #     if self.model_type[0] == 'sig_reward':
    #         ax.scatter(self.xp[:, 0], self.xp[:, 1], self.yp, s=100)
    #         ax.set_xlabel('$log(\sigma)$')
    #         ax.set_ylabel('$log(reward)$')
    #         ax.set_zlabel('$log(likelihood)$')
    #     if self.model_type[0] == 'sig_punish':
    #         ax.scatter(self.xp[:, 0], self.xp[:, 1], self.opt_paramsyp, s=100)
    #         ax.set_xlabel('$log(\sigma)$')
    #         ax.set_ylabel('$log(punishment)$')
    #         ax.set_zlabel('$log(likelihood)$')
    #
    # def plot_fits(self):
    #     fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8.5))
    #     plt.figtext(0.67, 0.67, "-log likelihood = {}".format(np.round(self.opt_likelihood, 3)))
    #
    #     sigma = np.exp(self.opt_params[0])
    #     if self.model_type[0] == 'sig':
    #         reward = 1
    #         punishment = 0
    #         fig.suptitle('Parameters: sigma = {}'.format(np.round(sigma, 3))
    #                      + ', Reward Scheme: {},'.format(self.model_type[1])
    #                      + ' Fine Model: {}'.format(self.model_type[2]))
    #     elif self.model_type[0] == 'sig_reward':
    #         reward = np.exp(self.opt_params[1])
    #         punishment = 0
    #         fig.suptitle('Parameters: sigma = {}'.format(np.round(sigma, 3))
    #                      + '; reward = {}'.format(np.round(reward, 3))
    #                      + ', Reward Scheme: {},'.format(self.model_type[1])
    #                      + ' Fine Model: {}'.format(self.model_type[2]))
    #     elif self.model_type[0] == 'sig_punish':
    #         punishment = np.exp(self.opt_params[1])
    #         reward = 1
    #         fig.suptitle('Parameters: sigma = {}'.format(np.round(sigma, 3))
    #                      + '; punishment = {}'.format(np.round(punishment))
    #                      + ', Reward Scheme: {},'.format(self.model_type[1])
    #                      + ' Fine Model: {}'.format(self.model_type[2]))
    #
    #     data_array = [self.sub_data.query('setsize == 8'), self.sub_data.query('setsize == 12'),
    #                   self.sub_data.query('setsize == 16')]
    #
    #     for i in range(self.N_array.size):
    #         N = self.N_array[i]
    #         abs_rt = self.opt_DDM.get_rt(N, 0)
    #         pres_rt = self.opt_DDM.get_rt(N, 1)
    #
    #         currdata = data_array[i]
    #         #pres_rts_0 = currdata.query('resp == 2 & target == \'Present\'').rt.values
    #         pres_rts_1 = currdata.query('resp == 1 & target == \'Present\'').rt.values
    #
    #         abs_rts_0 = currdata.query('resp == 2 & target == \'Absent\'').rt.values
    #         #abs_rts_1 = currdata.query('resp == 1 & target == \'Absent\'').rt.values
    #
    #         ax = axes[i]
    #         ax.set_title('N = {}'.format(N))
    #         sns.kdeplot(abs_rts_0, bw=0.1, shade=True, label='Data: con. = 0, resp. = 0',
    #                     color='darkblue', ax=ax)
    #         sns.kdeplot(pres_rts_1, bw=0.1, shade=True, label='Data: con. = 1, resp. = 1',
    #                     color='red', ax=ax)
    #
    #         self.opt_DDM.get_kde_dist(abs_rt, pres_rt, plot=True, ax=ax)
    #
    #         ax.set_ylabel('Density estimate')
    #         ax.legend()
    #
    #         if i == 2:
    #             ax.set_xlabel('RT (s)')
    #             ax.set_xlim([0, 11])

    def get_rt_dict(self):
        all_rt = {}
        display_con = ('pres', 'abs')
        for n_con in it.product(self.N_array, display_con):
            all_rt[n_con] = []

        for i in range(self.N_array.size):
            N = self.N_array[i]
            abs_rt = self.opt_DDM.get_rt(N, 0)
            pres_rt = self.opt_DDM.get_rt(N, 1)
            all_rt[N, 'abs'].append(abs_rt)
            all_rt[N, 'pres'].append(pres_rt)
