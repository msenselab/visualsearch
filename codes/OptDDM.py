import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm, uniform
import multiprocessing as mulpro
import numpy as np
from finegr_model import FineGrained
from bellman_utilities import BellmanUtil


class OptDDM:
    def __init__(self, log_params, model_type, inits):
        '''model type is formated as tuple with first argument denoting parameters to fits;
            options are:
                sig; fits just a fine grained sigma
                sig_reward; fits fine grained sigma and reward per subject, punishment set to 0
                sig_punish; fit fine grain sigma and punishment, reward set to 0
        the second argument denoting the reward scheme used in backwards induction
            options are:
                sym; symetric reward matrix in reward and punishment
                epsilon_punish; reward for correct fixed at 1 for both pres/abs correct response
                asym_reward; reward for correct absent fixed at 1
        and the third argument denoting the model used to bootstrap coarse_stats
            options are:
                const: constant mapping over all fine_sigma
                'sqrt': sqrt scaling of N weighting of fine_sigma
        sig_reward_sqrt; fits fine grained sigma and reward per subject with sqrt in d mapping

        inits is a tuple specifying (total time T, intertrival interval t_w, time step dt, size of grid, lapse rate)
        '''
        self.model_type = model_type
        self.params = log_params
        self.T, self.t_w, self.dt, self.size, self.lapse = inits
        self.N_array = np.array([8, 12, 16])

        self.bell_func = BellmanUtil(self.T, self.t_w, self.size, self.dt)
        self.g_values = self.bell_func.g_values

        if model_type[0] == 'sig':
            self.fine_sigma = np.exp(log_params[0])
            self.reward = 1
            self.punishment = 0
        elif model_type[0] == 'sig_reward':
            self.fine_sigma = np.exp(log_params[0])
            self.reward = np.exp(log_params[1])
            self.punishment = 0
        elif model_type[0] == 'epsilon_punish':
            self.fine_sigma = np.exp(log_params[0])
            self.reward = 1
            self.punishment = np.exp(log_params[1])
        else:
            raise Exception("Invalid entry in first argument of model_type")



        #something buggy going on with the N array
        self.finemodel = FineGrained(self.fine_sigma, model_type[2], int(1e5), np.array([8, 12, 16]))
        self.stats = self.finemodel.coarse_stats

        self.rho_vec = np.zeros(self.stats.shape[0])
        self.decision_vec = np.zeros((self.size, int(self.T/self.dt), self.stats.shape[0]))
        self.trans_vec = np.zeros((self.size, self.size, self.stats.shape[0]))
        for i in range(self.stats.shape[0]):
            mu = self.stats[i, :, 0]
            sigma = self.stats[i, :, 1]
            prob_grid = self.bell_func.trans_probs(sigma, mu)
            self.trans_vec[:,:,i] = prob_grid
            rho = self.bell_func.solve_rho(self.reward, self.punishment,
                            model_type[1], sigma, mu, prob_grid)
            self.rho_vec[i] = rho
            self.decision_vec[:,:,i] = self.bell_func.back_induct(self.reward, self.punishment, rho,
                                    sigma, mu, prob_grid, model_type[1])[1]


    def show_bounds(self):
        for i in range(self.stats.shape[0]):
            decision = self.decision_vec[:,:,i]
            plt.figure()
            plt.title('N = {}'.format(self.N_array[i]))
            plt.imshow(decision)

    def show_trans_probs(self):
        for i in range(self.stats.shape[0]):
            probs = self.trans_vec[:,:,i]
            plt.figure()
            plt.title(str(self.N_array[i]))
            plt.imshow(probs)

    def simulate_observer(self, N, condition):
        N_index = list(self.N_array).index(N)
        dt = self.dt
        T = self.T
        g_values = self.g_values
        C = condition
        if C != 1 and C != 0:
            raise Exception('condition must be 0 (abs) or 1 (pres)')
        decisions = self.decision_vec[:,:,N_index]
        mu = self.stats[N_index, :, 0]
        sigma = self.stats[N_index, :, 1]

        dec_vec = decisions[:, 0]
        abs_bound = g_values[np.amax(np.where(dec_vec == 1)[0])]
        pres_bound = g_values[np.where(dec_vec == 2)[0][0]]

        D_t = 0
        t = 0

        g_trajectory = np.ones(int(T / dt)) * 0.5
        D_trajectory = np.zeros(int(T / dt))

        while t < T :
            D_t = D_t + norm.rvs(mu[C]*dt, sigma[C]*dt)

            g_t = self.bell_func.D_to_g(D_t)
            D_trajectory[int(t / dt)] = D_t
            g_trajectory[int(t / dt)] = g_t
            t += dt

            if g_t < abs_bound:
                return (0, t, g_trajectory, D_trajectory)

            if g_t > pres_bound:
                return (1, t, g_trajectory, D_trajectory)

        return (np.NaN, T, g_trajectory, D_trajectory)


    def get_rt(self, N, condition, numsims=5000, parallelize=False):
        C = condition
        if C != 1 and C != 0:
            raise Exception('condition must be 0 (abs) or 1 (pres)')

        if not parallelize:
            observer_outputs = []
            for i in range(numsims):
                observer_outputs.append(self.simulate_observer(N, C))
        elif parallelize:
            cores = mulpro.cpu_count()
            pool = mulpro.Pool(processes=cores - 1)
            observer_outputs = pool.map(self.simulate_observer, (N, condition))
        response_info = np.array([(x[0], x[1]) for x in observer_outputs])
        return response_info


    def get_kde_dist(self, abs_rts, pres_rts, plot = False, ax = None):
        # 2x2 matrix of distributions, i (row) is the underlying condition C
        # and j (column) is the response
        dist = []
        sorted_rts = []
        perturb = norm.rvs(0, 0.01)
        sim_rt = [abs_rts, pres_rts]
        for i in range(2):
            cur_rt = sim_rt[i]
            for j in range(2):
                if not np.any(cur_rt[:,0] == j):
                    # case where there are none of the responses given in the model simulation
                    dist.append(uniform)
                    sorted_rts.append([])
                else:
                    i_j_sim_rt_marked = np.array(cur_rt[np.where(cur_rt[:,0] == j)[0]])
                    i_j_sim_rt = i_j_sim_rt_marked[:,1]
                    # if they are all the same or of size 1, perturb to allow kde
                    if np.var(i_j_sim_rt) == 0 or i_j_sim_rt.size == 1:
                    # if they are all the same, perturb to allow kde
                        i_j_sim_rt = np.append(i_j_sim_rt, i_j_sim_rt[0] + perturb)
                    if plot and i == j:
                        if i == 0:
                            sns.kdeplot(i_j_sim_rt, bw=0.1, shade=True, color = 'purple',
                                        label='Sim: con. = {0}, resp. = {1}'.format(i,j), ax=ax)
                        else:
                            sns.kdeplot(i_j_sim_rt, bw=0.1, shade=True, color = 'yellow',
                                        label='Sim: con. = {0}, resp. = {1}'.format(i,j), ax=ax)
                    sorted_rts.append( [i_j_sim_rt] )
                    dist.append(gaussian_kde(i_j_sim_rt, bw_method=0.1))

        return np.reshape(dist, (2,2)), np.reshape(sorted_rts, (2,2))

    def get_single_N_likelihood(self, data, dist_matrix, sorted_rt, reward):
        temp = np.mean(np.array(data['rt']))

        abs_0_sim_rt_dist = dist_matrix[0, 0]
        num_abs_0 = len(sorted_rt[0, 0])

        pres_1_sim_rt_dist = dist_matrix[1, 1]
        num_pres_1 = len(sorted_rt[1, 1])

        abs_1_sim_rt_dist = dist_matrix[0, 1]
        num_abs_1 = len(sorted_rt[0, 1])

        pres_0_sim_rt_dist = dist_matrix[1, 0]
        num_pres_0 = len(sorted_rt[1, 0])

        total_pres = num_pres_0 + num_pres_1
        total_abs = num_abs_0 + num_abs_1

        pres_rts_0 = data.query('resp == 2 & target == \'Present\'').rt.values
        pres_rts_1 = data.query('resp == 1 & target == \'Present\'').rt.values

        abs_rts_0 = data.query('resp == 2 & target == \'Absent\'').rt.values
        abs_rts_1 = data.query('resp == 1 & target == \'Absent\'').rt.values


        # frac_pres_inc = len(pres_rts_0) / (len(pres_rts_0) + len(pres_rts_1))
        # frac_pres_corr = len(pres_rts_1) / (len(pres_rts_0) + len(pres_rts_1))
        log_like_pres = np.concatenate((np.log(num_pres_0 / total_pres) +
                                        np.log(pres_0_sim_rt_dist.pdf(pres_rts_0)),
                                        np.log(num_pres_1 / total_pres) +
                                        np.log(pres_1_sim_rt_dist.pdf(pres_rts_1))))

        # frac_abs_inc = len(abs_rts_1) / (len(abs_rts_0) + len(abs_rts_1))
        # frac_abs_corr = len(abs_rts_0) / (len(abs_rts_0) + len(abs_rts_1))
        log_like_abs = np.concatenate((np.log(num_abs_0 / total_abs) +
                                       np.log(abs_0_sim_rt_dist.pdf(abs_rts_0)),
                                       np.log(num_abs_1 / total_abs) +
                                       np.log(abs_1_sim_rt_dist.pdf(abs_rts_1))))

        log_like_all = np.concatenate((log_like_pres, log_like_abs))

        likelihood_pertrial = (1 - self.lapse) * np.exp(log_like_all) + \
            (self.lapse / 2) * np.exp(-reward / temp)
        return -np.sum(np.log(likelihood_pertrial))

    def get_data_likelihood(self, sub_data):
        print(self.fine_sigma, self.reward, self.punishment)
        likelihood = 0
        data = [sub_data.query('setsize == 8'), sub_data.query('setsize == 12'),
                sub_data.query('setsize == 16')]

        for i in range(self.stats.shape[0]):
            N = self.N_array[i]
            abs_rt = self.get_rt(N, 0)
            pres_rt = self.get_rt(N, 1)
            dist_matrix = self.get_kde_dist(abs_rt, pres_rt)[0]
            sorted_rt = self.get_kde_dist(abs_rt, pres_rt)[1]
            likelihood += self.get_single_N_likelihood(data[i], dist_matrix, sorted_rt, self.reward)

        return likelihood
