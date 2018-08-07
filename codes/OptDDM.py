import itertools as it
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm, uniform
import multiprocessing as mulpro
import numpy as np
from FineGrained import FineGrained
from BellmanUtil import BellmanUtil


class OptDDM:
    def __init__(self, params, model_type):
        self.model_type = model_type
        self.params = params

        # GLOBAL CONSTANTS
        self.T = 10
        self.t_w = 0.5
        self.size = 100
        self.d_t = 0.05
        self.lapse = 1e-6
        self.d_map_samples = int(1e5)
        self.N_list = [8, 12, 16]

        if model_type[0] == 'sig':
            fine_sigma = params[0]
            reward = 1
            punishment = 0
        elif model_type[0] == 'sig_reward':
            fine_sigma = params[0]
            reward = params[1]
            punishment = 0
        elif model_type[0] == 'epsilon_punish':
            fine_sigma = params[0]
            reward = 1
            punishment = params[1]
        else:
            raise Exception("Invalid entry in first argument of model_type")

        bell_func = BellmanUtil(self.T, self.t_w, self.size, self.d_t)
        finemodel = FineGrained(fine_sigma, model_type[2], self.d_map_samples)
        self.stats = finemodel.coarse_stats

        self.rho_vec = np.zeros(self.stats.shape[0])
        decision_vec = np.zeros(self.stats.shape[0])
        trans_vec = np.zeros(self.stats.shape[0])
        for i in range(self.stats.shape[0]):
            mu = self.stats[i, :, 0]
            sigma = self.stats[i, :, 1]
            prob_grid = bell_func.trans_probs(sigma, mu)
            trans_vec[i] = prob_grid
            rho = bell_func.solve_rho(reward, punishment,
                            model_type[1], sigma, mu, prob_grid)
            rho_vec[i] = rho
            decisions = bell_func.back_induct(reward, punishment, rho,
                                    sigma, mu, prob_grid, model_type[1])[1]
            decision_vec[i] = decisions

        self.rho = rho_vec

    def get_bounds():
        for i in range(self.stats.shape[0]):
            decisions = decision_vec[i]
            plt.figure()
            plt.title(str(N_array[i]))
            plt.imshow(decisions)
            plt.show()

    def get_trans_probs():
        for i in range(self.stats.shape[0]):
            probs = trans_vec[i]
            plt.figure()
            plt.title(str(N_array[i]))
            plt.imshow(probs)
            plt.show()

    def simulate_observer(N, condition):
        N_index = N_list.index(N)
        C = condition
        decisions = decisions_vec[N_index]
        mu = self.self.stats[N_index, :, 0]
        sigma = self.self.stats[N_index, :, 1]

        dec_vec = decisions[:, 0]
        abs_bound = g_values[np.amax(np.where(dec_vec == 1)[0])]
        pres_bound = g_values[np.where(dec_vec == 2)[0][0]]

        D_t = 0
        t = 0

        g_trajectory = np.ones(int(T / self.dt)) * 0.5
        D_trajectory = np.zeros(int(T / self.dt))

        while t < T:
            D_t = D_t + norm.rvs(mu[C]*self.dt, sigma[C]*self.dt)

            g_t = D_to_g(D_t)
            D_trajectory[int(t / dt)] = D_t
            g_trajectory[int(t / dt)] = g_t
            t += dt

            if g_t < abs_bound:
                return (0, t, g_trajectory, D_trajectory)

            if g_t > pres_bound:
                return (1, t, g_trajectory, D_trajectory)

        return (np.NaN, T, g_trajectory, D_trajectory)


    def get_rt(N, condition, numsims=5000, parallelize=False):
        C_vals = [C] * numsims
        arglists = it.product(C_vals, [decisions], [sigma], [mu], [dt])
        if not parallelize:
            observer_outputs = []
            for arglist in arglists:
                observer_outputs.append(simulate_observer(arglist))
        elif parallelize:
            cores = mulpro.cpu_count()
            pool = mulpro.Pool(processes=cores - 1)
            observer_outputs = pool.map(simulate_observer, arglists)
        response_info = np.array([(x[0], x[1]) for x in observer_outputs])
        return response_info


    def get_kde_dist(abs_rts, pres_rts, plot = False, ax = None):
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

    def get_single_N_likelihood(data, dist_matrix, sorted_rt, reward):
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

        likelihood_pertrial = (1 - lapse) * np.exp(log_like_all) + \
            (lapse / 2) * np.exp(-reward / temp)
        return -np.sum(np.log(likelihood_pertrial))

    def get_data_likelihood(sub_data, log_reward, log_punishment, log_fine_sigma,
                            reward_scheme, fine_model_type):
        fine_sigma = np.exp(log_fine_sigma)
        reward = np.exp(log_reward)
        punishment = -np.exp(log_punishment)
        print(fine_sigma, reward, punishment)
        likelihood = 0
        data = [sub_data.query('setsize == 8'), sub_data.query('setsize == 12'),
                sub_data.query('setsize == 16')]

        self.stats = FineGrained(fine_sigma, fine_model_type, d_map_samples).coarse_stats

        for i in range(self.stats.shape[0]):
            mu = self.stats[i, :, 0]
            sigma = self.stats[i, :, 1]
            probs = trans_probs(sigma, mu)
            rho = solve_rho(reward, punishment, reward_scheme, sigma, mu, probs)
            decisions = back_induct(reward, punishment, rho, sigma, mu,
                                    probs, reward_scheme)[1]
            sim_rt = get_rt(sigma, mu, decisions)
            dist_matrix = get_kde_dist(sim_rt)[0]
            sorted_rt = get_kde_dist(sim_rt)[1]
            likelihood += get_single_N_likelihood(data[i], dist_matrix, sorted_rt, reward)

        return likelihood
