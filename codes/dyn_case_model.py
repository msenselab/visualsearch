
import sys
import itertools as it
import seaborn as sns
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde, norm, uniform
import pandas as pd
from pathlib import Path
import pickle
import multiprocessing as mulpro
from gauss_opt import bayesian_optimisation
import numpy as np


# Returns a path object that works as a string for most functions
datapath = Path("../data/exp1.csv")
savepath = Path("~/Documents/")  # Where to save figures
savepath = str(savepath.expanduser())

try:
    subject_num = sys.argv[1]
    if not subject_num.isnumeric():
        subject_num = 1
        print('Invalid subject number passed at prompt. Setting subject to 1')
except ValueError:
    subject_num = 1
    print('No subject number passed at prompt. Setting subject to 1')

print('Subject number {}'.format(subject_num))

# Loading in experimental data
exp1 = pd.read_csv(datapath, index_col=None)  # read data
exp1.rename(columns={'sub': 'subno'}, inplace=True)

# test_data = pd.read_csv(r"C:\Users\Alex\Documents\synth0_7.csv", index_col = None)
# test_data.rename(columns={'sub': 'subno'}, inplace=True)

# sub_data = test_data.query('subno == {} & dyn == \'Dynamic\''.format(666))
sub_data = exp1.query('subno == {} & dyn == \'Dynamic\''.format(subject_num))

temp = np.mean(np.array(sub_data['rt']))



# GLOBAL CONSTANTS
T = 10
t_w = 0.5
size = 100
g_values = np.linspace(1e-4, 1 - 1e-4, size)
d_map_samples = int(1e5)
dt = 0.05
N_array = np.array([8, 12, 16])
lapse = 1e-6
k = 3


class FineGrained:
    def __init__(self, fine_sigma, model_type, num_samples):
        ''' Instance containing necessary methods and functions to sample decision n_variables
        d from the fine-grained model of search.
            options for model_type are:
            'const' : constant fine_sigma
            'sqrt'  : scale sigma by sqrt of N'''
        self.fine_sigma = fine_sigma
        self.model_type = model_type

        N_min = np.amin(N_array)
        pres_samples = np.zeros((num_samples, N_array.shape[0]))
        abs_samples = np.zeros((num_samples, N_array.shape[0]))
        stats = np.zeros((len(N_array), 2, 2))

        for i in range(len(N_array)):
            N = N_array[i]
            # Determine the scaling of stimulus reliability by N
            if model_type == 'const':
                sigma_N = fine_sigma
            if model_type == 'sqrt':
                # If 'sqrt' scale sigma relative to smallest N in data
                sigma_N = fine_sigma * (np.sqrt(N / N_min))
            for j in range(num_samples):
                pres_samples[j, i] = self.d_map(
                    N, self.sample_epsilon(1, N, sigma_N), sigma_N)
                abs_samples[j, i] = self.d_map(
                    N, self.sample_epsilon(0, N, sigma_N), sigma_N)

            stats[i] = np.array([[np.mean(abs_samples), np.sqrt(np.var(abs_samples))],
                                 [np.mean(pres_samples), np.sqrt(np.var(pres_samples))]])

        self.pres_samples = pres_samples
        self.abs_samples = abs_samples
        self.coarse_stats = stats

    def sample_epsilon(self, C, N, sigma):
        '''
        Returns an N dimensional vector representing draws of evidence
        from each item in the display in a given condition C
        '''
        epsilons = norm.rvs(0, sigma, N)
        if C == 1:
            epsilons[0] = norm.rvs(1, sigma)
        return epsilons

    def d_map(self, N, epsilons, sigma_N):
        '''
        Computes the decisions variable d based on the log likelihood ratio
        between pres and abs conditions, used to bootstrap SDT-like distributions
        in get_coarse_stats function
        '''
        return -(1 / (2 * sigma_N**2)) + np.log(1 / N) + \
            np.log(np.sum(np.exp(epsilons / sigma_N**2)))


def g_to_D(g_t):
    return np.log(g_t / (1 - g_t))


def D_to_g(D_t):
    return np.exp(D_t) / (1 + np.exp(D_t))


def p_gtp1_gt(g_t, g_tp1, sigma, mu):
    D_t = g_to_D(g_t)
    D_tp1 = g_to_D(g_tp1)
    jacobian_factor = 1 / (D_to_g(D_t) * (1 - D_to_g(D_t)))

    pres_draw = g_t * norm.pdf(D_tp1, D_t + mu[1], sigma[1])
    abs_draw = (1 - g_t) * norm.pdf(D_tp1, D_t + mu[0], sigma[0])

    return jacobian_factor * (pres_draw + abs_draw)


def p_Dtp1_Dt(D_t, D_tp1, sigma, mu):
    g_t = D_to_g(D_t)

    pres_draw = g_t * norm.pdf(D_tp1, D_t + mu[1], sigma[1])
    abs_draw = (1 - g_t) * norm.pdf(D_tp1, D_t + mu[0], sigma[0])

    return pres_draw + abs_draw


def trans_probs(sigma, mu, space='g'):
    if space == 'g':
        dg = g_values[1] - g_values[0]
        prob_grid = np.zeros((size, size))
        for i, g_t in enumerate(g_values):
            updates = p_gtp1_gt(g_t, g_values, sigma, mu)
            updates = updates / (np.sum(updates) * dg)
            prob_grid[i, :] = updates

    if space == 'D':
        D_values = np.linspace(0, 1e2, 1e3)
        dD = D_values[1] - D_values[0]
        prob_grid = np.zeros((len(D_values), len(D_values)))
        for i, D_t in enumerate(D_values):
            updates = p_Dtp1_Dt(D_t, D_values, sigma, mu)
            updates = updates / (np.sum(updates) * dD)
            prob_grid[i, :] = updates

    return prob_grid


def back_induct(reward, punishment, rho, sigma, mu, prob_grid, reward_scheme, t_dependent=False):
    dg = g_values[1] - g_values[0]

    # Define the reward array
    if reward_scheme == 'sym':
        R = np.array([(reward, punishment),   # (abs/abs,   abs/pres)V
                      (punishment, reward)])  # (pres/abs, pres/pres) in form decision / actual
    elif reward_scheme == 'epsilon_punish':
        R = np.array([(1, punishment),
                      (punishment, 1)])
    elif reward_scheme == 'asym_reward':
        R = np.array([(reward, punishment),
                      (punishment, 1)])
    else:
        raise Exception('Entered invalid reward_scheme')
    # Decision values are static for a given g_t and independent of t. We compute these
    # in advance
    # N x 2 matrix. First column is resp. abs, second is pres.
    decision_vals = np.zeros((size, 2))
    decision_vals[:, 1] = g_values * R[1, 1] + \
        (1 - g_values) * R[1, 0] - (t_w * rho)  # resp pres
    decision_vals[:, 0] = (1 - g_values) * R[0, 0] + \
        g_values * R[0, 1] - (t_w * rho)  # resp abs

    # Create array to store V for each g_t at each t. N x (T / dt)
    V_full = np.zeros((size, int(T / dt)))
    # At large T we assume val of waiting is zero
    V_full[:, -1] = np.max(decision_vals, axis=1)
    # Corresponding array to store the identity of decisions made
    decisions = np.zeros((size, int(T / dt)))
    decisions[:, -1] = np.argmax(decision_vals, axis=1) + 1

    # Backwards induction
    for index in range(2, int(T / dt) + 1):
        for i in range(size):
            V_wait = np.sum(
                prob_grid[:, i] * V_full[:, -(index - 1)]) * dg - (rho * dt)
            # Find the maximum value b/w waiting and two decision options. Store value and identity.
            V_full[i, -index] = np.amax((V_wait,
                                         decision_vals[i, 0], decision_vals[i, 1]))
            decisions[i, -index] = np.argmax((V_wait,
                                              decision_vals[i, 0], decision_vals[i, 1]))
        if not t_dependent and index > 20:
            absolute_err = np.abs(V_full[:, -index] - V_full[:, -(index - 1)])
            converged = np.all(absolute_err[5:-5] < 1e-5)
            if converged:
                dec_vec = decisions[:, -index]
                # deal with degenerate cases in which there are no 1, 2s
                dec_vec[0] = 1
                dec_vec[-1] = 2
                abs_threshold = np.amax(np.where(dec_vec == 1)[0])
                pres_threshold = np.where(dec_vec == 2)[0][0]
                dec_vec[0:abs_threshold] = 1
                dec_vec[pres_threshold:len(dec_vec)] = 2
                V_full = np.reshape(np.repeat(V_full[:, -index], V_full.shape[1]),
                                    (size, V_full.shape[1]))
                decisions = np.reshape(np.repeat(dec_vec, decisions.shape[1]),
                                       (size, decisions.shape[1]))
                break
    return V_full, decisions


def solve_rho(reward, punishment, reward_scheme, sigma, mu, prob_grid):
    '''
    Root finding procedure to find rho given the constrain V(t=0)=0.
    This criteria comes from the invariance of policy with
    respect to linear shift in V
    '''
    def V_in_rho(log_rho):
        rho = np.exp(log_rho)
        values = back_induct(reward, punishment, rho, sigma, mu,
                             prob_grid, reward_scheme)[0]
        return values[int(size / 2), 0]

    # when optimizing for reward this optimization should be accounted for in choosing bounds
    try:
        opt_log_rho = brentq(
            V_in_rho, -10 + np.log(reward), 10 + np.log(reward))
    except ValueError:
        raise Exception("defective bounds in rho finding procedure")

    return np.exp(opt_log_rho)


def simulate_observer(arglist):
    C, decisions, sigma, mu, dt = arglist

    dec_vec = decisions[:, 0]
    abs_bound = g_values[np.amax(np.where(dec_vec == 1)[0])]
    pres_bound = g_values[np.where(dec_vec == 2)[0][0]]

    D_t = 0
    t = 0

    g_trajectory = np.ones(int(T / dt)) * 0.5
    D_trajectory = np.zeros(int(T / dt))

    while t < T:
        D_t = D_t + norm.rvs(mu[C]*dt, sigma[C]*dt)

        #D_t = norm.rvs(D_t + mu[C], sigma[C]) * dt

        g_t = D_to_g(D_t)
        D_trajectory[int(t / dt)] = D_t
        g_trajectory[int(t / dt)] = g_t
        t += dt

        if g_t < abs_bound:
            return (0, t + 0.1, g_trajectory, D_trajectory)

        if g_t > pres_bound:
            return (1, t + 0.1, g_trajectory, D_trajectory)

    return (np.NaN, T, g_trajectory, D_trajectory)


def get_rt(sigma, mu, decisions, numsims=5000, parallelize=False):
    C_vals = [0] * numsims
    C_vals.extend([1] * numsims)
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
    abs_info = response_info[:numsims, :]
    pres_info = response_info[numsims:, :]

    return (abs_info, pres_info)


def get_kde_dist(sim_rt, plot = False, ax = None):
    # 2x2 matrix of distributions, i (row) is the underlying condition C
    # and j (column) is the response
    dist = []
    sorted_rts = []
    perturb = norm.rvs(0, 0.01)
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


def plot_sig_trends(begin, end, num):
    if num == 1:
        raise Exception("num must be greater than 1")
    model_type = ('sig', 'sym', 'const')
    sig_points = np.linspace(begin, end, num)
    reward = 1
    punishment = 0

    data_array = [sub_data.query('setsize == 8'), sub_data.query('setsize == 12'),
            sub_data.query('setsize == 16')]


    fig, axes = plt.subplots(num, 3, sharex=True, sharey = True, figsize=(10, 8.5), squeeze = True)
    for i, sig in enumerate(sig_points):

        stats = FineGrained(sig, 'const', d_map_samples).coarse_stats
        for j in range(stats.shape[0]):
            mu = stats[j, :, 0]
            sigma = stats[j, :, 1]
            prob_grid = trans_probs(sigma, mu)
            rho = solve_rho(reward, punishment, model_type[1], sigma, mu, prob_grid)
            decisions = back_induct(reward, punishment, rho, sigma, mu, prob_grid, model_type[1])[1]
            sim_rt = get_rt(sigma, mu, decisions)

            ax = axes[i, j]
            ax.set_title('N = {}, sigma = {}'.format(N_array[j], np.round(sig, 3)))
            currdata = data_array[0]
            pres_rts_1 = currdata.query('resp == 1 & target == \'Present\'').rt.values
            abs_rts_0 = currdata.query('resp == 2 & target == \'Absent\'').rt.values

            sns.kdeplot(abs_rts_0, bw=0.1, shade=True, color='darkblue', ax=ax)
            sns.kdeplot(pres_rts_1, bw=0.1, shade=True, color='red', ax=ax)

            get_kde_dist(sim_rt, plot = True, ax = ax)

        if i == num-1:
            ax.set_xlabel('RT (s)')
            ax.set_xlim([0, 11])

def get_single_N_likelihood(data, dist_matrix, sorted_rt, reward):

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

    stats = FineGrained(fine_sigma, fine_model_type, d_map_samples).coarse_stats

    for i in range(stats.shape[0]):
        mu = stats[i, :, 0]
        sigma = stats[i, :, 1]
        probs = trans_probs(sigma, mu)
        rho = solve_rho(reward, punishment, reward_scheme, sigma, mu, probs)
        decisions = back_induct(reward, punishment, rho, sigma, mu,
                                probs, reward_scheme)[1]
        sim_rt = get_rt(sigma, mu, decisions)
        dist_matrix = get_kde_dist(sim_rt)[0]
        sorted_rt = get_kde_dist(sim_rt)[1]
        likelihood += get_single_N_likelihood(data[i], dist_matrix, sorted_rt, reward)

    return likelihood


if __name__ == '__main__':
    model_type = ('sig', 'sym', 'const')
    iter_bayesian_opt = 200
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
    '''

    if model_type[0] == 'sig':
        reward_scheme = model_type[1]
        fine_model_type = model_type[2]

        def subject_likelihood(params):
            log_sigma = params[0]
            return get_data_likelihood(sub_data, 0, -1e5, log_sigma,
                                       reward_scheme, fine_model_type)

        # [n_variables, 2] shaped array with bounds
        bnds = np.array(((-1.7, 1.),))
        x_opt = bayesian_optimisation(n_iters=iter_bayesian_opt, sample_loss=subject_likelihood,
                                      bounds=bnds, n_pre_samples=5)
    if model_type[0] == 'sig_reward':
        reward_scheme = model_type[1]
        fine_model_type = model_type[2]

        def subject_likelihood(params):
            log_sigma = params[0]
            log_reward = params[1]
            return get_data_likelihood(sub_data, log_reward, -1e5, log_sigma,
                                       reward_scheme, fine_model_type)

        # [n_variables, 2] shaped array with bounds
        bnds = np.array(((-1.7, 1.), (-1., 0.5)))
        x_opt = bayesian_optimisation(n_iters=iter_bayesian_opt, sample_loss=subject_likelihood,
                                      bounds=bnds, n_pre_samples=15)

    if model_type[0] == 'sig_punish':
        reward_scheme = model_type[1]
        fine_model_type = model_type[2]

        def subject_likelihood(params):
            log_sigma = params[0]
            log_punishment = params[1]
            return get_data_likelihood(sub_data, 0, log_punishment, log_sigma,
                                       reward_scheme, fine_model_type)

        # [n_variables, 2] shaped array with bounds
        bnds = np.array(((-1.7, 1.), (-5., -0.5)))
        x_opt = bayesian_optimisation(n_iters=iter_bayesian_opt, sample_loss=subject_likelihood,
                                      bounds=bnds, n_pre_samples=5)

    xp, yp = x_opt
    # Pull out each of the log(sigma) that the optimizer tested and put them in an array together
    # with the associated log(likelihood). datarr is (N x 2) where N is the number of optimize samps

    # Plot test points and likelihoods
    fig = plt.figure()
    ax = Axes3D(fig)
    if model_type[0] == 'sig':
        ax.scatter(xp[:, 0], yp, s=100)
    if model_type[0] == 'sig_reward':
        ax.scatter(xp[:, 0], xp[:, 1], yp, s=100)
        ax.set_xlabel('$log(\sigma)$')
        ax.set_ylabel('$log(reward)$')
        ax.set_zlabel('$log(likelihood)$')

    if model_type[0] == 'sig_punish':
        ax.scatter(xp[:, 0], xp[:, 1], yp, s=100)
        ax.set_xlabel('$log(\sigma)$')
        ax.set_ylabel('$log(punishment)$')
        ax.set_zlabel('$log(likelihood)$')



    best_likelihood = np.amin(yp)
    best_params = xp[np.argmin(yp)]
    best_sigma = np.exp(best_params[0])

    # # Plot KDE of distributions for data and actual on optimal fit. First we need to simulate.
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8.5))
    plt.figtext(0.67, 0.67, "-log likelihood = {}".format(np.round(best_likelihood, 3)))

    if model_type[0] == 'sig':
        reward = 1
        punishment = 0
        fig.suptitle('Parameters: sigma = {}'.format(np.round(best_sigma, 3))
                     + ', Reward Scheme: {},'.format(model_type[1])
                     + ' Fine Model: {}'.format(model_type[2]))
    elif model_type[0] == 'sig_reward':
        reward = np.exp(best_params[1])
        punishment = 0
        fig.suptitle('Parameters: sigma = {}'.format(np.round(best_sigma, 3))
                     + '; reward = {}'.format(np.round(reward, 3))
                     + ', Reward Scheme: {},'.format(model_type[1])
                     + ' Fine Model: {}'.format(model_type[2]))
    elif model_type[0] == 'sig_punish':
        punishment = np.exp(best_params[1])
        reward = 1
        fig.suptitle('Parameters: sigma = {}'.format(np.round(best_sigma, 3))
                     + '; punishment = {}'.format(np.round(punishment))
                     + ', Reward Scheme: {},'.format(model_type[1])
                     + ' Fine Model: {}'.format(model_type[2]))

    data_array = [sub_data.query('setsize == 8'), sub_data.query('setsize == 12'),
                  sub_data.query('setsize == 16')]

    all_rt = {}
    display_con = ('pres', 'abs')
    for n_con in it.product(N_array, display_con):
        all_rt[n_con] = []

    stats = FineGrained(best_sigma, model_type[2], d_map_samples).coarse_stats

    for i in range(stats.shape[0]):
        mu = stats[i, :, 0]
        sigma = stats[i, :, 1]
        prob_grid = trans_probs(sigma, mu)
        rho = solve_rho(reward, punishment,
                        model_type[1], sigma, mu, prob_grid)
        decisions = back_induct(reward, punishment, rho,
                                sigma, mu, prob_grid, model_type[1])[1]
        sim_rt = get_rt(sigma, mu, decisions)
        all_rt[N_array[i], 'abs'].append(sim_rt[0][:, 1])
        all_rt[N_array[i], 'pres'].append(sim_rt[1][:, 1])

        plt.figure()
        plt.title(str(N_array[i]))
        plt.imshow(decisions)
        plt.show()

        currdata = data_array[i]
        pres_rts_0 = currdata.query(
            'resp == 2 & target == \'Present\'').rt.values
        pres_rts_1 = currdata.query(
            'resp == 1 & target == \'Present\'').rt.values

        abs_rts_0 = currdata.query(
            'resp == 2 & target == \'Absent\'').rt.values
        abs_rts_1 = currdata.query(
            'resp == 1 & target == \'Absent\'').rt.values

        ax = axes[i]
        ax.set_title('N = {}'.format(N_array[i]))
        sns.kdeplot(abs_rts_0, bw=0.1, shade=True, label='Data: con. = 0, resp. = 0',
                    color='darkblue', ax=ax)
        sns.kdeplot(pres_rts_1, bw=0.1, shade=True, label='Data: con. = 1, resp. = 1',
                    color='red', ax=ax)

        get_kde_dist(sim_rt, plot=True, ax = ax)

        ax.set_ylabel('Density estimate')
        ax.legend()

        if i == 2:
            ax.set_xlabel('RT (s)')
            ax.set_xlim([0, 11])

    plt.savefig(
        savepath + '/subject_{}_bayes_opt_bestfits.png'.format(subject_num))

    fw = open(savepath + '/subject_{}_simrt_and_params.p'.format(subject_num), 'wb')
    outdict = {'best_sigma': best_params[0], 'best_reward': best_params[1], 'sim_rts': all_rt,
               'coarse_stats': stats}
    pickle.dump(outdict, fw)
    fw.close()
