'''
May 2018
Ad hoc model with differing sigma for the present and absent cases
'''

import sys
import itertools as it
import seaborn as sns
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, writers
from scipy.stats import gaussian_kde, norm
import pandas as pd
from pathlib import Path
import pickle
from gauss_opt import bayesian_optimisation
from dynamic_adhoc_twosigma import posterior
import numpy as np

# Returns a path object that works as a string for most functions
datapath = Path("../data/exp1.csv")
savepath = Path("~/Documents/")  # Where to save figures
savepath = str(savepath.expanduser())

T = 10
t_w = 0.5
size = 100
g_values = np.linspace(1/size, 1 - 1/size, size)
d_map_samples = int(1e5)
dt = 0.05
N_array = [8, 12, 16]
lapse = 1e-6
k = 3

try:
    subject_num = sys.argv[1]
    if not subject_num.isnumeric():
        subject_num = 1
        print('Invalid subject number passed at prompt. Setting subject to 1')
except ValueError:
    subject_num = 1
    print('No subject number passed at prompt. Setting subject to 1')

print('Subject number {}'.format(subject_num))

exp1 = pd.read_csv(datapath, index_col=None)  # read data
exp1.rename(columns={'sub': 'subno'}, inplace=True)

temp = np.mean(np.array(exp1['rt']))
sub_data = exp1.query('subno == {} & dyn == \'Dynamic\''.format(subject_num))

def d_map(N, epsilons, sigma_N):
    '''
    Computes the decisions variable d based on the log likelihood ratio
    between pres and abs conditions, used to bootstrap SDT-like distributions
    in get_coarse_stats function
    '''
    return -(1 / (2 * sigma_N**2)) + np.log(1 / N) + \
        np.log(np.sum(np.exp(epsilons / sigma_N**2)))

def sample_epsilon(C, N, sigma):
    '''
    Returns an N dimensional vector representing draws of evidence
    from each item in the display in a given condition C
    '''
    epsilons = norm.rvs(0, sigma, N)
    if C == 1:
        epsilons[0] = norm.rvs(1, sigma)
    return epsilons

def get_coarse_stats(fine_sigma, num_samples, model_type):
    '''
    return coarse stats for all N in experimental design
    options for model_type are:
    'const' : constant fine_sigma
    'sqrt'  : scale sigma by sqrt of N
    '''
    stats = np.zeros((len(N_array), 2, 2))
    N_min = np.amin(N_array)

    for i in range(len(N_array)):
        N = N_array[i]
        if model_type == 'const':
            sigma_N = fine_sigma
        if model_type == 'sqrt':
            sigma_N = fine_sigma * (np.sqrt(N)/np.sqrt(N_min))
        pres_samples = np.zeros(num_samples)
        abs_samples = np.zeros(num_samples)
        for j in range(num_samples):
            pres_samples[j] = d_map(N, sample_epsilon(1, N, sigma_N), sigma_N)
            abs_samples[j] = d_map(N, sample_epsilon(0, N, sigma_N), sigma_N)

        stats[i] = np.array([[np.mean(abs_samples), np.sqrt(np.var(abs_samples))],
                             [np.mean(pres_samples), np.sqrt(np.var(pres_samples))]])

    return stats

def g_to_D(g_t):
    return np.log(g_t/(1-g_t))

def D_to_g(D_t):
    return np.exp(D_t)/(1+np.exp(D_t))

def deriv_dg_dD(D_t):
    return D_to_g(D_t)*(1-D_to_g(D_t))

def p_gtp1_gt(g_t, g_tp1, sigma, mu):
    D_t = g_to_D(g_t)
    D_tp1 = g_to_D(g_tp1)
    jacobian_factor = 1/deriv_dg_dD(D_t)

    pres_draw = g_t*norm.pdf(D_tp1, D_t+mu[1], sigma[1])
    abs_draw = (1-g_t)*norm.pdf(D_tp1, D_t+mu[0], sigma[0])

    return jacobian_factor*(pres_draw+abs_draw)

def p_Dtp1_Dt(D_t, D_tp1, sigma, mu):
    g_t = D_to_g(D_t)

    pres_draw = g_t*norm.pdf(D_tp1, D_t+mu[1], sigma[1])
    abs_draw = (1-g_t)*norm.pdf(D_tp1, D_t+mu[0], sigma[0])

    return pres_draw+abs_draw


def trans_probs(sigma, mu, space = 'g'):
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
        prob_grid = np.zeros((len(D_values),len(D_values)))
        for i, D_t in enumerate(D_values):
            updates = p_Dtp1_Dt(D_t, D_values, sigma, mu)
            updates = updates / (np.sum(updates) * dD)
            prob_grid[i, :] = updates

    return prob_grid


def back_induct(reward, punishment, rho, sigma, mu, prob_grid, reward_scheme,
    t_dependent = False):
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
        (1 - g_values) * R[1, 0] - (t_w * rho)  # respond present
    decision_vals[:, 0] = (1 - g_values) * R[0, 0] + \
        g_values * R[0, 1] - (t_w * rho)  # respond absent

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
            V_wait = np.sum(prob_grid[:, i] * V_full[:, -(index - 1)]) * dg - (rho * dt)
            #Find the maximum value b/w waiting and two decision options. Store value and identity.
            V_full[i, -index] = np.amax((V_wait, decision_vals[i, 0], decision_vals[i, 1]))
            decisions[i, -index] = np.argmax((V_wait, decision_vals[i, 0], decision_vals[i, 1]))
        if not t_dependent and index > 20:
            absolute_err = np.abs(V_full[:, -index] - V_full[:, -(index-1)])
            converged = np.all(absolute_err[5:-5] < 1e-5)
            if converged:
                dec_vec = decisions[:, -index]
                #deal with degenerate cases in which there are no 1, 2s
                dec_vec[0] = 1
                dec_vec[-1] = 2
                abs_threshold = np.amax(np.where(dec_vec == 1)[0])
                pres_threshold = np.where(dec_vec == 2)[0][0]
                dec_vec[0:abs_threshold] = 1
                dec_vec[pres_threshold:len(dec_vec)] = 2
                V_full = np.reshape(np.repeat(V_full[:, -index], V_full.shape[1]), (size, V_full.shape[1]))
                decisions = np.reshape(np.repeat(dec_vec, decisions.shape[1]), (size, decisions.shape[1]))
                break
            if index == int(T / dt):
                print('!!!backward induction did not converge to fixed point!!!')
    return V_full, decisions


def solve_rho(reward, punishment, reward_scheme, sigma, mu, prob_grid):
    '''
    Root finding procedure to find rho given the constrain V(t=0)=0.
    This criteria comes from the invariance of policy with
    respect to linear shift in V
    '''
    def V_in_rho(log_rho):
        rho = np.exp(log_rho)
        values = back_induct(reward, punishment, rho, sigma, mu, \
            prob_grid, reward_scheme)[0]
        return values[int(size / 2), 0]

    # when optimizing for reward this optimization should be accounted for in choosing bounds
    try:
        opt_log_rho = brentq(V_in_rho, -10 + np.log(reward), 10 + np.log(reward))
    except ValueError:
            raise Exception("defective bounds in rho finding procedure")

    return np.exp(opt_log_rho)


def simulate_observer(arglist):
    C, decisions, sigma, mu, dt = arglist

    dec_vec = decisions[:,0]
    abs_bound = g_values[np.amax(np.where(dec_vec == 1)[0])]
    pres_bound = g_values[np.where(dec_vec == 2)[0][0]]

    D_t = 0
    t = 0

    g_trajectory = np.ones(int(T/dt))*0.5
    D_trajectory = np.zeros(int(T/dt))

    while t < T:
        if C == 1:
            D_t = norm.rvs(t * mu[C], t* sigma[C]) * dt
        if C == 0:
            D_t = norm.rvs(t * mu[C], t*sigma[C]) * dt

        g_t = D_to_g(D_t)
        D_trajectory[int(t/dt)] = D_t
        g_trajectory[int(t/dt)] = g_t
        t += dt

        if g_t < abs_bound:
            return (0, t, g_trajectory, D_trajectory)

        if g_t > pres_bound:
            return (1, t, g_trajectory, D_trajectory)

    return (0, T, g_trajectory, D_trajectory)



# def alt_sim(prob_grid, decisions):
#     ##COMPUTEs prob distribution and not density
#     dg = (g_values[1] - g_values[0])
#
#     g_t = np.zeros(prob_grid.shape[0])
#     g_t[int(len(g_t)/2)] = 1
#
#     D_t = np.zeros
#
#     dec_vec = decisions[:,0]
#     abs_threshold = np.amax(np.where(dec_vec == 1)[0])
#     pres_threshold = np.where(dec_vec == 2)[0][0]
#
#     resp_dist = np.zeros((2, int(T/dt)))
#     dist_evo = np.zeros((prob_grid.shape[0], int(T/dt)))
#     t = 0
#     remaining_density = 1
#     while t < (T-dt):
#         dist_evo[:, int(t/dt)] = g_t
#         abs_cum_density = np.sum(g_t[:abs_threshold])
#         pres_cum_density = np.sum(g_t[pres_threshold:])
#
#         resp_dist[0, int(t/dt)] = abs_cum_density
#         resp_dist[1, int(t/dt)] = pres_cum_density
#
#         remaining_density = remaining_density - (abs_cum_density + pres_cum_density)
#
#         print(remaining_density)
#
#         g_t = g_t/remaining_density
#
#         g_t = np.matmul(prob_grid[:,abs_threshold:pres_threshold], g_t[abs_threshold:pres_threshold])
#
#         t += dt
#
#     plt.figure()
#     plt.imshow(dist_evo[:, 5:])
#     plt.axhline(y = abs_threshold)
#     plt.axhline(y = pres_threshold)
#
#     resp_abs = resp_dist[0,:]/np.sum(resp_dist[0,:])
#     resp_pres = resp_dist[1,:]/np.sum(resp_dist[1,:])
#
#     return resp_abs, resp_pres, dist_evo


def get_rt(sigma, mu, decisions, numsims = 5000):
    C_vals = [0] * numsims
    C_vals.extend([1] * numsims)
    arglists = it.product(C_vals, [decisions], [sigma], [mu], [dt])
    observer_outputs = []
    for arglist in arglists:
        observer_outputs.append(simulate_observer(arglist))
    response_info = np.array([(x[0], x[1]) for x in observer_outputs])
    abs_info = response_info[:numsims,:]
    pres_info = response_info[numsims:,:]

    return (abs_info, pres_info)


def get_single_N_likelihood(data, sim_rt, reward):

    #simulate response times for C = 0
    abs_rt = sim_rt[0]
    #simulate response times for C = 1
    pres_rt = sim_rt[1]

    # Simulate model distribution for resp = 0, C = 0
    abs_0_sim_rt = np.array(abs_rt[np.where(abs_rt[:,0] == 0)[0]])[:,1]
    abs_0_sim_rt_dist = gaussian_kde(abs_0_sim_rt, bw_method=0.1)

    # Simulate model distribution for resp = 1, C = 1
    pres_1_sim_rt = np.array(pres_rt[np.where(pres_rt[:,0] == 1)[0]])[:,1]
    pres_1_sim_rt_dist = gaussian_kde(pres_1_sim_rt, bw_method=0.1)

    # Simulate model distribution for resp = 1, C = 0
    if np.all(abs_rt[:, 0] == True):
        # filler distribution as frac_pres_cor will eval to 0
        abs_1_sim_rt = np.random.uniform()
    else:
        abs_1_sim_rt = np.array(abs_rt[np.where(abs_rt[:,0] == 1)[0]])[:,1]
    abs_1_sim_rt_dist = gaussian_kde(abs_1_sim_rt, bw_method=0.1)

    # Simulate model distribution for resp = 0, C = 1
    if np.all(pres_rt[:, 0] == True):
        # filler distribution as frac_pres_inc will eval to 0
        pres_0_sim_rt = np.random.uniform()
    else:
        pres_0_sim_rt = np.array(pres_rt[np.where(pres_rt[:,0] == 0)[0]])[:,1]

    pres_0_sim_rt_dist = gaussian_kde(pres_0_sim_rt, bw_method=0.1)


    pres_rts_0 = data.query('resp == 2 & target == \'Present\'').rt.values
    pres_rts_1 = data.query('resp == 1 & target == \'Present\'').rt.values

    abs_rts_0 = data.query('resp == 2 & target == \'Absent\'').rt.values
    abs_rts_1 = data.query('resp == 1 & target == \'Absent\'').rt.values


    frac_pres_inc = len(pres_rts_0) / (len(pres_rts_0) + len(pres_rts_1))
    frac_pres_corr = len(pres_rts_1) / (len(pres_rts_0) + len(pres_rts_1))
    log_like_pres = np.concatenate((np.log(frac_pres_inc) +
                                    np.log(pres_0_sim_rt_dist.pdf(pres_rts_0)),
                                    np.log(frac_pres_corr) +
                                    np.log(pres_1_sim_rt_dist.pdf(pres_rts_1))))

    frac_abs_inc = len(abs_rts_1) / (len(abs_rts_0) + len(abs_rts_1))
    frac_abs_corr = len(abs_rts_0) / (len(abs_rts_0) + len(abs_rts_1))
    log_like_abs = np.concatenate((np.log(frac_abs_corr) +
                                   np.log(abs_0_sim_rt_dist.pdf(abs_rts_0)),
                                   np.log(frac_abs_inc) +
                                   np.log(abs_1_sim_rt_dist.pdf(abs_rts_1))))

    log_like_all = np.concatenate((log_like_pres, log_like_abs))

    likelihood_pertrial = (1 - lapse) * np.exp(log_like_all) + (lapse / 2) * np.exp(-reward / temp)
    return -np.sum(np.log(likelihood_pertrial))

def get_data_likelihood(sub_data, log_reward, log_punishment, log_sigma,
                                        reward_scheme, fine_model_type):
    sigma = np.exp(log_sigma)
    reward = np.exp(log_reward)
    punishment = -np.exp(log_punishment)
    print(sigma, reward, punishment)
    likelihood = 0
    data = [sub_data.query('setsize == 8'), sub_data.query('setsize == 12'),
            sub_data.query('setsize == 16')]

    stats = get_coarse_stats(sigma, d_map_samples, fine_model_type)

    for i in range(stats.shape[0]):
        mu = stats[i, :, 0]
        sigma = stats[i, :, 1]
        probs = trans_probs(sigma, mu)
        rho = solve_rho(reward, punishment, reward_scheme, sigma, mu, probs)
        decisions = back_induct(reward, punishment, rho, sigma, mu,
                                                probs, reward_scheme)[1]
        sim_rt = get_rt(sigma, mu, decisions)
        likelihood += get_single_N_likelihood(data[i], sim_rt, reward)

    return likelihood

if __name__ == '__main__':

    model_type = ('sig_punish', 'epsilon_punish', 'sqrt')
    iter_bayesian_opt = 15
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

        bnds = np.array(((-1.7, 1.),))  # [n_variables, 2] shaped array with bounds
        x_opt = bayesian_optimisation(n_iters=iter_bayesian_opt, sample_loss=subject_likelihood,
                                      bounds=bnds, n_pre_samples=5)
    if model_type[0] == 'sig_reward':
        reward_scheme = model_type[1]
        fine_model_type = model_type[2]
        def subject_likelihood(params):
            log_sigma = params[0]
            log_reward = params[1]
            return get_data_likelihood( sub_data, log_reward, -1e5, log_sigma,
                reward_scheme, fine_model_type)

        bnds = np.array(((-1.7, 1.), (-5., 0.5)))  # [n_variables, 2] shaped array with bounds
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

        bnds = np.array(((-1.7, 1.), (-5., -0.5)))  # [n_variables, 2] shaped array with bounds
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

    if  model_type[0] == 'sig_punish':
        ax.scatter(xp[:, 0], xp[:, 1], yp, s=100)
        ax.set_xlabel('$log(\sigma)$')
        ax.set_ylabel('$log(punishment)$')
        ax.set_zlabel('$log(likelihood)$')

    # def anim_update(i):
    #     ax.azim = (i / 540) * 360
    #     plt.draw()
    #     return
    #
    # Writer = writers['ffmpeg']
    # writer = Writer(fps=60, bitrate=1800)
    # anim = FuncAnimation(fig, anim_update, frames=360)
    # anim.save(savepath + '/subject_{}_bayes_opt_testpoints.mp4'.format(subject_num), writer=writer)

    # # Plot KDE of distributions for data and actual on optimal fit. First we need to simulate.
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8.5))

    best_params = xp[np.argmin(yp)]
    best_sigma = np.exp(best_params[0])

    if model_type[0] == 'sig':
        reward = 1
        punishment = 0
        fig.suptitle('Parameters: sigma = {}'.format(np.round(best_sigma, 3))
                            + ', Reward Scheme: {},'.format(model_type[1]) \
                            + ' Fine Model: {}'.format(model_type[2]))
    elif model_type[0] == 'sig_reward':
        reward = np.exp(best_params[1])
        punishment = 0
        fig.suptitle('Parameters: sigma = {}'.format(np.round(best_sigma, 3))
                            + '; reward = {}'.format(np.round(reward, 3))
                            + ', Reward Scheme: {},'.format(model_type[1]) \
                            + ' Fine Model: {}'.format(model_type[2]))
    elif model_type[0] == 'sig_punish':
        punishment = np.exp(best_params[1])
        reward = 1
        fig.suptitle('Parameters: sigma = {}'.format(np.round(best_sigma, 3))
                            + '; punishment = {}'.format(np.round(punishment))
                            + ', Reward Scheme: {},'.format(model_type[1]) \
                            + ' Fine Model: {}'.format(model_type[2]))

    data_array = [sub_data.query('setsize == 8'), sub_data.query('setsize == 12'),
            sub_data.query('setsize == 16')]

    all_rt = {}
    display_con = ('pres', 'abs')
    for n_con in it.product(N_array, display_con):
        all_rt[n_con] = []


    stats = get_coarse_stats(best_sigma, d_map_samples, model_type[2])

    for i in range(stats.shape[0]):
        mu = stats[i, :, 0]
        sigma = stats[i, :, 1]
        prob_grid = trans_probs(sigma, mu)
        rho = solve_rho(reward, punishment, model_type[1], sigma, mu, prob_grid)
        decisions = back_induct(reward, punishment, rho, sigma, mu, prob_grid, model_type[1])[1]
        sim_rt = get_rt(sigma, mu, decisions)
        all_rt[N_array[i], 'pres'].append(sim_rt)
        plt.figure()
        plt.title(str(N_array[i]))
        plt.imshow(decisions)
        plt.show()

        currdata = data_array[i]
        pres_rts_0 = currdata.query('resp == 2 & target == \'Present\'').rt.values
        pres_rts_1 = currdata.query('resp == 1 & target == \'Present\'').rt.values

        abs_rts_0 = currdata.query('resp == 2 & target == \'Absent\'').rt.values
        abs_rts_1 = currdata.query('resp == 1 & target == \'Absent\'').rt.values

        ax = axes[i]
        ax.set_title('N = {}'.format(N_array[i]))
        sns.kdeplot(sim_rt[1], bw=0.1, shade=True, label='Sim corr pres', color='blue', ax=ax)
        sns.kdeplot(sim_rt[0], bw=0.1, shade=True, label='Sim corr abs', color='orange', ax=ax)
        sns.kdeplot(abs_rts_0, bw=0.1, shade=True, label='Data corr abs', color='red', ax=ax)
        sns.kdeplot(pres_rts_1, bw=0.1, shade=True, label='Data corr pres', color='darkblue', ax=ax)

        ax.set_ylabel('Density estimate')
        ax.legend()

        if i == 2:
            ax.set_xlabel('RT (s)')
            ax.set_xlim([0, 11])

    plt.savefig(savepath + '/subject_{}_bayes_opt_bestfits.png'.format(subject_num))

    fw = open(savepath + '/subject_{}_simrt_and_params.p', 'wb')
    outdict = {'best_sigma': best_params[0], 'best_reward': best_params[1], 'sim_rts': all_rt,
               'coarse_stats': stats}
    pickle.dump(outdict, fw)
    fw.close()
