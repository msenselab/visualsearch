'''
May 2018
Ad hoc model with differing sigma for the present and absent cases
'''

import numpy as np
import sys
import itertools as it
import seaborn as sns
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm
import pandas as pd
from pathlib import Path
from gauss_opt import bayesian_optimisation
from dynamic_adhoc_twosigma import p_new_ev, posterior

# Returns a path object that works as a string for most functions
datapath = Path("../data/exp1.csv")
savepath = Path("~/Documents/")  # Where to save figures
savepath = str(savepath.expanduser())

T = 10
t_w = 0.5
size = 100
g_values = np.linspace(1e-3, 1 - 1e-3, size)
d_map_samples = int(1e4)
dt = 0.05
N_array = [8, 12, 16]
lapse = 0.001

try:
    subject_num = sys.argv[1]
    if not subject_num.isnumeric():
        subject_num = 1
        print('Invalid subject number passed at prompt. Setting subject to 1')
except ValueError:
    subject_num = 1
    print('No subject number passed at prompt. Setting subject to 1')

print('Subject number {}'.format(subject_num))
reward = 1
punishment = -.1

exp1 = pd.read_csv(datapath, index_col=None)  # read data
exp1.rename(columns={'sub': 'subno'}, inplace=True)

temp = np.mean(np.array(exp1['rt']))
sub_data = exp1.query('subno == {} & dyn == \'Dynamic\''.format(subject_num))


def d_map(N, epsilons, fine_sigma):
    return -(1 / (2 * fine_sigma**2)) + np.log(1 / N) + \
        np.log(np.sum(np.exp(epsilons / fine_sigma**2)))


def sample_epsilon(C, N, fine_sigma):
    epsilons = np.random.normal(0, fine_sigma, N)
    if C == 1:
        epsilons[0] = np.random.normal(1, fine_sigma)
    return epsilons


def get_coarse_stats(fine_sigma, num_samples):
    '''
    returns a 2x2 matrix, col 1 is abs stats, col 2 pres stats
    row 1 is the mean and row 2 is the sd
    '''
    stats = np.zeros((len(N_array), 2, 2))
    for i in range(len(N_array)):
        N = N_array[i]
        pres_samples = np.zeros(num_samples)
        abs_samples = np.zeros(num_samples)
        for j in range(num_samples):
            pres_samples[j] = d_map(N, sample_epsilon(1, N, fine_sigma), fine_sigma)
            abs_samples[j] = d_map(N, sample_epsilon(0, N, fine_sigma), fine_sigma)

        stats[i] = np.array([[np.mean(abs_samples), np.sqrt(np.var(abs_samples))],
                             [np.mean(pres_samples), np.sqrt(np.var(pres_samples))]])
    return stats


def f(x, g_t, sigma, mu):
    ''' x_(t + 1) is x
    Formally P(g_(t+1) | x_(t+1), g_t), for a given g_t and g_(t+1) this will only produce
    the appropriate g_(t+1) as an output for a single value of x_(t+1)
    '''
    pres_draw = norm.pdf(x, loc=mu[1], scale=sigma[1])
    abs_draw = norm.pdf(x, loc=mu[0], scale=sigma[0])

    post = (g_t * pres_draw) / (g_t * pres_draw + (1 - g_t) * abs_draw)

    if sigma[0] < sigma[1]:
        post[np.invert(np.isfinite(post))] = 1.
    elif sigma[1] < sigma[0]:
        post[np.invert(np.isfinite(post))] = 0.

    return post


def simulate_observer(arglist):
    C, decisions, sigma, mu, dt = arglist
    step = 0
    t = 0
    g_t = np.ones(int(T / dt)) * 0.5
    while t < (T - dt):
        step += 1
        t = step * dt
        x_t = np.random.normal(mu[C], sigma[C]) * dt
        g_t[step] = posterior(x_t, g_t[step - 1], C, sigma, mu)
        nearest_grid = np.abs(g_values - g_t[step]).argmin()
        decision_t = decisions[nearest_grid, step]
        if decision_t != 0:
            break
    return (decision_t, t, g_t)


def get_rootgrid(sigma, mu):
    testx = np.linspace(-50, 50, 1000)
    testeval = f(testx, 0.5, sigma, mu)
    if sigma[1] < sigma[0]:
        ourpeak = testx[np.argmax(testeval)]
    elif sigma[0] < sigma[1]:
        ourpeak = testx[np.argmin(testeval)]
    rootgrid = np.zeros((size, size, 2))  # NxN grid of values for g_t, g_tp1
    for i in range(size):
        g_t = g_values[i]
        testeval_gt = f(testx, g_t, sigma, mu)
        if sigma[1] < sigma[0]:
            peakval = np.amax(testeval_gt)
        elif sigma[0] < sigma[1]:
            peakval = np.amin(testeval_gt)
        for j in range(size):
            g_tp1 = g_values[j]
            if sigma[1] < sigma[0] and g_tp1 > peakval:
                skiproot = True
            elif sigma[0] < sigma[1] and g_tp1 < peakval:
                skiproot = True
            else:
                skiproot = False

            if not skiproot:
                def rootfunc(x):
                    return g_tp1 - f(x, g_t, sigma, mu)
                testx_neg = np.linspace(-50, ourpeak, 1000)
                testx_pos = np.linspace(ourpeak, 50, 1000)
                testeval_neg = rootfunc(testx_neg)
                testeval_pos = rootfunc(testx_pos)
                rootgrid[i, j, 0] = testx_neg[np.argmin(np.abs(testeval_neg))]
                rootgrid[i, j, 1] = testx_pos[np.argmin(np.abs(testeval_pos))]
            elif skiproot:
                if g_t >= g_tp1:
                    rootgrid[i, j, 0] = -50
                    rootgrid[i, j, 1] = -50
                elif g_t < g_tp1:
                    rootgrid[i, j, 0] = 50
                    rootgrid[i, j, 1] = 50
    return rootgrid


def back_induct(reward, punishment, rho, sigma, mu, rootgrid):
    R = np.array([(reward, punishment),   # (abs/abs,   abs/pres)
                  (punishment, reward)])  # (pres/abs, pres/pres) in form decision / actual

    # Decision values are static for a given g_t and independent of t. We compute these
    # in advance
    # N x 2 matrix. First column is resp. abs, second is pres.
    decision_vals = np.zeros((size, 2))
    decision_vals[:, 1] = (g_values * R[1, 1] + (1 - g_values) * R[1, 0]) - rho * t_w  # resp pres
    decision_vals[:, 0] = ((1 - g_values) * R[0, 0] + g_values * R[0, 1]) - rho * t_w  # resp abs

    # Create array to store V for each g_t at each t. N x (T / dt)
    V_full = np.zeros((size, int(T / dt)))
    # At large T we assume val of waiting is zero
    V_full[:, -1] = np.max(decision_vals, axis=1)

    # Corresponding array to store the identity of decisions made
    decisions = np.zeros((size, int(T / dt)))

    # Backwards induction
    for index in range(2, int(T / dt) + 1):
        tau = (index - 1) * dt
        t = T - tau

        for i in range(size):
            g_t = g_values[i]  # Pick ith value of g at t
            # Slice roots of our given g_t across all g_(t+1)
            roots = rootgrid[i, :, :]
            # Find the likelihood of roots x_(t+1)
            new_g_probs = p_new_ev(roots, g_t, sigma, mu)
            new_g_probs = new_g_probs / np.sum(new_g_probs)  # Normalize
            new_g_probs = np.sum(new_g_probs, axis=1)  # Sum across both roots
            # Sum and subt op cost
            V_wait = np.sum(new_g_probs * V_full[:, -(index - 1)]) - rho * t

            # Find the maximum value b/w waiting and two decision options. Store value and identity.
            V_full[i, -index] = np.amax((V_wait,
                                         decision_vals[i, 0], decision_vals[i, 1]))
            decisions[i, -index] = np.argmax((V_wait,
                                              decision_vals[i, 0], decision_vals[i, 1]))
    return V_full, decisions


def solve_rho(reward, sigma, mu, roots):
    '''
    Root finding procedure to find rho given the constrain V(t=0)=0.
    This criteria comes from the invariance of policy with
    respect to linear shift in V
    '''
    def V_in_rho(log_rho):
        rho = np.exp(log_rho)
        values = back_induct(reward, punishment, rho, sigma, mu, roots)[0]
        return values[int(size/2), 0]

    opt_log_rho = brentq(V_in_rho, -5, np.log(100 * reward))

    return np.exp(opt_log_rho)


def get_rt(sigma, mu, decisions):
    numsims = 2000
    # pool = mulpro.Pool(processes=mulpro.cpu_count())
    C_vals = [0] * numsims
    C_vals.extend([1] * numsims)
    arglists = it.product(C_vals, [decisions], [sigma], [mu], [dt])
    # observer_outputs = pool.map(simulate_observer, arglists)
    # pool.close()
    observer_outputs = []
    for arglist in arglists:
        observer_outputs.append(simulate_observer(arglist))
    response_times = np.array([x[1] for x in observer_outputs])
    return response_times.reshape(2, numsims)


def get_single_N_likelihood(data, sim_rt, reward):

    pres_rts_0 = data.query('resp == 2 & target == \'Present\'').rt.values
    pres_rts_1 = data.query('resp == 1 & target == \'Present\'').rt.values

    abs_rts_0 = data.query('resp == 2 & target == \'Absent\'').rt.values
    abs_rts_1 = data.query('resp == 1 & target == \'Absent\'').rt.values

    # deal with the cases where sim rt are all the same giving 0 var to KDE
    if np.var(sim_rt[1, :]) == 0:
        mean = np.mean(sim_rt[1, :])
        perturb = np.random.normal(mean, 0.01)
        sim_rt[1, 0] = mean + perturb

    pres_sim_rt_dist = gaussian_kde(sim_rt[1, :], bw_method=0.1)

    if np.var(sim_rt[0, :]) == 0:
        mean = np.mean(sim_rt[0, :])
        perturb = np.random.normal(mean, 0.1)
        sim_rt[0, 0] = mean + perturb

    abs_sim_rt_dist = gaussian_kde(sim_rt[0, :], bw_method=0.1)

    frac_pres_inc = len(pres_rts_0) / (len(pres_rts_0) + len(pres_rts_1))
    frac_pres_corr = len(pres_rts_1) / (len(pres_rts_0) + len(pres_rts_1))
    log_like_pres = np.concatenate((np.log(frac_pres_inc) +
                                    np.log(pres_sim_rt_dist.pdf(pres_rts_0)),
                                    np.log(frac_pres_corr) +
                                    np.log(pres_sim_rt_dist.pdf(pres_rts_1))))

    frac_abs_inc = len(abs_rts_1) / (len(abs_rts_0) + len(abs_rts_1))
    frac_abs_corr = len(abs_rts_0) / (len(abs_rts_0) + len(abs_rts_1))
    log_like_abs = np.concatenate((np.log(frac_abs_corr) +
                                   np.log(abs_sim_rt_dist.pdf(abs_rts_0)),
                                   np.log(frac_abs_inc) +
                                   np.log(abs_sim_rt_dist.pdf(abs_rts_1))))

    log_like_all = np.concatenate((log_like_pres, log_like_abs))
    # print(sim_rt[0, :])
    # print(np.mean(sim_rt[0, :]))
    # print(log_like_abs)
    #
    # print(sim_rt[1, :])
    # print(np.mean(sim_rt[1, :]))
    # print(log_like_pres)

    likelihood_pertrial = (1 - lapse) * np.exp(log_like_all) + (lapse / 2) * np.exp(-reward / temp)
    return - np.sum(np.log(likelihood_pertrial))


def get_data_likelihood(sub_data, sigma):
    sigma = np.exp(sigma)
    print(sigma)
    likelihood = 0
    data = [sub_data.query('setsize == 8'), sub_data.query('setsize == 12'),
            sub_data.query('setsize == 16')]

    stats = get_coarse_stats(sigma, d_map_samples)
    # print(stats)

    for i in range(stats.shape[0]):
        mu = stats[i, :, 0]
        sigma = stats[i, :, 1]
        rootgrid = get_rootgrid(sigma, mu)
        rho = solve_rho(reward, sigma, mu, rootgrid)
        decisions = back_induct(reward, punishment, rho, sigma, mu, rootgrid)[1]
        sim_rt = get_rt(sigma, mu, decisions)
        likelihood += get_single_N_likelihood(data[i], sim_rt, 1)

    return likelihood


if __name__ == '__main__':
    def subject_likelihood(log_sigma):
        return get_data_likelihood(sub_data, log_sigma)

    bnds = np.array(((-1.7, 1.),))  # [n_samples, 2] shaped array with bounds
    x_opt = bayesian_optimisation(n_iters=15, sample_loss=subject_likelihood,
                                  bounds=bnds, n_pre_samples=15)

    xp, yp = x_opt
    # Pull out each of the log(sigma) that the optimizer tested and put them in an array together
    # with the associated log(likelihood). datarr is (N x 2) where N is the number of optimize samps
    datarr = np.array((x_opt[0].reshape(-1), x_opt[1])).T
    sortdatarr = datarr[np.argsort(datarr[:, 0]), :]

    # Plot test points and likelihoods
    plt.figure()
    plt.scatter(sortdatarr[:, 0], sortdatarr[:, 1])
    plt.xlabel(r'$log(\sigma)$')
    plt.ylabel(r'log(likelihood)')
    plt.title('Subject {} Bayesian Opt tested points'.format(subject_num))

    plt.savefig(savepath + '/subject_{}_bayes_opt_testpoints.png'.format(subject_num))
    # Plot KDE of distributions for data and actual on optimal fit. First we need to simulate.
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8.5))

    best_logsig = datarr[np.argmin(yp), 0]
    best_sigma = np.exp(best_logsig)
    data = [sub_data.query('setsize == 8'), sub_data.query('setsize == 12'),
            sub_data.query('setsize == 16')]

    stats = get_coarse_stats(best_sigma, d_map_samples)
    for i in range(stats.shape[0]):
        mu = stats[i, :, 0]
        sigma = stats[i, :, 1]
        rootgrid = get_rootgrid(sigma, mu)
        rho = solve_rho(reward, sigma, mu, rootgrid)
        decisions = back_induct(reward, punishment, rho, sigma, mu, rootgrid)[1]
        sim_rt = get_rt(sigma, mu, decisions)

        currdata = data[i]
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
            ax.set_xlim([0, 6])

    plt.savefig(savepath + '/subject_{}_bayes_opt_bestfits.png'.format(subject_num))
