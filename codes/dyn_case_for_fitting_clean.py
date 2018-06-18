'''
May 2018
Ad hoc model with differing sigma for the present and absent cases
'''

import numpy as np
import itertools as it
import seaborn as sns
from scipy.optimize import brentq
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
import pandas as pd
from pathlib import Path
import GPy
import GPyOpt

# Returns a path object that works as a string for most functions
datapath = Path("../data/exp1.csv")

T = 10
t_w = 0.5
size = 100
g_values = np.linspace(1e-3, 1 - 1e-3, size)
d_map_samples = 2000
dt = 0.05
N_array = [8, 12, 16]
lapse = 0.001


# change paths to relative path, so it can be across computers

exp1 = pd.read_csv(datapath, index_col=None)  # read data
# .sub is a keyword, change it
exp1.rename(columns={'sub': 'subno'}, inplace=True)

temp = np.mean(np.array(exp1['rt']))

sub_data = exp1.query('subno == 1 & dyn == \'Dynamic\'')

# where sub data is all data from givne subject under dynamic condition


def fit_subject(sub_data):

    use_BO = False

    def subject_likelihood(sigma):
        return get_likelihood(sub_data, sigma)

    if use_BO:
        bounds = [{'name': 'sigma', 'type': 'continuous', 'domain': (0, 10)}]
        max_iter = 1

        opt_prob = GPyOpt.methods.BayesianOptimization(f=subject_likelihood,
                                                       domain=bounds,
                                                       acquisition_type='EI',
                                                       model_type='GP_MCMC')
        opt_prob.run_optimization(max_iter)

        return opt_prob.x_opt

    bounds = (0, 10)
    x_opt = minimize(subject_likelihood, 1, method='SLSQP', bounds=bounds)
    return x_opt


def get_likelihood(sub_data, sigma):
    sigma = sigma
    print(sigma)
    likelihood = 0
    data = [sub_data.query('setsize == 8'), sub_data.query('setsize == 12'),
            sub_data.query('setsize == 16')]

    stats = get_coarse_stats(sigma, d_map_samples)
    print(stats)

    for i in range(stats.shape[0]):
        mu = [stats[i][0][0], stats[i][1][0]]
        sigma = [stats[i][0][1], stats[i][1][1]]
        rootgrid = get_rootgrid(sigma, mu)
        decisions = back_induct(1, 0, 0.05, sigma, mu, rootgrid)
        sim_rt = get_rt(sigma, mu, decisions[1])
        likelihood = + get_likelihood_N(data[i], sim_rt, 1)

    return likelihood


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
    p_given_true = (g_t * np.exp(- (x - mu[1])**2 / (2 * sigma[1]**2)))
    return p_given_true / (p_given_true + (1 - g_t) * np.exp(- (x - mu[0])**2 / (2 * sigma[0]**2)))


def p_new_ev(x, g_t, sigma, mu):
    ''' The probability of a given observation x_(t+1) given our current belief
    g_t'''
    p_pres = np.exp(- (x - mu[1])**2 /
                    (2 * sigma[1]**2)) / np.sqrt(2 * np.pi * sigma[1]**2)
    p_abs = np.exp(- (x - mu[0])**2 / (2 * sigma[0]**2)) / \
        np.sqrt(2 * np.pi * sigma[0]**2)
    return p_pres * g_t + p_abs * (1 - g_t)


def posterior(x, g_t, C, sigma, mu):
    ''' x_(t + 1) is x
    Formally P(g_(t+1) | x_(t+1), g_t), for a given g_t and g_(t+1) this will only produce
    the appropriate g_(t+1) as an output for a single value of x_(t+1)
    '''
    p_given_true = (g_t * np.exp(- (x - mu[C])**2 / (2 * sigma[C]**2)))
    if C == 1:
        othmean = 0
    elif C == 0:
        othmean = 1
    return p_given_true / (p_given_true +
                           (1 - g_t) * np.exp(- (x - mu[othmean])**2 / (2 * sigma[othmean]**2)))


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
    testx = np.linspace(-150, 150, 1000)
    if sigma[1] < sigma[0]:
        ourpeak = testx[np.argmax(f(testx, 0.5, sigma, mu))]
    elif sigma[0] < sigma[1]:
        ourpeak = testx[np.argmin(f(testx, 0.5, sigma, mu))]
    rootgrid = np.zeros((size, size, 2))  # NxN grid of values for g_t, g_tp1
    for i in range(size):
        g_t = g_values[i]
        for j in range(size):
            g_tp1 = g_values[j]
            try:
                rootgrid[i, j, 0] = brentq(
                    lambda x: g_tp1 - f(x, g_t, sigma, mu), -150, ourpeak)
                rootgrid[i, j, 1] = brentq(
                    lambda x: g_tp1 - f(x, g_t, sigma, mu), ourpeak, 150)
            except ValueError:
                if g_t >= g_tp1:
                    rootgrid[i, j, 0] = -150
                    rootgrid[i, j, 1] = -150
                elif g_t < g_tp1:
                    rootgrid[i, j, 0] = 150
                    rootgrid[i, j, 1] = 150
    return rootgrid


def back_induct(reward, punishment, rho, sigma, mu, rootgrid):
    R = np.array([(reward, punishment),   # (abs/abs,   abs/pres)
                  (punishment, reward)])  # (pres/abs, pres/pres) in form decision / actual

    # Decision values are static for a given g_t and independent of t. We compute these
    # in advance
    # N x 2 matrix. First column is resp. abs, second is pres.
    decision_vals = np.zeros((size, 2))
    decision_vals[:, 1] = g_values * R[1, 1] + (1 - g_values) * R[1, 0]  # respond present
    decision_vals[:, 0] = (1 - g_values) * R[0, 0] + g_values * R[0, 1]  # respond absent

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
    #g_grid = np.array([x[2] for x in observer_outputs])
    response_times = np.array([x[1] for x in observer_outputs])
    return response_times.reshape(2, numsims)


def get_likelihood_N(data, sim_rt, reward):
    pres_rts_0 = np.array(data.query('resp == 2 & target == \'Present\'')['rt'])
    pres_rts_1 = np.array(data.query('resp == 1 & target == \'Present\'')['rt'])

    abs_rts_0 = np.array(data.query('resp == 2 & target == \'Absent\'')['rt'])
    abs_rts_1 = np.array(data.query('resp == 1 & target == \'Absent\'')['rt'])

    # deal with the cases where sim rt are all the same giving 0 var to KDE
    if np.var(sim_rt[1, :]) == 0:
        mean = np.mean(sim_rt[1, :])
        perturb = np.random.normal(mean, 0.1)
        sim_rt[1][0] = mean + perturb

    pres_sim_rt_dist = gaussian_kde(sim_rt[1, :])

    if np.var(sim_rt[0, :]) == 0:
        mean = np.mean(sim_rt[0, :])
        perturb = np.random.normal(mean, 0.1)
        sim_rt[0][0] = mean + perturb

    abs_sim_rt_dist = gaussian_kde(sim_rt[0, :])

    p_like_pres = len(pres_rts_0) / (len(pres_rts_0) + len(pres_rts_1)) * np.sum(pres_sim_rt_dist.pdf(pres_rts_0)) + \
        len(pres_rts_1) / (len(pres_rts_0) + len(pres_rts_1)) * \
        np.sum(pres_sim_rt_dist.pdf(pres_rts_1))

    p_like_abs = len(abs_rts_0) / (len(abs_rts_0) + len(abs_rts_1)) * np.sum(abs_sim_rt_dist.pdf(abs_rts_0)) + \
        len(abs_rts_1) / (len(abs_rts_0) + len(abs_rts_1)) * np.sum(abs_sim_rt_dist.pdf(abs_rts_1))
    print(sim_rt[0, :])
    print(np.mean(sim_rt[0, :]))
    print(p_like_abs)

    print(sim_rt[1, :])
    print(np.mean(sim_rt[1, :]))
    print(p_like_pres)

    p_like = (1 - lapse) * p_like_pres + (lapse / 2) * np.exp(-reward / temp) + \
        (1 - lapse) * p_like_abs + (lapse / 2) * np.exp(-reward / temp)
    return -np.sum(np.log(p_like))
