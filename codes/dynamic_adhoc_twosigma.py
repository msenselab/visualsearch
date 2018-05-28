'''
May 2018

Ad hoc model with differing sigma for the present and absent cases
'''

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mulpro
import itertools as it
import seaborn as sns
import time
from scipy.optimize import brentq
from matplotlib.animation import FuncAnimation
import pickle
import os

T = 25
t_w = 0.5
size = 100
g_values = np.linspace(1e-3, 1 - 1e-3, size)


def f(x, g_t, sigma):
    ''' x_(t + 1) is x
    Formally P(g_(t+1) | x_(t+1), g_t), for a given g_t and g_(t+1) this will only produce
    the appropriate g_(t+1) as an output for a single value of x_(t+1)
    '''
    p_given_true = (g_t * np.exp(- (x-1)**2 / (2 * sigma[1]**2)))
    return p_given_true / (p_given_true + (1 - g_t) * np.exp(- x**2 / (2 * sigma[0]**2)))


def p_new_ev(x, g_t, sigma):
    ''' The probability of a given observation x_(t+1) given our current belief
    g_t'''
    p_pres = np.exp(- (x - 1)**2 /
                    (2 * sigma[1]**2)) / np.sqrt(2 * np.pi * sigma[1]**2)
    p_abs = np.exp(- x**2 / (2 * sigma[0]**2)) / \
        np.sqrt(2 * np.pi * sigma[0]**2)
    return p_pres * g_t + p_abs * (1 - g_t)


def posterior(x, g_t, C, sigma):
    ''' x_(t + 1) is x
    Formally P(g_(t+1) | x_(t+1), g_t), for a given g_t and g_(t+1) this will only produce
    the appropriate g_(t+1) as an output for a single value of x_(t+1)
    '''
    p_given_true = (g_t * np.exp(- (x - C)**2 / (2 * sigma[int(C)]**2)))
    if C == 1:
        othmean = 0
    elif C == 0:
        othmean = 1
    return p_given_true / (p_given_true +
                           (1 - g_t) * np.exp(- (x - othmean)**2 / (2 * sigma[othmean]**2)))


def simulate_observer(arglist):
    C = arglist[0]
    decisions = arglist[1]
    sigma = arglist[2]
    dt = arglist[3]
    step = 0
    t = 0
    g_t = np.ones(int(T / dt)) * 0.5
    while t < (T - dt):
        step += 1
        t = step * dt
        x_t = np.random.normal(C, sigma[C]) * dt
        g_t[step] = posterior(x_t, g_t[step - 1], C, sigma)
        nearest_grid = np.abs(g_values - g_t[step]).argmin()
        decision_t = decisions[nearest_grid, step]
        if decision_t != 0:
            break
    return (decision_t, t, g_t)


def main(argvec):
    dt, sigma, rho, reward, punishment = argvec
    # First we find the roots of f(x) for all values of g_t and g_(t+1)
    testx = np.linspace(-150, 150, 1000)
    if sigma[1] < sigma[0]:
        ourpeak = testx[np.argmax(f(testx, 0.5, sigma))]
    elif sigma[0] < sigma[1]:
        ourpeak = testx[np.argmin(f(testx, 0.5, sigma))]
    rootgrid = np.zeros((size, size, 2))  # NxN grid of values for g_t, g_tp1
    for i in range(size):
        g_t = g_values[i]
        for j in range(size):
            g_tp1 = g_values[j]
            try:
                rootgrid[i, j, 0] = brentq(
                    lambda x: g_tp1 - f(x, g_t, sigma), -150, ourpeak)
                rootgrid[i, j, 1] = brentq(
                    lambda x: g_tp1 - f(x, g_t, sigma), ourpeak, 150)
            except ValueError:
                if g_t > g_tp1:
                    rootgrid[i, j, 0] = -150
                    rootgrid[i, j, 1] = -150
                elif g_t < g_tp1:
                    rootgrid[i, j, 0] = 150
                    rootgrid[i, j, 1] = 150

    # Define the reward array
    R = np.array([(reward, punishment),   # (abs/abs,   abs/pres)
                  (punishment, reward)])  # (pres/abs, pres/pres) in form decision / actual

    # Decision values are static for a given g_t and independent of t. We compute these
    # in advance
    # N x 2 matrix. First column is resp. abs, second is pres.
    decision_vals = np.zeros((size, 2))
    decision_vals[:, 1] = g_values * R[1, 1] + \
        (1 - g_values) * R[1, 0]  # respond present
    decision_vals[:, 0] = (1 - g_values) * R[0, 0] + \
        g_values * R[0, 1]  # respond absent

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
        # print(t)

        for i in range(size):
            g_t = g_values[i]  # Pick ith value of g at t
            # Slice roots of our given g_t across all g_(t+1)
            roots = rootgrid[i, :, :]
            # Find the likelihood of roots x_(t+1)
            new_g_probs = p_new_ev(roots, g_t, sigma)
            new_g_probs = new_g_probs / np.sum(new_g_probs)  # Normalize
            new_g_probs = np.sum(new_g_probs, axis=1)  # Sum across both roots
            # Sum and subt op cost
            V_wait = np.sum(new_g_probs * V_full[:, -(index - 1)]) - rho * t

            # Find the maximum value b/w waiting and two decision options. Store value and identity.
            V_full[i, -index] = np.amax((V_wait,
                                         decision_vals[i, 0], decision_vals[i, 1]))
            decisions[i, -index] = np.argmax((V_wait,
                                              decision_vals[i, 0], decision_vals[i, 1]))

    # Simulate a pool of observers
    numsims = 2000
    pool = mulpro.Pool(processes=8)
    C_vals = [0] * numsims
    C_vals.extend([1] * numsims)
    arglists = it.product(C_vals, [decisions], [sigma], [dt])
    observer_outputs = pool.map(simulate_observer, arglists)
    pool.close()
    g_grid = np.array([x[2] for x in observer_outputs])
    response_times = np.array([x[1] for x in observer_outputs])
    return g_grid, response_times, decisions


if __name__ == '__main__':
    rho = 0.05
    reward = 3
    punishment = -.1
    dt = 0.02
    multiple_trials = {}
    sigmas = np.linspace(0.9, 15, 10)
    raw_pairs = list(it.product(sigmas, sigmas))
    sigma_list = [x for x in raw_pairs if x[0] != x[1]]
    for sigma in sigma_list:
        multiple_trials[sigma] = main([dt, np.array(sigma), rho, reward, punishment])

    currtime = time.localtime()
    filename = os.getenv("HOME") + '/Documents/two_sigma_search_{}_{}_{}'.format(currtime.tm_mday,
                                                                                 currtime.tm_mon,
                                                                                 currtime.tm_year)
    fw = open(filename + '.p', 'wb')
    outdict = {'trials_dict': multiple_trials, 'T': T, 'dt': dt, 't_w': t_w, 'size': size,
               'g_values': g_values, 'rho': rho, 'reward': reward, 'punishment': punishment}
    pickle.dump(outdict, fw)
    fw.close()

    # Plotting code

    fig = plt.figure()
    ax = fig.add_subplot(111)

    def anim_update(i):
        ax.clear()
        currkey = sigma_list[i]
        obsweights = np.ones(2000) / 2000
        obsbins = np.linspace(0, T, int(T / dt / 5))
        plt.hist(multiple_trials[currkey][1][:2000], color='g', bins=obsbins,
                 label='Target Absent', alpha=0.5, weights=obsweights)
        plt.hist(multiple_trials[currkey][1][2000:], color='b', bins=obsbins,
                 label='Target Present', alpha=0.5, weights=obsweights)
        ax.set_xlim([0, 6])
        ax.set_ylim([0, 1])
        ax.set_title(r'$\sigma_{abs} = $' + str(currkey[0]) + r' $\sigma_{pres} = $' + str(currkey[1]))

    anim = FuncAnimation(fig, anim_update, frames=len(sigma_list), interval=2500)
    anim.save(filename + '_animated_responsedist.mp4')
