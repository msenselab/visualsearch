'''
June 2018

Shared functions for the dynamic case model. Keeps a lot of repetetive clutter out of other scripts.
'''

import numpy as np
from scipy.stats import norm


def d_map(N, epsilons, fine_sigma):
    return -(1 / (2 * fine_sigma**2)) + np.log(1 / N) + \
        np.log(np.sum(np.exp(epsilons / fine_sigma**2)))


def sample_epsilon(C, N, fine_sigma):
    epsilons = np.random.normal(0, fine_sigma, N)
    if C == 1:
        epsilons[0] = np.random.normal(1, fine_sigma)
    return epsilons


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


def p_new_ev(x, g_t, sigma, mu):
    ''' The probability of a given observation x_(t+1) given our current belief
    g_t'''
    p_pres = np.exp(- (x - mu[1])**2 /
                    (2 * sigma[1]**2)) / np.sqrt(2 * np.pi * sigma[1]**2)
    p_abs = np.exp(- (x - mu[0])**2 / (2 * sigma[0]**2)) / \
        np.sqrt(2 * np.pi * sigma[0]**2)
    return p_pres * g_t + p_abs * (1 - g_t)


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
