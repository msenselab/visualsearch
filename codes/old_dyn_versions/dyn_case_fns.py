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


N_array = [8, 12, 16]

def plot_trends(sub_data, sim_data_dict):
    '''
    sub_data an array with each entry subject data for given N
    '''
    for i in range(len(N_array)):
        sim_data_N = np.array(sim_data_dict[N_array[i]])
        abs_sim_mu_N = np.mean(sim_data_N)
        real_mu_N = np.mean(sub_data[i])


def f(x, g_t, sigma, mu):
    ''' x_(t + 1) is x
    Formally P(g_(t+1) | x_(t+1), g_t), for a given g_t and g_(t+1) this will only produce
    the appropriate g_(t+1) as an output for a single value of x_(t+1)
    '''
    p_pres = norm.pdf(x, loc=mu[1], scale=sigma[1])
    p_abs = norm.pdf(x, loc=mu[0], scale=sigma[0])

    post = (g_t * p_pres) / (g_t * p_pres + (1 - g_t) * p_abs)
    #TO DO: put all in exponent

    # if sigma[0] < sigma[1] and isinstance(x, np.ndarray) and np.any(np.isnan(x)):
    #     NaNmin = np.nanargmin(post)
    #     NaNmax = np.nanargmax(post)
    #     middle = np.argmin(post[NaNmin+5:NaNmax])+NaNmin
    #     reflect = np.roll(np.flip(post, 0), 2*(middle))
    #     post[0:middle] = reflect[0:middle]
    #     print(f(x[middle], 0.5, sigma, mu))
    #
    if sigma[0] < sigma[1] and isinstance(x, np.ndarray):
        post[np.invert(np.isfinite(post))] = 1.
    elif sigma[1] < sigma[0] and isinstance(x, np.ndarray):
        post[np.invert(np.isfinite(post))] = 0.

    return post

def contraction_find(f, eval_res, layers, init_min, init_max):
    '''
    find the zero of a function on the interval init_min to init_max,
    f must be monotonic on this given interval
    '''
    testspace = np.linspace(init_min, init_max, num = eval_res)
    for i in range(layers):
        f_space = f(testspace)
        asign = np.sign(f_space)
        signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
        if np.all(signchange[1:] == 0):
            #this will sometime come up because numerical errors
            #in f give artificial peaks indicating
            #that it may have a zero when it doesn't
             print('f has no zero on given interval (init_min, init_max)')
             return np.NaN
        new_end = testspace[np.argmax(signchange[1:])+1]
        new_begin = testspace[np.argmax(signchange[1:])]
        root = new_end - (new_end - new_begin)/2
        testspace = np.linspace(new_begin, new_end, num = eval_res)
    return root

def get_rootgrid(sigma, mu, k):
    testx = np.linspace(-50, 50, 10000)
    testeval = f(testx, 0.5, sigma, mu)
    #k factor truncated to deal with boundry conditions
    real_k = int(size*k - 2*np.floor(k/2))
    if sigma[1] < sigma[0]:
        ourpeak = testx[np.argmax(testeval)]
    elif sigma[0] < sigma[1]:
        ourpeak = testx[np.argmin(testeval)]
    rootgrid = np.zeros((size, real_k, 2))  # NxN grid of values for g_t, g_tp1
    g_tp1_values = np.linspace(1e-3, 1 - 1e-3, real_k)

    for i in range(size):
        g_t = g_values[i]
        peak = f(ourpeak, g_t, sigma, mu)
        for j in range(real_k):
            g_tp1 = g_tp1_values[j]
            if sigma[1] < sigma[0] and g_tp1 > peak:
                skiproot = True
            elif sigma[0] < sigma[1] and g_tp1 < peak:
                skiproot = True
            else:
                skiproot = False

            if not skiproot:
                def rootfunc(x):
                    if type(x) == float:
                        x = np.array([x])
                    return g_tp1 - f(x, g_t, sigma, mu)
                rootgrid[i, j, 0] = contraction_find(rootfunc, 30, 4, -100, ourpeak)
                rootgrid[i, j, 1] = contraction_find(rootfunc, 30, 4, ourpeak, 100)
            elif skiproot:
                    rootgrid[i, j, 0] = np.NaN
                    rootgrid[i, j, 1] = np.NaN
    return rootgrid

def p_new_ev(x, g_t, sigma, mu):
    ''' The probability of a given observation x_(t+1) given our current belief
    g_t'''
    def eval(x):
        p_pres = norm.pdf(x, loc=mu[1], scale=sigma[1])
        p_abs = norm.pdf(x, loc=mu[0], scale=sigma[0])
        return p_pres * g_t + p_abs * (1 - g_t)
    if type(x) == float:
        return eval(x)
    else:
        return np.where(np.isnan(x), np.zeros_like(x), eval(x))

def f_prime(x, g_t, sigma, mu):
    '''
    Analytic derivative of f
    '''
    pres_eval = g_t * norm.pdf(x, loc=mu[1], scale=sigma[1])
    abs_eval = (1- g_t) * norm.pdf(x, loc=mu[0], scale=sigma[0])
    a = (x - mu[1])/sigma[1]**2
    b = (x - mu[0])/sigma[0]**2

    gradient = -((a*pres_eval * (pres_eval + abs_eval) - \
        pres_eval*(a*pres_eval + b*abs_eval))/(pres_eval + abs_eval)**2)

    return np.where(gradient == -0.0, np.full(x.shape, 1e-2), gradient)
    return gradient

def jacobian(roots, g_t, sigma, mu):
    '''
    compute the jacobian scaling factor for root update_probs
    non-existant roots (NaN) evaluate to inf in the derivative
    and are set to zero (non-existant root have 0 weight)
    '''
    # jacob = np.where(np.isnan(f_prime(roots, g_t, sigma, mu)), np.zeros_like(roots), \
    #     1/abs(f_prime(roots, g_t, sigma, mu)))
    jacob = np.where(np.isnan(f_prime(roots, g_t, sigma, mu)), np.zeros_like(roots), \
        1/abs(f_prime(roots, g_t, sigma, mu)))
    return jacob

def update_probs(rootgrid, sigma, mu, resolution = 'low'):
    prob_grid = np.zeros((size, size))
    high_res_prob_grid = np.zeros(rootgrid.shape[:2]).T
    k = int(np.ceil(rootgrid.shape[1]/rootgrid.shape[0]))
    if k%1 == 1:
        k_boundry = int(np.floor(k/2))
    else:
        k_boundry = int(np.floor(k/2)+1)
    dg = (g_values[1] - g_values[0])/k

    for i in range(size):
        prob_slice = np.zeros(size)
        g_t = g_values[i]  # Pick ith value of g at t
        # Slice roots of our given g_t across all g_(t+1)
        roots = rootgrid[i, :, :]
        # Find the likelihood of roots x_(t+1)
        new_g_probs = p_new_ev(roots, g_t, sigma, mu)
        new_g_probs = new_g_probs*jacobian(roots, g_t, sigma, mu)
        new_g_probs = np.sum(new_g_probs, axis=1)  # Sum across both roots
        new_g_probs = new_g_probs  / ( np.sum(new_g_probs) * dg )  # Normalize
        ###line above it is sometime possible to get divide by zero if the roots
        ### are so abnormal that they all get evaluated to zero, hence will
        ###not return full transition probabilities... caused by instability
        ### in f function
        high_res_prob_grid[:, i] = new_g_probs
        prob_slice[1: -1] = np.sum(np.reshape(new_g_probs[k_boundry: -k_boundry], (size-2, k)), axis = 1)\
            /(k)
        prob_slice[0] = np.sum(new_g_probs[0: k_boundry])/k_boundry
        prob_slice[-1] = np.sum(new_g_probs[-k_boundry:])/k_boundry
        prob_grid[:, i] = prob_slice
    if resolution == 'high':
        return high_res_prob_grid
    else:
        return prob_grid

def discrimination_check(stats):
    '''
    Computes the discriminability of pres/abs distributions
    for each N in stats
    '''
    d_prime = np.zeros(len(N_array))
    for i in range(len(N_array)):
        stats_N = stats[i]
        delta_mu = stats_N[1, 0] - stats_N[0, 0]
        denom = np.sqrt(0.5 * (stats_N[0, 1] + stats_N[1, 1]))
        d_prime[i] = delta_mu/denom
    return d_prime

    # if np.var(pres_1_sim_rt) == 0:
    #     mean = np.mean(pres_1_sim_rt)
    #     perturb = norm.rvs(mean, 0.01)
    #     pres_1_sim_rt[0] = mean + perturb

        # #simulated response times for C = 0
        # abs_sim_rt = sim_rt[0]
        # #simulated response times for C = 1
        # pres_sim_rt = sim_rt[1]
        #
        # perturb = norm.rvs(0, 0.01)
        #
        # # Simulated model distribution for resp = 0, C = 0
        # if not np.any(abs_sim_rt[:, 0] == 0): #case where there are no correct responses
        #     # filler distribution as frac_pres_cor will eval to 0
        #     abs_0_sim_rt_dist = uniform
        # else:
        #     abs_0_sim_rt = np.array(abs_sim_rt[np.where(abs_sim_rt[:,0] == 0)[0]])[:,1]
        #     if np.var(abs_0_sim_rt) == 0 or abs_0_sim_rt.size == 1:
        #         abs_0_sim_rt = np.append(abs_0_sim_rt, abs_0_sim_rt[0] + perturb)
        #     abs_0_sim_rt_dist = gaussian_kde(abs_0_sim_rt, bw_method=0.1)
        #
        # # Simulated model distribution for resp = 1, C = 1
        # if not np.any(pres_sim_rt[:,0] == 1):
        #     # filler distribution as frac_pres_cor will eval to 0
        #     pres_1_sim_rt_dist = uniform
        # else:
        #     pres_1_sim_rt = np.array(pres_sim_rt[np.where(pres_sim_rt[:,0] == 1)[0]])[:,1]
        #     if np.var(pres_1_sim_rt) == 0 or pres_1_sim_rt.size == 1:
        #         pres_1_sim_rt = np.append(pres_1_sim_rt, pres_1_sim_rt[0] + perturb)
        #     pres_1_sim_rt_dist = gaussian_kde(pres_1_sim_rt, bw_method=0.1)
        #
        # # Simulated model distribution for resp = 1, C = 0
        # if np.all(abs_sim_rt[:, 0] == 0):
        #     # filler distribution as frac_pres_cor will eval to 0
        #     abs_1_sim_rt_dist = uniform
        # else:
        #     abs_1_sim_rt = np.array(abs_sim_rt[np.where(abs_sim_rt[:,0] != 0)[0]])[:,1]
        #     if np.var(pres_1_sim_rt) == 0 or pres_1_sim_rt.size == 1:
        #         abs_1_sim_rt = np.append(abs_1_sim_rt, abs_1_sim_rt[0] + perturb)
        #     abs_1_sim_rt_dist = gaussian_kde(abs_1_sim_rt, bw_method=0.1)
        #
        # # Simulated model distribution for resp = 0, C = 1
        # if np.all(pres_sim_rt[:, 0] == 1):
        #     # filler distribution as frac_pres_inc will eval to 0
        #     pres_0_sim_rt_dist = uniform
        # else:
        #     pres_0_sim_rt = np.array(pres_sim_rt[np.where(pres_sim_rt[:,0] != 1)[0]])[:,1]
        #     if np.var(pres_0_sim_rt) == 0 or pres_0_sim_rt.size == 1:
        #         pres_0_sim_rt = np.append(pres_0_sim_rt, pres_0_sim_rt[0] + perturb)
        #     pres_0_sim_rt_dist = gaussian_kde(pres_0_sim_rt, bw_method=0.1)

            # def anim_update(i):
            #     ax.azim = (i / 540) * 360
            #     plt.draw()
            #     return
            #
            # Writer = writers['ffmpeg']
            # writer = Writer(fps=60, bitrate=1800)
            # anim = FuncAnimation(fig, anim_update, frames=360)
            # anim.save(savepath + '/subject_{}_bayes_opt_testpoints.mp4'.format(subject_num), writer=writer)
