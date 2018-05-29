'''
Dear Berk,

Hope this finds you well. So the big idea goes like this: recall there are four cases in this
Bellman equation, choose abs, choose pres, switch, stay. Your code solves for both choice options
and the expected value for stay. Moreover, since the expected value for staying only depends future
draws from the CURRENT item, those three values will be constant even as the current item changes.
So what I've begun to do bellow is create a V_base which is basically the output of the ad-hoc
dynamic case model, backwards induction through time. With this as a basis I then loop through N
such grids to do backwards induction through items. There are two main pieces not working,
first finding the x responsible for an update to the phi values, the analog of the root finding
procedure that we were discussing, second the backwards induction through items, which should
be pretty straightforward... it just isn't complete yet. Remember Phi=(phi, phi_bar, beta, beta_bar)
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize
from itertools import product
import seaborn as sbn

# number of items in display
T = 6
t_w = 0.5
N = 8
size = 10
sigma = 1


# values that grid points might take
value_space = np.linspace(10**-3, 1 - 10**-3, size)


def combs(a, r):
    """
    Return successive r-length cartesian product of values in a...
    basically just sets up the grid - I know this doesn't have to be a fucntion
    but thats how it got written in my head and I haven't changed it yet
    """
    a = np.asarray(a)
    dt = np.dtype([('', a.dtype)] * r)
    b = np.fromiter(product(a, repeat=r), dt)
    return b.view(a.dtype).reshape(-1, r)


grid_val_test = combs(value_space, 4)

print(grid_val_test.shape)


def global_posterior(Phi, k):
    '''
    this is g_t in the write up
    '''
    phi = Phi[0], phi_bar = Phi[1], beta = Phi[2], beta_bar = Phi[3]

    pres_likelihood = 1 / N * \
        (phi * beta_bar + phi_bar * beta + (N - k) * phi_bar * beta_bar)
    abs_likelihood = phi_bar * beta_bar
    return pres_likelihood / (pres_likelihood + abs_likelihood)


def local_posterior(Phi, k):
    '''
    this is b_t,k in the write up
    '''
    phi = Phi[0], phi_bar = Phi[1], beta = Phi[2], beta_bar = Phi[3]

    pres_likelihood = phi * beta_bar
    Z_b = phi * beta_bar + phi_bar * beta + \
        (N - k) * phi_bar * beta_bar  # this is the normalizing factor

    return pres_likelihood / Z_b


def p_new_ev_stay(x, Phi, sigma, k):
    '''
    this returns the probability of a new piece of evidence given
    evidence set Phi for the staying case i.e. evidence drawn from
    current location. Phi is a vector length 4.
    '''
    g_t = global_posterior(Phi, k)
    b_t = local_posterior(Phi, k)

    draw_1 = np.exp(-(x - 1)**2 / (2 * sigma**2))
    draw_0 = np.exp(-x**2 / (2 * sigma**2))

    return (1 - g_t) * draw_0 + g_t * (b_t * draw_1 + (1 - b_t) * draw_0)


def p_new_ev_switch(x, Phi, sigma, k):
    '''
    this returns the probability of a new piece of evidence given
    evidence set Phi for the switching case, Phi is a vector length 4
    '''
    phi = Phi[0], phi_bar = Phi[1], beta = Phi[2], beta_bar = Phi[3]
    g_t = global_posterior(phi, phi_bar, beta, beta_bar, k)

    Z_b = phi * beta_bar + phi_bar * beta + (N - k) * phi_bar * beta_bar

    draw_1 = np.exp(-(x - 1)**2 / (2 * sigma**2))
    draw_0 = np.exp(-x**2 / (2 * sigma**2))

    # this is the bit of the equation that captures weights the draws on local post conditioned on target present
    # mostly just for notational convience
    weight_draw_pres = ((phi_bar * beta_bar) / Z_b) * draw_1 + ((phi * beta_bar +
                                                                 phi_bar * beta + (N - (k - 1)) * phi_bar * beta_bar) / Z_b) * draw_0

    return (1 - g_t) * draw_0 + g_t * (weight_draw_pres)


def get_Update_X(Phi_t):
    '''
    takes a Phi and returns a size**4 vector of x's that update current Phi to future Phi
    it should be noted that updates only apply to the phi's and so the same update vector
    can be used for all Phi with given phi, phi_bar
    '''
    update_Xs = np.zeros_like(grid_values, dtype=list)
    phi_roots = np.zeros(size)

    for i in range(size):
        phi_tp1 = value_space[i]
        phi_t = Phi_t[0]
        # the values of X such that Phi_t is updated to each potential new Phi_t+1
        try:
            phi_roots[i] = brentq(lambda x: phi_tp1 - np.exp(-(x - 1)
                                                             ** 2 / (2 * sigma**2)) * phi_t, -150, 150)  # root finding
        except ValueError:
            if phi_t > phi_tp1:
                phi_roots[i] = -150
            elif phi_t < phi_tp1:
                phi_roots[i] = 150

    return phi_roots


print(get_Update_X((0.25, 0.25, 0.25, 0.25)))


def main(argvec):
    dt, sigma, rho, reward, punishment = argvec

    R = np.array([(reward, punishment),   # (abs/abs,   abs/pres)
                  (punishment, reward)])  # (pres/abs, pres/pres) in form decision / actual

    root_gird = np.zeros_like(grid_values, dtype=array)
    for i in range(size):
        for j in range(size):
            # the value of X such that Phi_t is updated to each potential new Phi_t+1
            update_Xs =
            root_grid[i, j, :, :] = updates_Xs

    # N x 2 matrix. First column is resp. abs, second is pres.
    decision_vals = np.zeros((size**4, 2))
    decision_vals[:, 1] = global_posterior(grid_values.flatten, np.zeros(size**4)) * R[1, 1] + (
        1 - global_posterior(grid_values.flatten, np.zeros(size**4))) * R[1, 0]  # respond present
    decision_vals[:, 0] = (1 - global_posterior(grid_values.flatten, np.zeros(size**4))) * R[0, 0] + \
        global_posterior(grid_values.flatten, np.zeros(
            size**4)) * R[0, 1]  # respond absent

    V_base = np.zeros((size**4, int(T / dt)))
    V_base[:, -1] = np.max(decision_vals, axis=1)

    decisions = np.zeros((size**4, int(T / dt)))

    # backwards induction through time
    # this emulates the code for the adhoc model
    for index in range(2, int(T / dt) + 1):
        tau = (index - 1) * dt
        t = T - tau
        print(t)

        for i in range(size**4):
            Phi = grid_values.flatten[i]
            roots = update_Xs(Phi)
            new_phi_probs = p_new_ev_stay(roots, Phi, sigma, N)
            new_phi_probs = new_phi_probs / np.sum(new_phi_probs)
            V_stay = np.sum(new_phi_probs * V_base[:, -(index - 1)]) - rho * t

            V_base[i, -index] = np.amax((V_wait,
                                         decision_vals[i, 0], decision_vals[i, 1]))
            decisions[i, -index] = np.argmax((V_wait,
                                              decision_vals[i, 0], decision_vals[i, 1]))

    V_full = V_base.expand_dims(np.arrange(N), 3)
    decision_full = decisions.expand_dims(np.arange(N), 3)

    # backwards induction through items

    for index in range(1, N):
        V_item = V_full[:, :, N - index]
        decision_item = decision_full[:, :, N - index]
        for i in range(size**4):
            for j in range(int(T / dt)):
            V_switch =  # HERE CALCULATE THE EXPECPTED VALUE FOR SWITCHING, should be analogous except with the ev_prob_switch function defined above
            if V_switch > V_item[i, j]:
                # it switch is more valuable, update the value function and the decision
                V_item[i, j] = V_switch
                decision_item[i, j] = 3
        V_full[:, :, index] = V_item
        decision_full[:, :, index] = decision_item


if __name__ == '__main__':
    rho = 0.1
    reward = 1
    punishment = 0
    dt = 0.05
    sigma = 15
#    grid, times, decisions = main([dt, sigma, rho, reward, punishment])
