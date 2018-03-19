'''
Feb 14 2018

Implementation of uncorrelated prior and likelihood version of the task in
Tajima et. al. 2016

'''


from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
import itertools as it
from matplotlib.animation import FuncAnimation

T = 3
t_w = 0.5
dt = 0.005
sigma_z = 15
sigma_x = 4
mean_z = 0
rho = 5
c = 8


def sigma_t(t, sigma, sigma_z):
    '''Sigma(t)'''
    return (sigma_z**-1 + t * sigma**-1)**-1


def posterior_dist(val, x, z_prior, sigma, sigma_z, t):
    ''' Posterior distribution as defined in the methods section, using a prior covariance sigma_z
    and a true covariance sigma. Note that when the experienced reward _r_ is equal to the
    true reward, the mean estimated option reward is the mean of this distribution'''
    sig_t = sigma_t(t, sigma, sigma_z)
    mean = sig_t * (sigma_z**-1 * z_prior + sigma ** -1 * x)
    return np.exp(-(val - mean)**2 / (2 * sig_t)) / np.sqrt(2 * np.pi * sig_t)


def r_hat_next(val, t, r_hat, sigma, sigma_z):
    mean = r_hat
    sigma_next = sigma_t(t + dt, sigma, sigma_z)
    var = sigma_next * (sigma**-1) * sigma_next * dt
    return np.exp(-(val - mean)**2 / (2 * var)) / np.sqrt(2 * np.pi * var)


def z_next_prior(z, z_next, sigma_z):
    '''P(z(t + dt) | z(t))'''
    return np.exp(- (z - z_next)**2 / (2 * sigma_z)) / np.sqrt(2 * np.pi * sigma_z)


def V_d(r):
    '''value of making a decision. Note that as usual r needs to be a 2x1 vector.'''
    return np.max(r) - rho * t_w


def V(r, V_next_exp):
    '''Needs to be passed the future value of V'''
    return np.max((V_d(r), V_next_exp - (c + rho) * t_w))


if __name__ == '__main__':
    size = 40
    r_range = np.linspace(-10, 10, size)
    r_grid_base = r_range * np.ones((size, size))
    base_1 = r_grid_base.reshape(size, size, 1)
    base_2 = r_grid_base.T.reshape(size, size, 1)
    r_grid = np.concatenate((r_grid_base.reshape(size, size, 1),
                             r_grid_base.T.reshape(size, size, 1)),
                            axis=2)
    V_init = np.max(r_grid, axis=2) - rho * t_w

    V_cube = np.zeros((size, size, int(T / dt)))
    V_cube[:, :, -1] = V_init
    decision_cube = np.ones_like(V_cube)
    decision_cube[:, :, -1] = np.argmax(r_grid, axis=2)
    for index in range(2, int(T / dt) + 1):
        tau = index * dt
        t = T - tau
        print(index)
        sig_t = sigma_t(t, sigma_x, sigma_z)
        maxes = np.zeros((size, size))
        for i, j in it.product(range(size), range(size)):
            r = r_grid[i, j, :]
            V_d_r = V_d(r)
            transition_probs = np.zeros_like(V_init)
            transition_probs = np.outer(z_next_prior(r[1], r_range, sig_t),
                                        z_next_prior(r[0], r_range, sig_t))
            weighted_vals = V_cube[:, :, -(index - 1)] * transition_probs
            V_cube[i, j, -index] = np.max((np.mean(weighted_vals) - (c + rho) * t_w, V_d_r))
            decision_cube[i, j, -
                          index] = np.argmax((np.mean(weighted_vals) - (c + rho) * t_w, V_d_r))
            if decision_cube[i, j, -index] == 1:
                decision_cube[i, j, -index] = np.argmax(r) + 1

    fig = plt.figure()
    ax = fig.add_subplot(111)

    def anim_update(i):
        ax.clear()
        ax.pcolor(decision_cube[:, :, 2 * i])
        plt.draw()
        return

    anim = FuncAnimation(fig, anim_update, frames=int(T / dt / 2))
    anim.save('reward_task_value_evolution.mp4', fps=60)