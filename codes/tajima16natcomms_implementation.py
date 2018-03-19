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
sigma_z = 10
sigma_x = 4
mean_z = 0
rho = 5
c = 8


def sigma_t(t, sigma, sigma_z):
    '''Sigma(t)'''
    return (sigma_z**-1 + t * sigma**-1)**-1


def z_next_prior(z, z_next, sigma_z):
    '''P(z(t + dt) | z(t))'''
    return np.exp(- (z - z_next)**2 / (2 * sigma_z)) / np.sqrt(2 * np.pi * sigma_z)


def V_d(r, t):
    '''value of making a decision. Note that as usual r needs to be a 2x1 vector.'''
    return np.max(r) - t * rho * t_w


if __name__ == '__main__':
    size = 100  # Number of points within range of prior r to test
    r_range = np.linspace(-10, 10, size)  # Vector of values of r
    r_grid_base = r_range * np.ones((size, size))  # Above row vector repeated in N=size rows

    # N x N x 2 array in which i,j,0 are the values of r_1 repeated row-wise and i,j,0 are
    # the values of r_2 repeated column-wise. The max across the last dimension at any i,j
    # will be the max of r_1, r_2 for any given pair of values
    base_1 = r_grid_base.reshape(size, size, 1)
    base_2 = r_grid_base.T.reshape(size, size, 1)
    r_grid = np.concatenate((r_grid_base.reshape(size, size, 1),
                             r_grid_base.T.reshape(size, size, 1)),
                            axis=2)

    # Construct an N x N x (T / dt) cube, in which each N x N array along the
    # last axis is an expected value for deciding
    V_end = np.max(r_grid, axis=2)
    V_cube = np.zeros((size, size, int(T / dt)))
    V_cube[:, :, -1] = V_end
    # Create a similar cube in which each of the elements is the decision (i.e. argmax)
    # of the corresponding V_cube element
    decision_cube = np.ones_like(V_cube)
    decision_cube[:, :, -1] = np.argmax(r_grid, axis=2) + 1

    # Backwards induction
    for index in range(2, int(T / dt) + 1):
        # Tau is the delta T backwards from T
        tau = (index - 1) * dt
        t = T - tau
        print(index)

        # Compute the variance as a function of prior variance and observation uncertainty sigma_x
        sig_t = sigma_t(t, sigma_x, sigma_z)

        # For each r_1, r_2 pair compute the decision value and expected V at the next time step
        for i, j in it.product(range(size), range(size)):
            r = r_grid[i, j, :]  # both values of r
            V_d_r = V_d(r, t)  # Value for decision

            # Outer product of the gaussian transition probability distributions for each r produces
            # the transition probability distribution for a given r pair
            transition_probs = np.zeros_like(V_end)
            transition_probs = np.outer(z_next_prior(r[1], r_range, sig_t),
                                        z_next_prior(r[0], r_range, sig_t))

            # Weight the V at next t by the transition probabilities
            weighted_vals = V_cube[:, :, -(index - 1)] * transition_probs
            # Use mean of above minus opportunity cost in addition to V_d_r to find max V at curr r
            V_cube[i, j, -index] = np.max((np.mean(weighted_vals) - (c + rho) * t_w * t, V_d_r))
            # Store decision identity in cube
            decision_cube[i, j, -index] = np.argmax((np.mean(weighted_vals) - (c + rho) * t_w * t,
                                                     V_d_r))
            if decision_cube[i, j, -index] == 1:
                decision_cube[i, j, -index] = np.argmax(r) + 1

    # Little movie of decision boundary change
    fig = plt.figure(figsize=(6, 9))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    def anim_update(i):
        ax1.clear()
        ax1.pcolor(r_range, r_range, V_cube[:, :, 2 * i])
        ax1.set_xlabel('r_1')
        ax1.set_ylabel('r_2')

        ax2.clear()
        ax2.pcolor(r_range, r_range, decision_cube[:, :, 2 * i])
        ax2.set_xlabel('r_1')
        ax2.set_ylabel('r_2')
        plt.draw()
        return

    anim = FuncAnimation(fig, anim_update, frames=int(T / dt / 2))
    anim.save('reward_task_value_evolution.mp4', fps=60)
