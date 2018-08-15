'''
August 2018

Class based implementation of the dynamic model
'''

import numpy as np
from finegr_model import FineGrained
from bellman_utilities import BellmanUtil
from observer_sim import Observer

fine_sigma = 0.6
model = 'const'
num_samples = int(1e5)
N_array = np.array([8, 12, 16])
reward = 1
punishment = 0
reward_scheme = 'sym'

T = 10
dt = 0.05
t_w = 0.5
size = 100
g_values = np.linspace(1e-4, 1 - 1e-4, size)
lapse = 1e-6

finemodel = FineGrained(fine_sigma, model, num_samples, N_array)
coarse_stats = finemodel.coarse_stats

mu, sigma = coarse_stats[1, :, :]

bellman_eqs = BellmanUtil(T, t_w, size, dt)
prob_grid = bellman_eqs.trans_probs(sigma, mu)
rho = bellman_eqs.solve_rho(reward, punishment, reward_scheme, sigma, mu, prob_grid)

V_full, decisions = bellman_eqs.back_induct(reward, punishment, rho, sigma, mu, prob_grid,
                                            reward_scheme)

obs = Observer(T, dt, t_w, size, g_values, lapse)
sim_rt = obs.get_rt(sigma, mu, decisions)
dist_matrix, sorted_rt = obs.get_kde_dist(sim_rt)
