import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import multiprocessing as mulpro
from gauss_opt import bayesian_optimisation
import numpy as np
from finegr_model import FineGrained
from bellman_utilities import BellmanUtil

fine_sigma = 0.6
model = 'const'
num_samples = int(1e5)
N_array = np.array([8, 12, 16])
reward = 1
punishment = 0
reward_scheme = 'sym'

T = 10
d_t = 0.05
t_w = 0.5
size = 100
g_values = np.linspace(1e-4, 1 - 1e-4, size)
lapse = 1e-6

finemodel = FineGrained(fine_sigma, model, num_samples, N_array)
coarse_stats = finemodel.coarse_stats

mu, sigma = coarse_stats[1, :, :]

bellman_eqs = BellmanUtil(T, t_w, size, d_t)
prob_grid = bellman_eqs.trans_probs(sigma, mu)
rho = bellman_eqs.solve_rho(reward, punishment, reward_scheme, sigma, mu, prob_grid)

V_full, decisions = bellman_eqs.back_induct(reward, punishment, rho, sigma, mu, prob_grid,
                                            reward_scheme)
