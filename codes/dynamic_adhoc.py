'''
March 2018

Implementation of the ad-hoc model for the dynamic version of Strongway's
visual search task
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import seaborn as sbn

T = 5
t_w = 0.5
dt = 0.005
size = 1000
sigma = 1
rho = 1
g_values = np.linspace(1e-3, 1 - 1e-3, size)


def f(x, g_t):
    ''' x_(t + 1) is x
    '''
    p_given_true = (g_t * np.exp(- (x-1)**2 / 2))
    return p_given_true / (p_given_true + (1 - g_t) * np.exp(- x**2 / 2))


def df_dx(x, g_t):
    '''x_(t + 1) is x
    derivative of the f() which we seek to use a root-finding
    procedure on
    this is ugly'''
    numerator = (g_t - 1) * g_t * np.exp(x + 0.5)
    denominator = (np.sqrt(np.e) * (g_t - 1) - g_t * np.exp(x))**2
    return numerator / denominator


def p_new_ev(x, g_t):
    ''' The probability of a given observation x_(t+1) given our current belief
    g_t'''
    p_pres = np.exp(- (x - 1)**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)
    p_abs = np.exp(- x**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)
    return p_pres * g_t + p_abs * (1 - g_t)


# First we find the roots of f(x) for all values of g_t and g_(t+1)
rootgrid = np.zeros((size, size))
for i in range(size):
    g_t = g_values[i]
    for j in range(size):
        g_tp1 = g_values[j]
        rootgrid[i, j] = brentq(lambda x: g_tp1 - f(x, g_t),-20, 20)


R = np.array([(10, 0),   # (abs/abs,   abs/pres)
              (0, 10)])  # (pres/abs, pres/pres) in form decision / actual

decision_vals = np.zeros((size, 2))
decision_vals[:, 1] = g_values * R[1, 1] + (1 - g_values) * R[1, 0]  # respond present
decision_vals[:, 0] = (1 - g_values) * R[0, 0] + g_values * R[0, 1]  # respond absent

V_full = np.zeros((size, int(T / dt)))
V_full[:, -1] = np.max(decision_vals, axis=1)

decisions = np.zeros((size, int(T / dt)))

for index in range(2, int(T / dt) + 1):
    tau = (index - 1) * dt
    t = T - tau
    print(t)

    for i in range(size):
        g_t = g_values[i]
        roots = rootgrid[i, :]
        new_g_probs = p_new_ev(roots, g_t)
        new_g_probs = new_g_probs / np.sum(new_g_probs)
        V_wait = np.sum(new_g_probs * V_full[:, -(index - 1)]) - rho * t
        V_full[i, -index] = np.amax((V_wait, decision_vals[i, 0], decision_vals[i, 1]))
        decisions[i, -index] = np.argmax((V_wait, decision_vals[i, 0], decision_vals[i, 1]))
