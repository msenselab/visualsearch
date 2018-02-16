'''
Feb 14 2018

Implementation of uncorrelated prior and likelihood version of the task in
Tajima et. al. 2016

'''


from __future__ import division
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import seaborn as sbn
import pickle
import sys, time, glob, datetime, os, re

# Means of the reward distributions for the two possible options z
z = (10, 10)
sigma = (0.1, 0.1)

dt = 0.01


def posterior_dist(val, x, z_prior, sigma, sigma_z, t):
	''' Posterior distribution as defined in the methods section, using a prior covariance sigma_z
	and a true covariance sigma. Note that when the experienced reward _r_ is equal to the
	true reward, the mean estimated option reward is the mean of this distribution'''
	sigma_t = (sigma**-1 + t * sigma_z**-1)**-1
	mean = sigma_t * (sigma_z**-1 * z_prior + sigma ** -1 * x)
	return np.exp(- (val - mean)**2 / (2 * sigma_t)) / np.sqrt(2 * np.pi * sigma_t)


def r_hat_next(val, r_hat, sigma_next, sigma):
	mean = r_hat
	var = sigma_next * (sigma**-1) * sigma_next * dt
	return np.exp( - (val - mean)**2 / (2 * var)) / np.sqrt(2 * np.pi * var)


