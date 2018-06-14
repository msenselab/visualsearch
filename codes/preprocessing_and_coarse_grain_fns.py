'''
June 2018

Functions to take in the .csv files containing subject data and return them. Also will
provide the function to compute coarse-grained statistics.

'''
import numpy as np
import pandas as pd
from pathlib import Path


def get_subject_data(subno):
    ''' Returns numpy array for the subject with columns representing target presence, setsize, and
    response time'''
    # Returns a path object that works as a string for most functions
    if subno not in range(1, 12):
        raise ValueError('Subject number not present in data')
    datapath = Path("../data/exp1.csv")
    exp1 = pd.read_csv(datapath, index_col=None)  # read data
    # .sub is a keyword, change it
    exp1.rename(columns={'sub': 'subno'}, inplace=True)

    if subno not in set(exp1.subno):
        raise ValueError('Subject number not present in data')

    subject_data = exp1.query('subno == {} & dyn == \'Dynamic\' & correct == 1'.format(subno))
    return subject_data.values


def d_map(N, epsilons, fine_sigma):
    return -(1 / (2 * fine_sigma**2)) + np.log(1 / N) + \
        np.log(np.sum(np.exp(epsilons / fine_sigma**2)))


def sample_epsilon(C, N, fine_sigma):
    epsilons = np.random.normal(0, fine_sigma, N)
    if C == 1:
        epsilons[0] = np.random.normal(1, fine_sigma)
    return epsilons


def get_coarse_stats(fine_sigma, num_samples):
    '''
    returns a 2x2 matrix, col 1 is abs stats, col 2 pres stats
    row 1 is the mean and row 2 is the sd
    '''
    N_array = [8, 12, 16]

    stats = np.zeros((len(N_array), 2, 2))
    for i in range(len(N_array)):
        N = N_array[i]
        pres_samples = np.zeros(num_samples)
        abs_samples = np.zeros(num_samples)
        for j in range(num_samples):
            pres_samples[j] = d_map(N, sample_epsilon(1, N, fine_sigma), fine_sigma)
            abs_samples[j] = d_map(N, sample_epsilon(0, N, fine_sigma), fine_sigma)

        stats[i] = np.array([[np.mean(abs_samples), np.sqrt(np.var(abs_samples))],
                             [np.mean(pres_samples), np.sqrt(np.var(pres_samples))]])
    return stats
