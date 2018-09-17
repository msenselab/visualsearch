import numpy as np
import pandas as pd
from pathlib import Path


class DataLikelihoods:
    def __init__(self, subject_num, **kwargs):
        """
        Class to load in subject data and then compute likelihood of data given simulation

        Parameters
        ----------
        subject_num : Number of the subject as found in the CSV

        Outputs
        ----------
        self.likelihood : Likelihood of data. Computed incrementally. Requires calling
            self.increment_likelihood for each N and associated sim data dists and RTs.

        """
        datapath = Path('../data/exp1.csv')
        exp1 = pd.read_csv(datapath, index_col=None)
        exp1.rename(columns={'sub': 'subno'}, inplace=True)

        self.sub_data = exp1.query('subno == {} & dyn == \'Dynamic\''.format(subject_num))
        self.likelihood = 0.

    def increment_likelihood(self, dist_matrix, rts_matrix, N, reward, lapse, **kwargs):
        """
        Increments internal likelihood with the likelihood of given dists and rts

        Parameters
        ----------
        dist_matrix : 2 x 2 matrix of simulated data distributions (KDE) produced by obs class
        rts_matrix : 2 x 2 matrix of simulated reaction times (each entry a list of RTs) also
            produced by obs class
        N : N for given set of sim data
        reward : curr reward from optimizer
        lapse : lapse rate hyperparameter

        Outputs
        ----------
        Increment to self.likelihood

        """
        N_data = self.sub_data.query('setsize == {}'.format(N))
        temp = np.mean(np.array(N_data['rt']))

        num_abs_0 = len(rts_matrix[0, 0])
        num_pres_1 = len(rts_matrix[1, 1])
        num_abs_1 = len(rts_matrix[0, 1])
        num_pres_0 = len(rts_matrix[1, 0])

        total_pres = num_pres_0 + num_pres_1
        total_abs = num_abs_0 + num_abs_1

        pres_rts_0 = N_data.query('resp == 2 & target == \'Present\'').rt.values
        pres_rts_1 = N_data.query('resp == 1 & target == \'Present\'').rt.values

        abs_rts_0 = N_data.query('resp == 2 & target == \'Absent\'').rt.values
        abs_rts_1 = N_data.query('resp == 1 & target == \'Absent\'').rt.values

        with np.errstate(divide='ignore'):
            # frac_pres_inc = len(pres_rts_0) / (len(pres_rts_0) + len(pres_rts_1))
            # frac_pres_corr = len(pres_rts_1) / (len(pres_rts_0) + len(pres_rts_1))
            log_like_pres = np.concatenate((np.log(num_pres_0 / total_pres) +
                                            np.log(dist_matrix[1, 0].pdf(pres_rts_0)),
                                            np.log(num_pres_1 / total_pres) +
                                            np.log(dist_matrix[1, 1].pdf(pres_rts_1))))

            # frac_abs_inc = len(abs_rts_1) / (len(abs_rts_0) + len(abs_rts_1))
            # frac_abs_corr = len(abs_rts_0) / (len(abs_rts_0) + len(abs_rts_1))
            log_like_abs = np.concatenate((np.log(num_abs_0 / total_abs) +
                                           np.log(dist_matrix[0, 0].pdf(abs_rts_0)),
                                           np.log(num_abs_1 / total_abs) +
                                           np.log(dist_matrix[0, 1].pdf(abs_rts_1))))

        log_like_all = np.concatenate((log_like_pres, log_like_abs))

        likelihood_pertrial = (1 - lapse) * np.exp(log_like_all) + \
            (lapse / 2) * np.exp(-reward / temp)

        self.likelihood += -np.sum(np.log(likelihood_pertrial))
