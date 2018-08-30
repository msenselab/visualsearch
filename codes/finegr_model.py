from scipy.stats import norm
import numpy as np


class FineGrained:
    def __init__(self, fine_sigma, model, num_samples):
        ''' Instance containing necessary methods and functions to sample decision n_variables
        d from the fine-grained model of search.
            options for model are:
            'const' : constant fine_sigma
            'sqrt'  : scale sigma by sqrt of N'''

        if model != 'const' and model != 'sqrt':
            raise Exception('Invalid entry for model type in fine grained model')
        self.fine_sigma = fine_sigma
        self.model = model
        self.num_samples = num_samples
        N_array = np.array([8, 12, 16])


        N_min = np.amin(N_array)
        pres_samples = np.zeros((num_samples, N_array.shape[0]))
        abs_samples = np.zeros((num_samples, N_array.shape[0]))
        stats = np.zeros((len(N_array), 2, 2))

        for i in range(len(N_array)):
            N = N_array[i]
            # Determine the scaling of stimulus reliability by N
            if model == 'const':
                sigma_N = fine_sigma
            if model == 'sqrt':
                # If 'sqrt' scale sigma relative to smallest N in data
                sigma_N = fine_sigma * (np.sqrt(N / N_min))
            for j in range(num_samples):
                pres_samples[j, i] = self.d_map(
                    N, self.sample_epsilon(1, N, sigma_N), sigma_N)
                abs_samples[j, i] = self.d_map(
                    N, self.sample_epsilon(0, N, sigma_N), sigma_N)

            stats[i] = np.array([[np.mean(abs_samples), np.sqrt(np.var(abs_samples))],
                                 [np.mean(pres_samples), np.sqrt(np.var(pres_samples))]])

        self.pres_samples = pres_samples
        self.abs_samples = abs_samples
        self.coarse_stats = stats

    def sample_epsilon(self, C, N, sigma):
        '''
        Returns an N dimensional vector representing draws of evidence
        from each item in the display in a given condition C
        '''
        epsilons = norm.rvs(0, sigma, N)
        if C == 1:
            epsilons[0] = norm.rvs(1, sigma)
        return epsilons

    def d_map(self, N, epsilons, sigma_N):
        '''
        Computes the decisions variable d based on the log likelihood ratio
        between pres and abs conditions, used to bootstrap SDT-like distributions
        in get_coarse_stats function
        '''
        return -(1 / (2 * sigma_N**2)) + np.log(1 / N) + \
            np.log(np.sum(np.exp(epsilons / sigma_N**2)))
