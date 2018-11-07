import numpy as np


class FineGrained:
    def __init__(self, fine_sigma, fine_model='const', num_samples=10000, N_values=(8, 12, 16),
                 **kwargs):
        """
        Model which uses fine sigma to generate coarse mu and sigma for various N

        Parameters
        ----------
        fine_sigma : the overall sigma used for observations.
        fine_model : whether the noise in each observation within a single display scales with
            sqrt(N) (\'sqrt\') or
            is constant (\'const\').
        num_samples : the number of simulated displays.

        Outputs
        ----------
        self.coarse_stats : number of N x conditions (present, abs) x statistics (mu, sigma) array

        FineGrained uses an initial fine sigma to describe overall noise in observations. Then takes
        various values of N and uses N number of observations from a normal distribution
        to calculate a simulated single observation of a field of stimuli. Does this for present
        case, in which one observation has mean 1 and variance fine_sigma, and all others 0 mean and
        identical sigma, and absent case in which all observations are drawn from 0 mean
        distribution.

        num_samples number of observation sets are simulated. For each there is a decision variable
        d_map produced (likelihood ratio of present to absent) which takes into account the number
        of observations N. The distribution of these d_map for each given N is then used to compute
        the coarse mu and coarse sigma for present and absent at a given N. This is saved under
        the class variable coarse_stats.
        """

        if fine_model != 'const' and fine_model != 'sqrt':
            raise ValueError('Invalid entry for fine_model type in fine grained model')
        self.fine_sigma = fine_sigma
        self.fine_model = fine_model
        self.num_samples = num_samples
        N_values = (8, 12, 16)

        N_min = np.amin(N_values)
        pres_samples = np.zeros((num_samples, len(N_values)))
        abs_samples = np.zeros((num_samples, len(N_values)))
        coarse_stats = np.zeros((len(N_values), 2, 2))  # Store the 2 vals of sig and mu for each N

        for i in range(len(N_values)):
            N = N_values[i]
            # Determine the scaling of stimulus reliability by N
            if fine_model == 'const':
                sigma_N = fine_sigma
            if fine_model == 'sqrt':
                # If 'sqrt' scale sigma relative to smallest N in data
                sigma_N = fine_sigma * (np.sqrt(N / N_min))

            # For present and absent case calculate each decision variable from set of observations
            for j in range(num_samples):
                pres_samples[j, i] = self._d_map(N, self._sample_epsilon(1, N, sigma_N), sigma_N)
                abs_samples[j, i] = self._d_map(N, self._sample_epsilon(0, N, sigma_N), sigma_N)

            coarse_stats[i] = np.array([[np.mean(abs_samples[:, i]), np.sqrt(np.var(abs_samples[:, i]))],
                                        [np.mean(pres_samples[:, i]), np.sqrt(np.var(pres_samples[:, i]))]])

        self.pres_samples = pres_samples
        self.abs_samples = abs_samples
        self.coarse_stats = coarse_stats

    def _sample_epsilon(self, C, N, sigma):
        '''
        Returns an N dimensional vector representing draws of evidence
        from each item in the display in a given condition C
        '''
        epsilons = np.random.normal(0, sigma, N)
        if C == 1:
            epsilons[0] = np.random.normal(1, sigma)
        return epsilons

    def _d_map(self, N, epsilons, sigma_N):
        '''
        Computes the decisions variable d based on the log likelihood ratio
        between pres and abs conditions, used to bootstrap SDT-like distributions
        in get_coarse_stats function
        '''
        return -(1 / (2 * sigma_N**2)) + np.log((1 / N) * np.sum(np.exp(epsilons / sigma_N**2)))
