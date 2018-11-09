import numpy as np
from scipy.stats import uniform, gaussian_kde, norm


class ObserverSim:
    def __init__(self, T, dt, g_values, sigma, mu, decisions, simulate=False, numsims=5000,
                 **kwargs):
        """
        Generates a pool of observers and, given decision bounds, produces simulated reaction times

        Parameters
        ----------
        T : Arbitrary choice of maximum time for a single trial
        dt : Length of a single timestep
        g_values : Values of g corresponding to rows of decision matrix
        sigma : Observation noise at current N
        mu : Mean of observations at current N
        decisions : Decision boundaries for the given conditions. Computed by BellmanUtil class
        numsims : Number of observers for each condition (absent, present) to simulate

        Outputs
        ----------
        self.dist_matrix : 2 x 2 matrix of distributions such that
            dist_matrix[i, j] is j decision for i truth. E.g. dist_matrix[0, 1] is the distribution
            of \'present\' responses when the target is absent
        self.rts_matrix : Response times categorized into the same matrix shape as dist_matrix.
            Each entry in the 2 x 2 matrix is a list of response times for that
            (condition, response) pair.

        """
        if simulate:
            presence_list = [0] * numsims + [1] * numsims
            observer_responses = []
            for C in presence_list:
                observer_responses.append(self.simulate_observer(T, dt, g_values, mu,
                                                                 sigma, decisions, C))

            response_info = np.array([(x[0], x[1]) for x in observer_responses])
            self.numsims = numsims
            self.rt_abs = response_info[:numsims]
            self.rt_pres = response_info[numsims:]
            self.dist_matrix, self.rts_matrix = self.get_kde_dist()
        else:
            self.decisions = decisions
            dg = g_values[1] - g_values[0]
            prob_grid_abs = self.fixed_trans_probs(sigma, mu, g_values, 0) * dg
            prob_grid_pres = self.fixed_trans_probs(sigma, mu, g_values, 1) * dg
            fractions_abs = self.fp_observer_diffusion(T, dt, sigma, mu, prob_grid_abs, g_values)
            fractions_pres = self.fp_observer_diffusion(T, dt, sigma, mu, prob_grid_pres, g_values)
            self.fractions = (fractions_abs, fractions_pres)

    def g_to_D(self, g_t):
        return np.log(g_t / (1 - g_t))

    def D_to_g(self, D_t):
        return np.exp(D_t) / (1 + np.exp(D_t))

    def p_gtp1_C(self, C, g_t, g_tp1, sigma, mu):
        D_t = self.g_to_D(g_t)
        D_tp1 = self.g_to_D(g_tp1)
        jacobian_factor = 1 / (self.D_to_g(D_t) * (1 - self.D_to_g(D_t)))

        draw = norm.pdf(D_tp1, D_t + mu[C], sigma[C])

        return jacobian_factor * draw

    def fixed_trans_probs(self, sigma, mu, g_values, C):
        dg = g_values[1] - g_values[0]
        size = g_values.shape[0]
        prob_grid = np.zeros((size, size))
        for i, g_t in enumerate(g_values):
            updates = self.p_gtp1_C(C, g_t, g_values, sigma, mu)
            updates = updates / (np.sum(updates) * dg)
            prob_grid[i, :] = updates

        return prob_grid

    def simulate_observer(self, T, dt, g_values, mu, sigma, decisions, C):
        if C != 1 and C != 0:
            raise ValueError('condition C must be 0 (abs) or 1 (pres)')

        dec_vec = decisions[:, 0]
        abs_bound = g_values[np.amax(np.where(dec_vec == 1)[0])]
        pres_bound = g_values[np.where(dec_vec == 2)[0][0]]

        D_t = 0
        t = 0

        g_trajectory = np.ones(int(T / dt)) * 0.5
        D_trajectory = np.zeros(int(T / dt))

        while t < T:
            D_t = D_t + np.random.normal(mu[C] * dt, sigma[C] * dt)

            g_t = self.D_to_g(D_t)
            D_trajectory[int(t / dt)] = D_t
            g_trajectory[int(t / dt)] = g_t
            t += dt

            if g_t < abs_bound:
                return (0, t, g_trajectory, D_trajectory)

            if g_t > pres_bound:
                return (1, t, g_trajectory, D_trajectory)

        # Return NaN if end of trial reached with no decision
        return (np.NaN, T, g_trajectory, D_trajectory)

    def fp_observer_diffusion(self, T, dt, sigma, mu, prob_grid, g_values):
        observerarr = np.zeros((g_values.shape[0], int(T / dt)))
        observerarr[int(observerarr.shape[0] / 2), 0] = 1

        fractions = np.zeros((3, int(T / dt)))
        fractions[2, 0] = 1
        for i in range(1, int(T / dt)):
            currstep = np.zeros(g_values.shape[0])
            for j, g_t in enumerate(g_values):
                currstep = currstep + prob_grid[j, :] * observerarr[j, i-1]
            lowerbound = np.amax(np.where(self.decisions[:, i] == 1)[0]) + 1
            upperbound = np.where(self.decisions[:, i] == 2)[0][0]
            observerarr[lowerbound:upperbound, i] = currstep[lowerbound:upperbound]
            fractions[2, i] = np.sum(currstep[lowerbound:upperbound])
            fractions[0, i] = np.sum(currstep[:lowerbound])
            fractions[1, i] = np.sum(currstep[upperbound:])
        return fractions

    def get_kde_dist(self):
        dists = []
        rts_matrix = []
        perturb = np.random.normal(0, 0.01)
        sim_rt = [self.rt_abs, self.rt_pres]
        for i in range(2):
            cur_rt = sim_rt[i]
            for j in range(2):
                if not np.any(cur_rt[:, 0] == j):
                    # case where there are none of the responses given in the model simulation
                    dists.append(uniform)
                    rts_matrix.append([])
                else:
                    i_j_sim_rt = cur_rt[cur_rt[:, 0] == j, 1]
                    # if they are all the same or of size 1, perturb to allow kde
                    if np.var(i_j_sim_rt) == 0 or i_j_sim_rt.size == 1:
                        i_j_sim_rt = np.append(i_j_sim_rt, i_j_sim_rt[0] + perturb)

                    rts_matrix.append([i_j_sim_rt])
                    dists.append(gaussian_kde(i_j_sim_rt, bw_method=0.1))

        return np.reshape(dists, (2, 2)), np.reshape(rts_matrix, (2, 2))
