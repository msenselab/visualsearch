from scipy.stats import norm, gaussian_kde, uniform
import numpy as np
import itertools as it


class Observer:
    def __init__(self, T, dt, t_w, size, g_values, lapse):
        self.T = T
        self.dt = dt
        self.t_w = t_w
        self.g_values = g_values
        self.lapse = lapse

    def g_to_D(self, g_t):
        return np.log(g_t / (1 - g_t))

    def D_to_g(self, D_t):
        return np.exp(D_t) / (1 + np.exp(D_t))

    def p_gtp1_gt(self, g_t, g_tp1, sigma, mu):
        D_t = self.g_to_D(g_t)
        D_tp1 = self.g_to_D(g_tp1)
        jacobian_factor = 1 / (self.D_to_g(D_t) * (1 - self.D_to_g(D_t)))

        pres_draw = g_t * norm.pdf(D_tp1, D_t + mu[1], sigma[1])
        abs_draw = (1 - g_t) * norm.pdf(D_tp1, D_t + mu[0], sigma[0])

        return jacobian_factor * (pres_draw + abs_draw)

    def p_Dtp1_Dt(self, D_t, D_tp1, sigma, mu):
        g_t = self.D_to_g(D_t)

        pres_draw = g_t * norm.pdf(D_tp1, D_t + mu[1], sigma[1])
        abs_draw = (1 - g_t) * norm.pdf(D_tp1, D_t + mu[0], sigma[0])

        return pres_draw + abs_draw

    def simulate_observer(self, arglist):
        C, decisions, sigma, mu, dt = arglist

        dec_vec = decisions[:, 0]
        abs_bound = self.g_values[np.amax(np.where(dec_vec == 1)[0])]
        pres_bound = self.g_values[np.where(dec_vec == 2)[0][0]]

        D_t = 0
        t = 0

        g_trajectory = np.ones(int(self.T / self.dt)) * 0.5
        D_trajectory = np.zeros(int(self.T / self.dt))

        while t < self.T:
            D_t = D_t + norm.rvs(mu[C] * dt, sigma[C] * dt)

            # D_t = norm.rvs(D_t + mu[C], sigma[C]) * dt

            g_t = self.D_to_g(D_t)
            D_trajectory[int(t / dt)] = D_t
            g_trajectory[int(t / dt)] = g_t
            t += dt

            if g_t < abs_bound:
                return (0, t + 0.1, g_trajectory, D_trajectory)

            if g_t > pres_bound:
                return (1, t + 0.1, g_trajectory, D_trajectory)

        return (np.NaN, self.T, g_trajectory, D_trajectory)

    def get_rt(self, sigma, mu, decisions, numsims=5000, parallelize=False):
        C_vals = [0] * numsims
        C_vals.extend([1] * numsims)
        arglists = it.product(C_vals, [decisions], [sigma], [mu], [self.dt])
        if not parallelize:
            observer_outputs = []
            for arglist in arglists:
                observer_outputs.append(self.simulate_observer(arglist))
        elif parallelize:
            raise ValueError(
                'Parallelization is not supported for class based observers')

        response_info = np.array([(x[0], x[1]) for x in observer_outputs])
        abs_info = response_info[:numsims, :]
        pres_info = response_info[numsims:, :]

        return (abs_info, pres_info)

    def get_kde_dist(self, sim_rt, plot=False, ax=None):
        # 2x2 matrix of distributions, i (row) is the underlying condition C
        # and j (column) is the response
        dist = []
        sorted_rts = []
        perturb = norm.rvs(0, 0.01)
        for i in range(2):
            cur_rt = sim_rt[i]
            for j in range(2):
                if not np.any(cur_rt[:, 0] == j):
                    # case where there are none of the responses given in the model simulation
                    dist.append(uniform)
                    sorted_rts.append([])
                else:
                    i_j_sim_rt_marked = np.array(
                        cur_rt[np.where(cur_rt[:, 0] == j)[0]])
                    i_j_sim_rt = i_j_sim_rt_marked[:, 1]
                    # if they are all the same or of size 1, perturb to allow kde
                    if np.var(i_j_sim_rt) == 0 or i_j_sim_rt.size == 1:
                        # if they are all the same, perturb to allow kde
                        i_j_sim_rt = np.append(i_j_sim_rt, i_j_sim_rt[0] + perturb)
                    sorted_rts.append([i_j_sim_rt])
                    dist.append(gaussian_kde(i_j_sim_rt, bw_method=0.1))

        return np.reshape(dist, (2, 2)), np.reshape(sorted_rts, (2, 2))
