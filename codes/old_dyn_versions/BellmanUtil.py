from scipy.optimize import brentq
from scipy import norm
import numpy as np


class BellmanUtil:
    def __init__(self, T, t_w, size, d_t):
        self.T = T
        self.t_w = t_w
        self.self.size = self.size
        self.dt = d_t
        self.g_values = np.linspace(1e-4, 1 - 1e-4, self.size)
        self.N_array = np.array([8, 12, 16])

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

    def trans_probs(self, sigma, mu, space='g'):
        if space == 'g':
            dg = self.self.g_values[1] - self.g_values[0]
            prob_grid = np.zeros((self.size, self.size))
            for i, g_t in enumerate(self.g_values):
                updates = self.p_gtp1_gt(g_t, self.g_values, sigma, mu)
                updates = updates / (np.sum(updates) * dg)
                prob_grid[i, :] = updates

        if space == 'D':
            D_values = np.linspace(0, 1e2, 1e3)
            dD = D_values[1] - D_values[0]
            prob_grid = np.zeros((len(D_values), len(D_values)))
            for i, D_t in enumerate(D_values):
                updates = self.p_Dtp1_Dt(D_t, D_values, sigma, mu)
                updates = updates / (np.sum(updates) * dD)
                prob_grid[i, :] = updates

        return prob_grid


    def back_induct(self, reward, punishment, rho, sigma, mu, prob_grid, reward_scheme, t_dependent=False):
        dg = self.g_values[1] - self.g_values[0]

        # Define the reward array
        if reward_scheme == 'sym':
            R = np.array([(reward, punishment),   # (abs/abs,   abs/pres)V
                          (punishment, reward)])  # (pres/abs, pres/pres) in form decision / actual
        elif reward_scheme == 'epsilon_punish':
            R = np.array([(1, punishment),
                          (punishment, 1)])
        elif reward_scheme == 'asym_reward':
            R = np.array([(reward, punishment),
                          (punishment, 1)])
        else:
            raise Exception('Entered invalid reward_scheme')
        # Decision values are static for a given g_t and independent of t. We compute these
        # in advance
        # N x 2 matrix. First column is resp. abs, second is pres.
        decision_vals = np.zeros((self.size, 2))
        decision_vals[:, 1] = self.g_values * R[1, 1] + \
            (1 - self.g_values) * R[1, 0] - (self.t_w * rho)  # resp pres
        decision_vals[:, 0] = (1 - self.g_values) * R[0, 0] + \
            self.g_values * R[0, 1] - (self.back_induct_w * rho)  # resp abs

        # Create array to store V for each g_t at each t. N x (T / dt)
        V_full = np.zeros((self.size, int(self.T / self.dt)))
        # At large T we assume val of waiting is zero
        V_full[:, -1] = np.max(decision_vals, axis=1)
        # Corresponding array to store the identity of decisions made
        decisions = np.zeros((self.size, int(self.T / self.dt)))
        decisions[:, -1] = np.argmax(decision_vals, axis=1) + 1

        # Backwards induction
        for index in range(2, int(self.T / self.dt) + 1):
            for i in range(self.size):
                V_wait = np.sum(
                    prob_grid[:, i] * V_full[:, -(index - 1)]) * dg - (rho * self.dt)
                # Find the maximum value b/w waiting and two decision options. Store value and identity.
                V_full[i, -index] = np.amax((V_wait,
                                             decision_vals[i, 0], decision_vals[i, 1]))
                decisions[i, -index] = np.argmax((V_wait,
                                                  decision_vals[i, 0], decision_vals[i, 1]))
            if not t_dependent and index > 20:
                absolute_err = np.abs(V_full[:, -index] - V_full[:, -(index - 1)])
                converged = np.all(absolute_err[5:-5] < 1e-5)
                if converged:
                    dec_vec = decisions[:, -index]
                    # deal with degenerate cases in which there are no 1, 2s
                    dec_vec[0] = 1
                    dec_vec[-1] = 2
                    abs_threshold = np.amax(np.where(dec_vec == 1)[0])
                    pres_threshold = np.where(dec_vec == 2)[0][0]
                    dec_vec[0:abs_threshold] = 1
                    dec_vec[pres_threshold:len(dec_vec)] = 2
                    V_full = np.reshape(np.repeat(V_full[:, -index], V_full.shape[1]),
                                        (self.size, V_full.shape[1]))
                    decisions = np.reshape(np.repeat(dec_vec, decisions.shape[1]),
                                           (self.size, decisions.shape[1]))
                    break
        return V_full, decisions


    def solve_rho(self, reward, punishment, reward_scheme, sigma, mu, prob_grid):
        '''
        Root finding procedure to find rho given the constrain V(t=0)=0.
        This criteria comes from the invariance of policy with
        respect to linear shift in V
        '''
        def V_in_rho(log_rho):
            rho = np.exp(log_rho)
            values = self.back_induct(reward, punishment, rho, sigma, mu,
                                 prob_grid, reward_scheme)[0]
            return values[int(self.size / 2), 0]

        # when optimizing for reward this optimization should be accounted for in choosing bounds
        try:
            opt_log_rho = brentq(
                V_in_rho, -10 + np.log(reward), 10 + np.log(reward))
        except ValueError:
            raise Exception("defective bounds in rho finding procedure")

        return np.exp(opt_log_rho)
