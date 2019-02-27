import numpy as np
import csv
from copy import deepcopy
from bellman_utilities import BellmanUtil
from scipy.stats import norm


class DataGen:
    def __init__(self, model_params, tot_samples=70):
        curr_params = deepcopy(model_params)
        bellutil = BellmanUtil(**curr_params)
        curr_params['rho'] = bellutil.rho
        curr_params['decisions'] = bellutil.decisions
        decisions = bellutil.decisions

        for i, column in enumerate(decisions.T):
            try:
                upperbound = (column == 2).nonzero()[0][0]
                lowerbound = (column == 1).nonzero()[0][-1]
            except IndexError:
                raise(ValueError, 'Non-existant bounds at some timestep')
            column[upperbound:] = 2
            column[:lowerbound] = 1
            decisions[:, i] = column

        condN = tot_samples // 2
        dt = model_params['dt']
        T = model_params['T']
        t_d = model_params['t_delay']
        t_max = model_params['t_max'] - t_d
        maxind = int(t_max / dt) - 1
        t_values = np.arange(0, T, dt)
        g_values = model_params['g_values']

        ev_values = np.zeros((2, tot_samples // 2, t_values.shape[0]))
        ev_values[0, :, :] = np.random.normal(loc=0, scale=model_params['sigma'][0],
                                              size=ev_values.shape[1:])
        ev_values[1, :, :] = np.random.normal(loc=1, scale=model_params['sigma'][1],
                                              size=ev_values.shape[1:])

        g_traces = np.zeros_like(ev_values)
        for C in (0, 1):
            for sample in range(condN):
                for samplen in range(t_values.shape[0]):
                    g_traces[C, sample, samplen] = self.g_t(ev_values[C, sample, :samplen + 1],
                                                            model_params['mu'],
                                                            model_params['sigma'])
        binned_traces = np.digitize(g_traces, g_values, right=True)
        binned_traces[binned_traces == g_values.shape[0]] = g_values.shape[0] - 1

        response_times = np.zeros(ev_values.shape[:-1])
        response_idents = np.zeros(ev_values.shape[:-1])
        for C in (0, 1):
            for sample in range(condN):
                i = 0
                while i <= maxind:
                    currdec = decisions[binned_traces[C, sample, i], i]
                    if currdec == 1:
                        response_times[C, sample] = t_values[i] + t_d
                        response_idents[C, sample] = 0
                        break
                    elif currdec == 2:
                        response_times[C, sample] = t_values[i] + t_d
                        response_idents[C, sample] = 1
                        break
                    elif (currdec == 0) and (i == maxind):
                        response_times[C, sample] = t_max + t_d
                        response_idents[C, sample] = 2
                    i += 1

        self.response_times = response_times
        self.response_idents = response_idents
        self.model_params = curr_params

    def g_t(self, x, mu, sigma, prior=0.5):
        presprobs = norm.pdf(x, loc=mu[1], scale=sigma[1])
        absprobs = norm.pdf(x, loc=mu[0], scale=sigma[0])
        denom = np.product(presprobs) * prior + np.product(absprobs) * prior
        return np.product(presprobs) * prior / denom

    def save_csv(self, savepath):
        with open(savepath, 'r') as fr:
            existing = fr.read(6) == 'target'

        with open(savepath, 'a') as fw:
            writer = csv.writer(fw)
            if not existing:
                writer.writerow(['target', 'setsize', 'dyn', 'resp', 'rt', 'sub', 'exp', 'correct'])
            curr_N = self.model_params['N']
            abs_resp = zip(self.response_idents[0, :], self.response_times[0, :])
            pres_resp = zip(self.response_idents[1, :], self.response_times[1, :])
            # First write all responses for target absent simulations
            for response, rt in abs_resp:
                correct = response == 0
                if response == 0:
                    adjusted_response = 2
                elif response == 1:
                    adjusted_response = 1
                elif response == 2:
                    adjusted_response = -1
                writer.writerow(['Absent', curr_N, 'Dynamic',
                                 adjusted_response, "{:.7f}".format(rt), 666, 1, correct])
            # Then write all responses for target present sims
            for response, rt in pres_resp:
                correct = response == 1
                if response == 0:
                    adjusted_response = 2
                elif response == 1:
                    adjusted_response = 1
                elif response == 2:
                    adjusted_response = -1
                writer.writerow(['Present', curr_N, 'Dynamic',
                                 adjusted_response, "{:.7f}".format(rt), 666, 1, correct])
