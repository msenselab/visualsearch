import numpy as np
from copy import deepcopy
import warnings
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.offline as plotly
import plotly.graph_objs as go
from fine_grain_model import FineGrained
from bellman_utilities import BellmanUtil
from observers import ObserverSim
from data_and_likelihood import DataLikelihoods


class OptAnalysis:
    def __init__(self, subject_num, T, dt, t_w, t_max, size, lapse, mu, N_values, g_values,
                 experiment, N, reward_scheme, tested_params, likelihoods_returned, t_delay,
                 opt_regime=None, **kwargs):
        model_params = {'T': 10,
                        'dt': 0.05,
                        't_w': 0.5,
                        't_delay': 0.2,
                        't_max': 5.,
                        'size': size,
                        'lapse': 1e-6,
                        'mu': np.array((0, 1)),
                        'N': int(N),
                        'N_values': (8, 12, 16),
                        'g_values': np.linspace(1e-4, 1 - 1e-4, size),
                        'subject_num': int(subject_num),
                        'reward_scheme': 'asym_reward',
                        'experiment': experiment}

        curr_params = deepcopy(model_params)
        self.opt_params = tested_params[np.argmin(likelihoods_returned)]
        opt_likelihood = np.amin(likelihoods_returned)
        curr_params['sigma'] = np.exp(self.opt_params[:2])
        curr_params['reward'] = np.exp(self.opt_params[2])
        curr_params['punishment'] = -np.exp(self.opt_params[3])
        curr_params['alpha'] = np.exp(self.opt_params[4])

        # If we don't have the decisions, rho, and reaction times for opt params, compute them
        if not opt_regime:
            bellutil = BellmanUtil(**curr_params)
            rho = bellutil.rho
            decisions = bellutil.decisions

            obs = ObserverSim(decisions=decisions, **curr_params)
            fractions = obs.fractions
            opt_regime = {'rho': rho, 'decisions': decisions,
                          'fractions': fractions}

        curr_params['opt_regime'] = opt_regime
        curr_params['rho'] = opt_regime['rho']
        curr_params['decisions'] = opt_regime['decisions']
        curr_params['fractions'] = opt_regime['fractions']
        likelihood_data = DataLikelihoods(**curr_params)
        likelihood_data.increment_likelihood(**curr_params)
        if not np.abs(likelihood_data.likelihood - opt_likelihood) < 1e-3:
            warnings.warn(
                'Diff between computed likelihood and opt is greater than 0.0001')
            print(likelihood_data.likelihood - opt_likelihood)

        self.model_params = curr_params
        self.opt_regim = opt_regime
        self.N_data = likelihood_data.sub_data.query('setsize == {}'.format(N))

    def get_opt_corr_traces_plotly(self, bw=0.1):
        fractions = self.model_params['fractions']
        T = self.model_params['T']
        dt = self.model_params['dt']
        t_max = self.model_params['t_max']
        t_d = self.model_params['t_delay']
        t_values = np.arange(0, T, dt)
        maxind = int(t_max / dt)
        t_delayed = t_values[:maxind] + t_d
        presCorr = fractions[1][1, :]
        absCorr = fractions[0][0, :]
        normPres = presCorr[:maxind] / np.sum(presCorr[:maxind] * dt)
        normAbs = absCorr[:maxind] / np.sum(absCorr[:maxind] * dt)
        presMean = np.sum(fractions[1][1, :maxind] * t_delayed) / np.sum(fractions[1][1, :maxind])\
            + t_d
        absMean = np.sum(fractions[0][0, :maxind] * t_delayed) / np.sum(fractions[0][0, :maxind])\
            + t_d

        subj_rts = np.zeros((2, 3), dtype=object)
        subj_rts[0, 0] = self.N_data.query(
            'resp == 2 & target == \'Absent\'').rt.values
        subj_rts[0, 1] = self.N_data.query(
            'resp == 1 & target == \'Absent\'').rt.values
        subAbsMean = np.mean(subj_rts[0, 0])
        abs_timeouts = len(self.N_data.query(
            'resp == -1 & target == \'Absent\'').rt.values)
        tot_abs = len(subj_rts[0, 0]) + len(subj_rts[0, 1]) + abs_timeouts
        subAbsCorr_kde = gaussian_kde(
            subj_rts[0, 0], bw_method=0.1).evaluate(t_values[:maxind])

        subj_rts[1, 1] = self.N_data.query(
            'resp == 1 & target == \'Present\'').rt.values
        subj_rts[1, 0] = self.N_data.query(
            'resp == 2 & target == \'Present\'').rt.values
        subPresMean = np.mean(subj_rts[1, 1])
        pres_timeouts = len(self.N_data.query(
            'resp == -1 & target == \'Present\'').rt.values)
        tot_pres = len(subj_rts[1, 1]) + len(subj_rts[1, 0]) + pres_timeouts

        subPresCorr_kde = gaussian_kde(
            subj_rts[1, 1], bw_method=0.1).evaluate(t_values[:maxind])
        subAbsMean = np.mean(subj_rts[0, 0])

        # Graph objects for traces of RT
        simPresTrace = go.Scatter(x=t_values[:maxind], y=normPres, fill='tozeroy',
                                  name='Sim Pres Correct', hovertext='{:.2%} incorrect | pres, {:.2f}s mrt'.
                                  format(np.sum(fractions[1][0, :]), presMean),
                                  line={'color': 'orange'})
        simAbsTrace = go.Scatter(x=t_values[:maxind], y=normAbs, fill='tozeroy',
                                 name='Sim Abs Correct', hovertext='{:.2%} incorrect | abs, {:.2f}s mrt'.
                                 format(np.sum(fractions[0][1, :]), absMean),
                                 line={'color': 'purple'})
        subjPresTrace = go.Scatter(x=t_values[:maxind], y=subPresCorr_kde, fill='tozeroy',
                                   name='Subj Pres Correct', hovertext='{:.2%} incorrect | pres, {:.2f}s mrt'.
                                   format(len(subj_rts[1, 0]) / tot_pres, subPresMean),
                                   line={'color': 'red'})
        subjAbsTrace = go.Scatter(x=t_values[:maxind], y=subAbsCorr_kde, fill='tozeroy',
                                  name='Subj Abs Correct', hovertext='{:.2%} incorrect | abs, {:.2f}s mrt'.
                                  format(len(subj_rts[0, 1]) / tot_abs, subAbsMean),
                                  line={'color': 'blue'})

        # Layout dict containing lines indicating means of distributions
        layout = {'shapes': [{'type': 'line',
                              'xref': 'x',
                              'yref': 'paper',
                              'x0': presMean,
                              'y0': 0,
                              'x1': presMean,
                              'y1': 1,
                              'line': {'color': 'orange',
                                       'width': 4}
                              },
                             {'type': 'line',
                              'xref': 'x',
                              'yref': 'paper',
                              'x0': absMean,
                              'y0': 0,
                              'x1': absMean,
                              'y1': 1,
                              'line': {'color': 'purple',
                                       'width': 4}
                              },
                             {'type': 'line',
                              'xref': 'x',
                              'yref': 'paper',
                              'x0': subAbsMean,
                              'y0': 0,
                              'x1': subAbsMean,
                              'y1': 1,
                              'line': {'color': 'blue',
                                       'width': 4}
                              },
                             {'type': 'line',
                              'xref': 'x',
                              'yref': 'paper',
                              'x0': subPresMean,
                              'y0': 0,
                              'x1': subPresMean,
                              'y1': 1,
                              'line': {'color': 'red',
                                       'width': 4}
                              }
                             ]
                  }
        return [simAbsTrace, simPresTrace, subjAbsTrace, subjPresTrace], layout


if __name__ == '__main__':
    from pathlib import Path
    import os
    datapath = Path('~/Documents/fit_data/')
