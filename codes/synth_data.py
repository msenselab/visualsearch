import numpy as np
import csv
import dyn_case_model as dcn


class DataGen:
    def __init__(self, fine_sigma, reward=None, punishment=None, reward_scheme='sym',
                 model_type='const'):

        self.d_map_samples = int(100000)
        self.numsims = 5000
        # Set default reward and punishment if none passed
        if not reward and not punishment:
            reward = 1
            punishment = 0

        # Generate coarse statistics from fine sigma
        finemodel = dcn.FineGrained(fine_sigma, model_type, self.d_map_samples)
        self.coarse_stats = finemodel.coarse_stats

        self.synth_rts = []
        for i in range(self.coarse_stats.shape[0]):
            mu = self.coarse_stats[i, :, 0]
            sigma = self.coarse_stats[i, :, 1]
            probs = dcn.trans_probs(sigma, mu)
            rho = dcn.solve_rho(reward, punishment, reward_scheme, sigma, mu, probs)
            decisions = dcn.back_induct(reward, punishment, rho, sigma, mu, probs, reward_scheme)[1]
            self.synth_rts.append(dcn.get_rt(sigma, mu, decisions, numsims=self.numsims,
                                             parallelize=True))

    def save_csv(self, savepath):
        with open(savepath, 'w') as fw:
            writer = csv.writer(fw)
            writer.writerow(['target', 'setsize', 'dyn', 'resp', 'rt', 'sub', 'exp', 'correct'])
            for i in range(dcn.N_array.shape[0]):
                curr_N = dcn.N_array[i]
                abs_resp = self.synth_rts[i][0]
                pres_resp = self.synth_rts[i][1]
                # First write all responses for target absent simulations
                for response, rt in abs_resp:
                    correct = response == 0
                    if response == 0:
                        adjusted_response = 2
                    elif response == 1:
                        adjusted_response = 1
                    elif response is np.nan:
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
                    elif response is np.nan:
                        adjusted_response = -1
                    writer.writerow(['Present', curr_N, 'Dynamic',
                                     adjusted_response, "{:.7f}".format(rt), 666, 1, correct])
