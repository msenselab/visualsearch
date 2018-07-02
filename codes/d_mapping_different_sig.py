import numpy as np
from scipy.stats import gaussian_kde
from scipy.stats import invgamma
import matplotlib.pyplot as plt
import seaborn as sns


def d_map(N, epsilons, sigma):
    return -1 * (1 / (2 * (sigma**2))) + np.log(1 / N) + np.log(np.sum(np.exp(epsilons / sigma**2)))


def d_map_alt(N, x_s, sigma):
    sum = 0
    for i in range(N):
        sum += np.exp((x_s[i] - 0.5) / sigma[i]**2)

    return np.log((1 / N) * sum)


def sample_epsilon(C, N, sigma):
    for i in range(N):
        epsilons = np.random.normal(0, sigma, N)
    if C == 1:
        epsilons[0] = np.random.normal(1, sigma)
    return epsilons


def sample_epsilon_alt(C, N, sigma):
    for i in range(N):
        epsilons = np.random.normal(0, sigma, N)
    if C == 1:
        epsilons[0] = np.random.normal(1, sigma)
    return epsilons


def sim(fine_sigma, num_change, N):
    sim_data_pres = []
    sim_data_abs = []
    for j in range(num_change):
        length = np.random.uniform(0, 1, size = N)
        sigma = length**2 * fine_sigma
        for i in range(200):
            sim_data_pres.append(d_map_alt(N, sample_epsilon_alt(1, N, fine_sigma), sigma))
            sim_data_abs.append(d_map_alt(N, sample_epsilon_alt(1, N, fine_sigma), sigma))
    return sim_data_pres, sim_data_abs

fine_sigma = 5

sim_data_8 = sim(fine_sigma, 50, 8)
#sim_data_12 = sim(fine_sigma, 500, 12)
#sim_data_16 = sim(fine_sigma, 500, 16)


# sim_data = np.zeros((10000, 2, 3))
# for i in range(200):

# sim_data[i][0][0] = d_map(8, sample_epsilon(1, 8))
# sim_data[i][1][0] = d_map(8, sample_epsilon(0, 8))
# sim_data[i][0][1] = d_map(12, sample_epsilon(1, 12))
# sim_data[i][1][1] = d_map(12, sample_epsilon(0, 12))
# sim_data[i][0][2] = d_map(16, sample_epsilon(1, 16))
# sim_data[i][1][2] = d_map(16, sample_epsilon(0, 16))


# stats = np.zeros((2, 2, 3))
# for i in range(2):
#     for j in range(3):
#         stats[0][i][j] = np.mean(sim_data[:, i, j])
#         stats[1][i][j] = np.sqrt(np.var(sim_data[:, i, j]))

sns.set_style('whitegrid')

fig = plt.figure()
ax1 = fig.add_subplot(131)
# sns.kdeplot(sim_data[:, 0, 0], bw=0.5)
# sns.kdeplot(sim_data[:, 1, 0], bw=0.5)

sns.kdeplot(sim_data_8[0], bw=0.5)
sns.kdeplot(sim_data_8[1], bw=0.5)

ax1.set_title("N=8")

plt.show()

ax2 = fig.add_subplot(132)

# sns.kdeplot(sim_data[:, 0, 1], bw=0.5)
# sns.kdeplot(sim_data[:, 1, 1], bw=0.5)

sns.kdeplot(sim_data_12[0], bw=0.5)
sns.kdeplot(sim_data_12[1], bw=0.5)


ax2.set_title("N=12")


ax3 = fig.add_subplot(133)

# sns.kdeplot(sim_data[:, 0, 2], bw=0.5)
# sns.kdeplot(sim_data[:, 1, 2], bw=0.5)

sns.kdeplot(sim_data_16[0], bw=0.5)
sns.kdeplot(sim_data_16[1], bw=0.5)

ax3.set_title("N=16")

plt.subplots_adjust(wspace=0.35)

plt.show()
