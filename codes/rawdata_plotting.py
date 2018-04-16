'''
1 December 2017

Script for plotting the distributions of response time across
various conditions

--Berk
'''

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import csv
import itertools as it

datapath = '/home/berk/Documents/strongway_dynsearch/data/'
savepath = '/home/berk/Documents/vis_search_misc/figs/'

## load in experiment 1: No rewards for answers
readfile = open(datapath + 'exp1.csv', 'r')
reader = csv.reader(readfile)

exp1 = []
for row in reader:
    exp1.append(row)
exp1 = np.array(exp1[1:])

# iterate through combinations of set size, display condition,
# and correct/incorrect response
response_times = {}
response = 0
setsizes = [8, 12, 16]
targets = ['Present', 'Absent']
displays = ['Static', 'Dynamic']

combinations = it.product(setsizes, displays, targets)

# create a list of the response times for each combination of conditions and
# store it in the dict of response times
for setsize, display, target in combinations:
    set_rts = [float(x[4]) for x in exp1 if x[1] == str(setsize)
                                        and x[2] == display
                                        and x[0] == str(target)
                                        and x[-1] == str(response)]
    response_times[(setsize, display, target)] = np.array(set_rts)

# plot distributions
fig, ax = plt.subplots(2, 2, figsize = (11,8.5))

for i, display in enumerate(displays):
    for j, target in enumerate(targets):
        for setsize in setsizes:
            keytuple = (setsize, display, target)
            ax[i, j].hist(response_times[keytuple],
                          bins=25, alpha=0.5, label="setsize %d, N = %d" %
                          (setsize, len(response_times[keytuple])))
            ax[i, j].set_title("%s, target = %s" % (display, target))
            ax[i, j].set_xlabel('RT (s)')
plt.legend()
plt.savefig(savepath + 'noreward_RT_dists.png', DPI=500)

# Experiment 2: Rewards with no static display density
readfile = open(datapath + 'exp2.csv', 'r')
reader = csv.reader(readfile)

exp2 = []
for row in reader:
    exp2.append(row)
exp2 = np.array(exp2[1:])

# iterate through combinations of set size, display condition,
# correct/incorrect response, and which correct response was rewarded
response_times = {}
response = 0
setsizes = [8, 12, 16]
targets = ['Absent', 'Present']
displays = ['Static', 'Dynamic']
reward_conds = ['Absent', 'Present']

combinations = it.product(setsizes, displays, targets, reward_conds)

# create a list of the response times for each combination of conditions and
# store it in the dict of response times
for setsize, display, target, reward_cond in combinations:
    set_rts = [float(x[4]) for x in exp2 if x[1] == str(setsize)
                                        and x[2] == display
                                        and x[-1] == str(response)
                                        and x[-2] == reward_cond
                                        and x[0] == target]
    keytuple = (setsize, display, target, reward_cond)
    response_times[keytuple] = np.array(set_rts)

# Plot absent reward distributions
fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))

for i, display in enumerate(displays):
    for j, target in enumerate(targets):
        for setsize in setsizes:
            keytuple = (setsize, display, target, 'Absent')
            ax[i, j].hist(response_times[keytuple], bins=25, alpha=0.5,
                          label="setsize %d, N = %d" % (setsize, len(response_times[keytuple])))
            ax[i, j].set_title("%s, target = %s" % (display,
                                                    target))
            ax[i, j].set_xlabel('RT (s)')
plt.suptitle("Reward for correct Absent response")
plt.legend()
plt.savefig(savepath + "absreward_RT_dists.png", DPI=500)

# Plot present reward distributions
fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))

for i, display in enumerate(displays):
    for j, target in enumerate(targets):
        for setsize in setsizes:
            keytuple = (setsize, display, target, 'Present')
            ax[i, j].hist(response_times[keytuple], bins=25, alpha=0.5,
                          label="setsize %d, N = %d" % (setsize, len(response_times[keytuple])))
            ax[i, j].set_title("%s, target = %s" % (display,
                                                    target))
            ax[i, j].set_xlabel("RT (s)")
plt.suptitle("Reward for correct Present response")
plt.legend()
plt.savefig(savepath + "presreward_RT_dists.png", DPI=500)
