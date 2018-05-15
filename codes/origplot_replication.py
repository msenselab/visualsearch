'''
27 April 2018

Script for plotting the summary of response time across
various conditions

--Berk

Use Pandas to simplify data manipulation - Strongway
'''

import numpy as np
import matplotlib.pyplot as plt
import csv
import itertools as it
import pandas as pd
import seaborn as sns

# change paths to relative path, so it can be across computers
datapath = '../data/'
savepath = '../figs/'

usePandas = True

if usePandas:
    exp1 = pd.read_csv(datapath + 'exp1.csv',index_col = None) # read data
    exp1.rename(columns = {'sub':'subno'},inplace = True) # .sub is a keyword, change it
    # filter correct trials, group by factors and average RTs
    mrt = exp1.query('correct == 1') \
            .groupby(['subno','dyn','setsize','target'])\
            .agg({'rt':'mean'}).reset_index() 
    # visualize it   
    sns.factorplot(x='setsize', y = 'rt', hue = 'target',col = 'dyn',data = mrt)
    

# original code       
else:
    # load in experiment 1: No rewards for answers
    readfile = open(datapath + 'exp1.csv', 'r')
    reader = csv.reader(readfile)
    
    exp1 = []
    for row in reader:
        exp1.append(row)
    exp1 = np.array(exp1[1:])
    subs = [x[-3] for x in exp1]
    numsubs = len(set(subs))
    
    # iterate through combinations of set size, display condition,
    # and correct/incorrect response
    response_times = {}
    response = 1
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
    
    setsize_traces = {}
    for i, display in enumerate(displays):
        for j, target in enumerate(targets):
            setsize_traces[(display, target)] = []
            for setsize in setsizes:
                keytuple = (setsize, display, target)
                currdata = response_times[keytuple]
                mean = np.mean(currdata)
                sem = np.std(currdata) / np.sqrt(numsubs)
                setsize_traces[(display, target)].append((mean, sem))
    
    plt.figure()
    for display in displays:
        if display == 'Dynamic':
            c = 'r'
        else:
            c = 'b'
        for target in targets:
            if target == 'Present':
                ls = '--'
                form = '^'
            else:
                ls = '-'
                form = 'o'
            currdata = np.array(setsize_traces[(display, target)])
            plt.errorbar(setsizes, currdata[:, 0], yerr=currdata[:, 1],
                         linestyle=ls, color=c, fmt=form, capsize=4,
                         label="{} {}".format(display, target))
    plt.legend()
