'''
August 2018

Class based implementation of the dynamic model
'''
import numpy as np
from finegr_model import FineGrained
from bellman_utilities import BellmanUtil
from fitter import Fitter
import sys
import pandas as pd
from pathlib import Path

T = 10
dt = 0.05
t_w = 0.5
size = 100
lapse = 1e-6

# Returns a path object that works as a string for most functions
datapath = Path("../data/exp1.csv")
savepath = Path("~/Documents/")  # Where to save figures
savepath = str(savepath.expanduser())

try:
    subject_num = sys.argv[1]
    if not subject_num.isnumeric():
        subject_num = 1
        print('Invalid subject number passed at prompt. Setting subject to 1')
except ValueError:
    subject_num = 1
    print('No subject number passed at prompt. Setting subject to 1')

print('Subject number {}'.format(subject_num))

exp1 = pd.read_csv(datapath, index_col=None)  # read data
exp1.rename(columns={'sub': 'subno'}, inplace=True)


model_list = [
    ('sig', 'sym', 'const'),
    #    ('sig_reward', 'asym_reward', 'const'),
    #    ('sig_punish', 'epsilon_punish', 'const'),
    #    ('sig', 'sym', 'sqrt'),
    #    ('sig_reward', 'asym_reward', 'sqrt'),
    #    ('sig_punish', 'epsilon_punish', 'sqrt')
]

model_dict = {}
for model_type in model_list:
    print(model_type)
    model_dict[model_type] = Fitter(exp1, 1, model_type, 3, T, t_w, dt, size, lapse)
