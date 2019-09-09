#!/usr/bin/env python

#import ROOT
#from ROOT import TMVA, TFile, TTree, TCut
from subprocess import call
from os.path import isfile

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.plots import plot_convergence

import numpy as np
import uproot
import pandas as pd
import h5py
import time


def build_custom_model(n_estimators=100, max_depth=3, learning_rate=1.e-3):#, 
                                    #min_samples_split=0.01, min_samples_leaf=1):
    #model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
    return model

def train(model):
    model.fit(X_train_val, Y_train_val)
    Y_predict = model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(Y_test, Y_predict)
    roc_auc = auc(fpr, tpr)
    return roc_auc

space  = [Integer(100, 2000, name='n_estimators'),
          Integer(2, 5, name='max_depth'),
          Real(10**-5, 10**-1, "log-uniform", name='learning_rate'),
          #Real(0.001, 0.05, name='min_samples_split'),
          #Integer(1, 50,name='min_samples_leaf'),
          ]

@use_named_args(space)
def objective(**X):
    print("New configuration: {}".format(X))

    model = build_custom_model(n_estimators=X['n_estimators'], max_depth=X['max_depth'], learning_rate=X['learning_rate'])#,
                                        #min_samples_split=X['min_samples_split'], min_samples_leaf=X['min_samples_leaf'])

    best_auc = train(model)

    print("Best auc: {}".format(best_auc))
    return -best_auc


filename = {}
upfile = {}
params = {}
df = {}

filename['bkg'] = "../BsPhiJpsiEE_MVATraining_Bkg.root"
filename['sig'] = "../BsPhiJpsiEE_MVATraining_Sig.root"

branches = ['BToKEE_cos2D', 'BToKEE_l_xy_sig', 'BToKEE_svprob', 'BToKEE_l1_unBiased', 'BToKEE_l2_unBiased']

input_dim = len(branches)

upfile['bkg'] = uproot.open(filename['bkg'])
upfile['sig'] = uproot.open(filename['sig'])

params['bkg'] = upfile['bkg']['background'].arrays(branches)
params['sig'] = upfile['sig']['signal'].arrays(branches)

df['sig'] = pd.DataFrame(params['sig'])[:1000]
df['bkg'] = pd.DataFrame(params['bkg'])[:1000]

# add isSignal variable
df['sig']['isSignal'] = np.ones(len(df['sig']))
df['bkg']['isSignal'] = np.zeros(len(df['bkg']))

df_all = pd.concat([df['sig'],df['bkg']])
dataset = df_all.values
X = dataset[:,0:input_dim]
Y = dataset[:,input_dim]

X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.2, random_state=7)


begt = time.time()
print("Begin Bayesian optimization")
res_gp = gp_minimize(objective, space, n_calls=20, random_state=3)
print("Finish optimization in {}s".format(time.time()-begt))

plt.figure()
plot_convergence(res_gp)
plt.savefig('test.png')


print("Best parameters: \
\nbest_n_estimators = {} \
\nbest_max_depth = {} \
\nbest_learning_rate = {} \
\nbest_min_samples_split = {} \
\nbest_min_samples_leaf = {} \
\nbest_roc_auc = {}".format(res_gp.x[0],
                            res_gp.x[1],
                            res_gp.x[2],
                            0,#res_gp.x[3],
                            0,#res_gp.x[4],
                            -1.0*res_gp.fun))


