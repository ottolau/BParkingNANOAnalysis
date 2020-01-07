#!/usr/bin/env python

#import ROOT
#from ROOT import TMVA, TFile, TTree, TCut
from subprocess import call
from os.path import isfile

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import xgboost as xgb

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.plots import plot_convergence

import numpy as np
import uproot
import pandas as pd
import h5py
import time

space  = [Integer(5, 7, name='max_depth'),
          Real(0.01, 0.2, name='eta'),
          Real(0.0, 1.0, name='gamma'),
          Real(0.5, 1.0, name='subsample'),
          Real(0.5, 1.0, name='colsample_bytree'),
          Real(0.0, 0.2, name='alpha'),
          Real(1.0, 1.5, name='lambda'),
          ]

# define the preprocessing function
# used to return the preprocessed training, test data, and parameter
# we can use this to do weight rescale, etc.
# as a example, we try to set scale_pos_weight
def fpreproc(dtrain, dtest, param):
    label = dtrain.get_label()
    ratio = float(np.sum(label == 0)) / np.sum(label == 1)
    param['scale_pos_weight'] = ratio
    return (dtrain, dtest, param)

@use_named_args(space)
def objective(**X):
    global best_auc
    global best_config
    print("New configuration: {}".format(X))
    params = X.copy()
    params['objective'] = 'binary:logitraw'
    params['eval_metric'] = 'auc'
    params['early_stopping_rounds'] = 100
    params['nthread'] = 6
    params['silent'] = 1

    cv_result = xgb.cv(params, xgtrain, num_boost_round=800, nfold=5, shuffle=True, fpreproc=fpreproc)
    ave_auc = cv_result['test-auc-mean'].iloc[-1]
    print("Average auc: {}".format(ave_auc))
    if ave_auc > best_auc:
      best_auc = ave_auc
      best_config = X
    print("Best auc: {}, Best configuration: {}".format(best_auc, best_config))

    return -ave_auc


import argparse
parser = argparse.ArgumentParser(description="A simple ttree plotter")
parser.add_argument("-s", "--signal", dest="signal", default="RootTree_BParkingNANO_2019Sep12_BuToKJpsi_Toee_mvaTraining_sig_training_pf.root", help="Signal file")
parser.add_argument("-b", "--background", dest="background", default="RootTree_BParkingNANO_2019Sep12_Run2018A2A3B2B3C2C3D2_mvaTraining_bkg_training_pf.root", help="Background file")
parser.add_argument("-f", "--suffix", dest="suffix", default=None, help="Suffix of the output name")
args = parser.parse_args()

filename = {}
upfile = {}
params = {}
df = {}

filename['bkg'] = args.background
filename['sig'] = args.signal

#branches = ['BToKEE_fit_l1_normpt', 'BToKEE_fit_l1_eta', 'BToKEE_fit_l1_phi', 'BToKEE_l1_dxy_sig', 'BToKEE_l1_dz', 'BToKEE_l1_mvaId', 'BToKEE_fit_l2_normpt', 'BToKEE_fit_l2_eta', 'BToKEE_fit_l2_phi', 'BToKEE_l2_dxy_sig', 'BToKEE_l2_dz', 'BToKEE_l2_mvaId', 'BToKEE_fit_k_normpt', 'BToKEE_fit_k_eta', 'BToKEE_fit_k_phi', 'BToKEE_k_DCASig', 'BToKEE_fit_normpt', 'BToKEE_svprob', 'BToKEE_fit_cos2D', 'BToKEE_l_xy_sig']

branches = ['BToKEE_fit_l1_normpt', 'BToKEE_fit_l1_eta', 'BToKEE_fit_l1_phi', 'BToKEE_l1_dxy_sig', 'BToKEE_l1_dz', 'BToKEE_fit_l2_normpt', 'BToKEE_fit_l2_eta', 'BToKEE_fit_l2_phi', 'BToKEE_l2_dxy_sig', 'BToKEE_l2_dz', 'BToKEE_fit_k_normpt', 'BToKEE_fit_k_eta', 'BToKEE_fit_k_phi', 'BToKEE_k_DCASig', 'BToKEE_fit_normpt', 'BToKEE_svprob', 'BToKEE_fit_cos2D', 'BToKEE_l_xy_sig']

input_dim = len(branches)

upfile['bkg'] = uproot.open(filename['bkg'])
upfile['sig'] = uproot.open(filename['sig'])

params['bkg'] = upfile['bkg']['tree'].arrays(branches)
params['sig'] = upfile['sig']['tree'].arrays(branches)

df['sig'] = pd.DataFrame(params['sig'])#[:30]
df['bkg'] = pd.DataFrame(params['bkg'])#[:30]

df['sig'].replace([np.inf, -np.inf], 10.0**+10, inplace=True)
df['bkg'].replace([np.inf, -np.inf], 10.0**+10, inplace=True)

nData = min(df['sig'].shape[0], df['bkg'].shape[0])

df['sig'] = df['sig'].sample(frac=1)#[:nData]
df['bkg'] = df['bkg'].sample(frac=1)#[:nData]


# add isSignal variable
df['sig']['isSignal'] = np.ones(len(df['sig']))
df['bkg']['isSignal'] = np.zeros(len(df['bkg']))

df_all = pd.concat([df['sig'],df['bkg']]).sample(frac=1).reset_index(drop=True)
dataset = df_all.values
X_data = dataset[:,0:input_dim]
Y_data = dataset[:,input_dim]

xgtrain = xgb.DMatrix(X_data, label=Y_data)

begt = time.time()
print("Begin Bayesian optimization")
best_auc = 0.0
best_config = {}
res_gp = gp_minimize(objective, space, n_calls=200, n_random_starts=100, random_state=3)
print("Finish optimization in {}s".format(time.time()-begt))

plt.figure()
plot_convergence(res_gp)
plt.savefig('BayesianOptimization_ConvergencePlot_BToKJpsi_Toee_XGB_{}.png'.format(args.suffix))



