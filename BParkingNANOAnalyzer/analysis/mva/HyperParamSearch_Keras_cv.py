#!/usr/bin/env python
from subprocess import call
from os.path import isfile

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Dropout
from keras.regularizers import l2
from keras import initializers
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import roc_curve, auc
from sklearn.utils import compute_class_weight

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.plots import plot_convergence

import numpy as np
import uproot
import pandas as pd
import h5py
import time


def build_custom_model(num_hiddens=2, initial_node=32, 
                          dropout=0.20, l2_lambda=1.e-5):
    inputs = Input(shape=(input_dim,), name = 'input')
    for i in range(num_hiddens):
        hidden = Dense(units=int(round(initial_node/np.power(2,i))), kernel_initializer='glorot_normal', activation='relu', kernel_regularizer=l2(l2_lambda))(inputs if i==0 else hidden)
        hidden = Dropout(np.float32(dropout))(hidden)
    outputs = Dense(1, name = 'output', kernel_initializer='normal', activation='sigmoid')(hidden)
    model = Model(inputs=inputs, outputs=outputs)
    
    return model


def train(X_train_val, Y_train_val, X_test, Y_test, model, classWeight, batch_size=64):
    history = model.fit(X_train_val, 
                    Y_train_val, 
                    epochs=200, 
                    batch_size=batch_size, 
                    verbose=0,
                    class_weight=classWeight,
                    callbacks=[early_stopping, model_checkpoint], 
                    validation_split=0.25)
    Y_predict = model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(Y_test, Y_predict)
    roc_auc = auc(fpr, tpr)
    return roc_auc
    #best_acc = max(history.history['val_acc'])
    #return best_acc

space  = [Integer(2, 4, name='hidden_layers'),
          Integer(32, 256, name='initial_nodes'),
          Real(10**-5, 10**-1, "log-uniform", name='l2_lambda'),
          Real(0.15,0.5,name='dropout'),
          Integer(256,4096,name='batch_size'),
          Real(10**-5, 10**-1, "log-uniform", name='learning_rate'),
          ]

@use_named_args(space)
def objective(**X):
    global best_auc
    global best_config
    print("New configuration: {}".format(X))

    model = build_custom_model(num_hiddens=X['hidden_layers'], initial_node=X['initial_nodes'], 
                      dropout=X['dropout'], l2_lambda=X['l2_lambda'])

    model.compile(optimizer=Adam(lr=X['learning_rate']), loss='binary_crossentropy', metrics=['accuracy'], weighted_metrics=['accuracy'])
    model.summary()

    aucs = []
    cv = StratifiedKFold(n_splits=10, shuffle=True)
    for train_idx, test_idx in cv.split(X_data, Y_data):
      X_train_val = X_data[train_idx]
      X_test = X_data[test_idx]
      Y_train_val = Y_data[train_idx]
      Y_test = Y_data[test_idx]

      classWeight = compute_class_weight('balanced', np.unique(Y_train_val), Y_train_val) 
      classWeight = dict(enumerate(classWeight))

      aucs.append(train(X_train_val, Y_train_val, X_test, Y_test, model, classWeight, batch_size=X['batch_size']))

    ave_auc = sum(aucs)/float(len(aucs))
    print("Average auc: {}".format(ave_auc))
    if ave_auc > best_auc:
      best_auc = ave_auc
      best_config = X
    print("Best auc: {}, Best configuration: {}".format(best_auc, best_config))

    return -ave_auc


import argparse
parser = argparse.ArgumentParser(description="A simple ttree plotter")
parser.add_argument("-s", "--signal", dest="signal", default="RootTree_BParkingNANO_2019Sep12_BuToKJpsi_Toee_mvaTraining_sig_pf.root", help="Signal file")
parser.add_argument("-b", "--background", dest="background", default="RootTree_BParkingNANO_2019Sep12_Run2018A2A3B2B3C2C3D2_mvaTraining_bkg_pf.root", help="Background file")
parser.add_argument("-f", "--suffix", dest="suffix", default=None, help="Suffix of the output name")
args = parser.parse_args()

filename = {}
upfile = {}
params = {}
df = {}

filename['bkg'] = args.background
filename['sig'] = args.signal

#branches = ['BToKEE_l1_normpt', 'BToKEE_l1_eta', 'BToKEE_l1_phi', 'BToKEE_l1_dxy_sig', 'BToKEE_l1_dz', 'BToKEE_l1_mvaId', 'BToKEE_l2_normpt', 'BToKEE_l2_eta', 'BToKEE_l2_phi', 'BToKEE_l2_dxy_sig', 'BToKEE_l2_dz', 'BToKEE_l2_mvaId', 'BToKEE_k_normpt', 'BToKEE_k_eta', 'BToKEE_k_phi', 'BToKEE_k_DCASig', 'BToKEE_normpt', 'BToKEE_svprob', 'BToKEE_cos2D', 'BToKEE_l_xy_sig']
branches = ['BToKEE_l1_normpt', 'BToKEE_l1_eta', 'BToKEE_l1_phi', 'BToKEE_l1_dxy_sig', 'BToKEE_l1_dz', 'BToKEE_l2_normpt', 'BToKEE_l2_eta', 'BToKEE_l2_phi', 'BToKEE_l2_dxy_sig', 'BToKEE_l2_dz', 'BToKEE_k_normpt', 'BToKEE_k_eta', 'BToKEE_k_phi', 'BToKEE_k_DCASig', 'BToKEE_normpt', 'BToKEE_svprob', 'BToKEE_cos2D', 'BToKEE_l_xy_sig']
#branches = ['BToKEE_l1_normpt', 'BToKEE_l1_eta', 'BToKEE_l1_phi', 'BToKEE_l1_dxy_sig', 'BToKEE_l1_dz', 'BToKEE_l2_normpt', 'BToKEE_l2_eta', 'BToKEE_l2_phi', 'BToKEE_l2_dxy_sig', 'BToKEE_l2_dz', 'BToKEE_k_normpt', 'BToKEE_k_eta', 'BToKEE_k_phi', 'BToKEE_k_DCASig', 'BToKEE_normpt', 'BToKEE_svprob', 'BToKEE_cos2D']

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


early_stopping = EarlyStopping(monitor='val_loss', patience=100)

model_checkpoint = ModelCheckpoint('dense_model.h5', monitor='val_loss', 
                                   verbose=0, save_best_only=True, 
                                   save_weights_only=False, mode='auto', 
                                   period=1)


begt = time.time()
print("Begin Bayesian optimization")
best_auc = 0.0
best_config = {}
res_gp = gp_minimize(objective, space, n_calls=300, n_random_starts=150, random_state=3)
print("Finish optimization in {}s".format(time.time()-begt))

plt.figure()
plot_convergence(res_gp)
plt.savefig('BayesianOptimization_ConvergencePlot_BToKJpsi_Toee_{}.png'.format(args.suffix))


print("Best parameters: \
\nbest_hidden_layers = {} \
\nbest_initial_nodes = {} \
\nbest_l2_lambda = {} \
\nbest_dropout = {} \
\nbest_batch_size = {} \
\nbest_learning_rate = {} \
\nbest_roc_auc = {}".format(res_gp.x[0],
                            res_gp.x[1],
                            res_gp.x[2],
                            res_gp.x[3],
                            res_gp.x[4],
                            res_gp.x[5],
                            -1.0*res_gp.fun))


