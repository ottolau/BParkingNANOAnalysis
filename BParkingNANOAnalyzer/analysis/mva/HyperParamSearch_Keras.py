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


def build_custom_model(num_hiddens=2, initial_node=32, 
                          dropout=0.20, l2_lambda=1.e-5):
    inputs = Input(shape=(input_dim,), name = 'input')
    for i in range(num_hiddens):
        hidden = Dense(units=int(round(initial_node/np.power(2,i))), kernel_initializer='glorot_normal', activation='relu', kernel_regularizer=l2(l2_lambda))(inputs if i==0 else hidden)
        hidden = Dropout(np.float32(dropout))(hidden)
    outputs = Dense(1, name = 'output', kernel_initializer='normal', activation='sigmoid')(hidden)
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def train(model, batch_size=64):
    history = model.fit(X_train_val, 
                    Y_train_val, 
                    epochs=200, 
                    batch_size=batch_size, 
                    verbose=0,
                    callbacks=[early_stopping, model_checkpoint], 
                    validation_split=0.25)
    Y_predict = model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(Y_test, Y_predict)
    roc_auc = auc(fpr, tpr)
    return roc_auc
    #best_acc = max(history.history['val_acc'])
    #return best_acc

space  = [Integer(2, 5, name='hidden_layers'),
          Integer(32, 1024, name='initial_nodes'),
          Real(10**-6, 10**-3, "log-uniform", name='l2_lambda'),
          Real(0.0,0.5,name='dropout'),
          Integer(512,4096,name='batch_size'),
          Real(10**-5, 10**-1, "log-uniform", name='learning_rate'),
          ]

@use_named_args(space)
def objective(**X):
    print("New configuration: {}".format(X))

    model = build_custom_model(num_hiddens=X['hidden_layers'], initial_node=X['initial_nodes'], 
                      dropout=X['dropout'], l2_lambda=X['l2_lambda'])

    model.compile(optimizer=Adam(lr=X['learning_rate']), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    best_auc = train(model, batch_size=X['batch_size'])

    print("Best auc: {}".format(best_auc))
    return -best_auc


filename = {}
upfile = {}
params = {}
df = {}

filename['bkg'] = "../BsPhiJpsiEE_MVATraining_Bkg_pTcuts.root"
filename['sig'] = "../BsPhiJpsiEE_MVATraining_Sig_pTcuts.root"

branches = ['BToKEE_cos2D', 'BToKEE_l_xy_sig', 'BToKEE_svprob', 'BToKEE_l1_unBiased', 'BToKEE_l2_unBiased']

input_dim = len(branches)

upfile['bkg'] = uproot.open(filename['bkg'])
upfile['sig'] = uproot.open(filename['sig'])

params['bkg'] = upfile['bkg']['background'].arrays(branches)
params['sig'] = upfile['sig']['signal'].arrays(branches)

df['sig'] = pd.DataFrame(params['sig'])#[:1000]
df['bkg'] = pd.DataFrame(params['bkg'])#[:1000]

# add isSignal variable
df['sig']['isSignal'] = np.ones(len(df['sig']))
df['bkg']['isSignal'] = np.zeros(len(df['bkg']))

df_all = pd.concat([df['sig'],df['bkg']])
dataset = df_all.values
X = dataset[:,0:input_dim]
Y = dataset[:,input_dim]

X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.2, random_state=7)

early_stopping = EarlyStopping(monitor='val_loss', patience=30)

model_checkpoint = ModelCheckpoint('dense_model.h5', monitor='val_loss', 
                                   verbose=0, save_best_only=True, 
                                   save_weights_only=False, mode='auto', 
                                   period=1)


begt = time.time()
print("Begin Bayesian optimization")
res_gp = gp_minimize(objective, space, n_calls=200, random_state=3)
print("Finish optimization in {}s".format(time.time()-begt))

plt.figure()
plot_convergence(res_gp)
plt.savefig('BayesianOptimization_BsPhiJpsiEE_ConvergencePlot_GPU_noPhiM_pTcuts.png')


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


