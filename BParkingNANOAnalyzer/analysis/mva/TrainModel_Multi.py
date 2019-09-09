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
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb

import numpy as np
import uproot
import pandas as pd
import h5py
import time


def build_custom_model(hyper_params, classifier):
    if classifier == 'Keras':
        inputs = Input(shape=(input_dim,), name = 'input')
        for i in range(hyper_params['num_hiddens']):
            hidden = Dense(units=int(round(hyper_params['initial_node']/np.power(2,i))), kernel_initializer='glorot_normal', activation='relu', kernel_regularizer=l2(hyper_params['l2_lambda']))(inputs if i==0 else hidden)
            hidden = Dropout(np.float32(hyper_params['dropout']))(hidden)
        outputs = Dense(1, name = 'output', kernel_initializer='normal', activation='sigmoid')(hidden)
        model = Model(inputs=inputs, outputs=outputs)

    if classifier == 'GTB':
        #model = GradientBoostingClassifier(n_estimators=hyper_params['n_estimators'], learning_rate=hyper_params['learning_rate'], max_depth=hyper_params['max_depth'], min_samples_split=hyper_params['min_samples_split'], min_samples_leaf=hyper_params['min_samples_leaf'])
        model = GradientBoostingClassifier(n_estimators=hyper_params['n_estimators'], learning_rate=hyper_params['learning_rate'], max_depth=hyper_params['max_depth'])

    if classifier == 'XGB':
        model = xgb.XGBClassifier(max_depth=hyper_params['max_depth'], n_estimators=hyper_params['n_estimators'], learning_rate=hyper_params['learning_rate'])

    return model



def train(model, classifier, hyper_params=None):
    if classifier == 'Keras':
        history = model.fit(X_train_val, 
                        Y_train_val, 
                        epochs=200, 
                        batch_size=hyper_params['batch_size'], 
                        verbose=0,
                        callbacks=[early_stopping, model_checkpoint], 
                        validation_split=0.25)
        Y_predict = model.predict(X_test)
        fpr, tpr, thresholds = roc_curve(Y_test, Y_predict)
        roc_auc = auc(fpr, tpr)
        return roc_auc, history, model
        #best_acc = max(history.history['val_acc'])
        #return best_acc

    if classifier == 'GTB' or classifier == 'XGB':
        model.fit(X_train_val, Y_train_val)
        Y_predict = model.predict(X_test)
        fpr, tpr, thresholds = roc_curve(Y_test, Y_predict)
        roc_auc = auc(fpr, tpr)
        return roc_auc, model


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

use_classifiers = {'Keras': True, 'GTB': True, 'XGB': True}
model = {}

if use_classifiers['Keras']:
    early_stopping = EarlyStopping(monitor='val_loss', patience=30)

    model_checkpoint = ModelCheckpoint('dense_model.h5', monitor='val_loss', 
                                       verbose=0, save_best_only=True, 
                                       save_weights_only=False, mode='auto', 
                                       period=1)

    hyper_params = {'hidden_layers': 3, 'initial_nodes': 64, 'l2_lambda': 10.0**-4, 'dropout': 0.25, 'batch_size': 512, 'learning_rate': 10.0**-3}

    model['Keras'] = build_custom_model(hyper_params, 'Keras') 
    model['Keras'].compile(optimizer=Adam(lr=hyper_params['learning_rate']), loss='binary_crossentropy', metrics=['accuracy'])
    model['Keras'].summary()
    best_auc, history, model['Keras'] = train(model, 'Keras', hyper_params)

    print("Best auc: {}".format(best_auc))


if use_classifiers['GTB']:
    hyper_params = {'n_estimators': 200, 'max_depth': 3, 'min_samples_split': 0.001, 'min_samples_leaf': 2, 'learning_rate': 10.0**-3}

    model['GTB'] = build_custom_model(hyper_params, 'GTB')
    best_auc, model['GTB'] = train(model['GTB'])

    print("Best auc: {}".format(best_auc))

if use_classifiers['XGB']:
    hyper_params = {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 10.0**-3}

    model['XGB'] = build_custom_model(hyper_params, 'XGB')
    best_auc, model['XGB'] = train(model['XGB'])

    print("Best auc: {}".format(best_auc))






