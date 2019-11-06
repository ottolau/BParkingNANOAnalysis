#!/usr/bin/env python

from subprocess import call
from os.path import isfile
import sys
import os

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Dropout
from keras.regularizers import l2
from keras import initializers
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import compute_class_weight, compute_sample_weight
import xgboost as xgb
from sklearn import svm

from sklearn.externals import joblib

import numpy as np
import uproot
import pandas as pd
import h5py
import time


def build_custom_model(hyper_params, classifier):
    if classifier == 'Keras':
        inputs = Input(shape=(input_dim,), name = 'input')
        for i in range(hyper_params['hidden_layers']):
            hidden = Dense(units=int(round(hyper_params['initial_nodes']/np.power(2,i))), kernel_initializer='glorot_normal', activation='relu', kernel_regularizer=l2(hyper_params['l2_lambda']))(inputs if i==0 else hidden)
            hidden = Dropout(np.float32(hyper_params['dropout']))(hidden)
        outputs = Dense(1, name = 'output', kernel_initializer='normal', activation='sigmoid')(hidden)
        model = Model(inputs=inputs, outputs=outputs)

    if classifier == 'GTB':
        #model = GradientBoostingClassifier(n_estimators=hyper_params['n_estimators'], learning_rate=hyper_params['learning_rate'], max_depth=hyper_params['max_depth'], min_samples_split=hyper_params['min_samples_split'], min_samples_leaf=hyper_params['min_samples_leaf'])
        model = GradientBoostingClassifier(n_estimators=hyper_params['n_estimators'], learning_rate=hyper_params['learning_rate'], max_depth=hyper_params['max_depth'], verbose=1)

    if classifier == 'XGB':
        model = xgb.XGBClassifier(max_depth=hyper_params['max_depth'], n_estimators=hyper_params['n_estimators'], learning_rate=hyper_params['learning_rate'])
    
    if classifier == 'SVM':
        model = svm.SVC(C=hyper_params['C'], gamma=hyper_params['gamma'], verbose=1, class_weight=classWeight)

    return model



def train(model, classifier, hyper_params=None):
    if classifier == 'Keras':
        history = model.fit(X_train_val, 
                        Y_train_val, 
                        epochs=400, 
                        batch_size=hyper_params['batch_size'], 
                        verbose=1,
                        class_weight=classWeight,
                        callbacks=[early_stopping, model_checkpoint], 
                        validation_split=0.25)
        return model, history

    if classifier == 'GTB' or classifier == 'SVM':
        model.fit(X_train_val, Y_train_val)
        return model

    if classifier == 'XGB':
        xgtrain = xgb.DMatrix(X_train_val, label=Y_train_val)
        params = hyper_params.copy()
        label = xgtrain.get_label()
        ratio = float(np.sum(label == 0)) / np.sum(label == 1)
        params['scale_pos_weight'] = ratio
        params['objective'] = 'binary:logitraw'
        #params['objective'] = 'binary:logistic'
        params['eval_metric'] = 'auc'
        #params['early_stopping_rounds'] = 100
        params['nthread'] = 10
        params['silent'] = 1

        model = xgb.train(params, xgtrain, num_boost_round=800, verbose_eval=False)
        return model


if __name__ == "__main__":
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

    branches = sorted(['BToKEE_l1_normpt', 'BToKEE_l1_eta', 'BToKEE_l1_phi', 'BToKEE_l1_dxy_sig', 'BToKEE_l1_dz', 'BToKEE_l1_mvaId', 'BToKEE_l2_normpt', 'BToKEE_l2_eta', 'BToKEE_l2_phi', 'BToKEE_l2_dxy_sig', 'BToKEE_l2_dz', 'BToKEE_l2_mvaId', 'BToKEE_k_normpt', 'BToKEE_k_eta', 'BToKEE_k_phi', 'BToKEE_k_DCASig', 'BToKEE_normpt', 'BToKEE_svprob', 'BToKEE_cos2D', 'BToKEE_l_xy_sig'])
    #branches = sorted(['BToKEE_l1_normpt', 'BToKEE_l1_eta', 'BToKEE_l1_phi', 'BToKEE_l1_dxy_sig', 'BToKEE_l1_dz', 'BToKEE_l2_normpt', 'BToKEE_l2_eta', 'BToKEE_l2_phi', 'BToKEE_l2_dxy_sig', 'BToKEE_l2_dz', 'BToKEE_k_normpt', 'BToKEE_k_eta', 'BToKEE_k_phi', 'BToKEE_k_DCASig', 'BToKEE_normpt', 'BToKEE_svprob', 'BToKEE_cos2D', 'BToKEE_l_xy_sig'])

    input_dim = len(branches)

    upfile['bkg'] = uproot.open(filename['bkg'])
    upfile['sig'] = uproot.open(filename['sig'])

    params['bkg'] = upfile['bkg']['tree'].arrays(branches)
    params['sig'] = upfile['sig']['tree'].arrays(branches)

    df['sig'] = pd.DataFrame(params['sig']).sort_index(axis=1)#[:30]
    df['bkg'] = pd.DataFrame(params['bkg']).sort_index(axis=1)#[:30]
    #print(df['sig'][np.logical_not(np.isfinite(df['sig']['BToKEE_l_xy_sig']))])

    df['sig'].replace([np.inf, -np.inf], 10.0**+10, inplace=True)
    df['bkg'].replace([np.inf, -np.inf], 10.0**+10, inplace=True)
    #print(df['sig'][np.logical_not(np.isfinite(df['sig']['BToKEE_l_xy_sig']))])

    nData = min(df['sig'].shape[0], df['bkg'].shape[0])
    print(nData)

    #df['sig'] = df['sig'].sample(frac=1)[:nData]
    #df['bkg'] = df['bkg'].sample(frac=1)[:nData]

    # add isSignal variable
    df['sig']['isSignal'] = np.ones(len(df['sig']))
    df['bkg']['isSignal'] = np.zeros(len(df['bkg']))

    df_all = pd.concat([df['sig'],df['bkg']]).sort_index(axis=1).sample(frac=1).reset_index(drop=True)
    dataset = df_all.values
    X_train_val = dataset[:,0:input_dim]
    Y_train_val = dataset[:,input_dim]

    classWeight = compute_class_weight('balanced', np.unique(Y_train_val), Y_train_val) 
    classWeight = dict(enumerate(classWeight))
    print(classWeight)

    use_classifiers = {'Keras': False, 'GTB': False, 'XGB': True, 'SVM': False}
    model = {}

    if use_classifiers['Keras']:
        print('Training Keras Neural Network...')
        early_stopping = EarlyStopping(monitor='val_loss', patience=100)

        model_checkpoint = ModelCheckpoint('dense_model_fulldata_{}.h5'.format(args.suffix), monitor='val_loss', 
                                           verbose=1, save_best_only=True, 
                                           save_weights_only=False, mode='auto', 
                                           period=1)

        #hyper_params = {'hidden_layers': 3, 'initial_nodes': 64, 'l2_lambda': 10.0**-4, 'dropout': 0.25, 'batch_size': 512, 'learning_rate': 10.0**-3}
        hyper_params = {'hidden_layers': 2, 'initial_nodes': 247, 'l2_lambda': 4.009859234023707e-05, 'dropout': 0.408861611174969, 'batch_size': 1758, 'learning_rate': 0.0009008120868100766} # PF
        #hyper_params = {'hidden_layers': 3, 'initial_nodes': 471, 'l2_lambda': 0.0009291606742300548, 'dropout': 0.3082756432562473, 'batch_size': 492, 'learning_rate': 0.00033698412975590987} # Low

        model['Keras'] = build_custom_model(hyper_params, 'Keras') 
        model['Keras'].compile(optimizer=Adam(lr=hyper_params['learning_rate']), loss='binary_crossentropy', metrics=['accuracy'], weighted_metrics=['accuracy'])
        model['Keras'].summary()
        model['Keras'], history = train(model['Keras'], 'Keras', hyper_params)


    if use_classifiers['GTB']:
        print('Training Gradient Boosting Tree...')
        hyper_params = {'n_estimators': 200, 'max_depth': 3, 'min_samples_split': 0.001, 'min_samples_leaf': 2, 'learning_rate': 10.0**-3}

        model['GTB'] = build_custom_model(hyper_params, 'GTB')
        model['GTB'] = train(model['GTB'], 'GTB')
        # save model to file
        joblib.dump(model['GTB'], "gtb_fulldata_{}.joblib.dat".format(args.suffix))

    if use_classifiers['XGB']:
        print('Training XGBoost...')
        #hyper_params = {'colsample_bytree': 0.9044646018957753, 'subsample': 0.6626530919329603, 'eta': 0.013916755880706982, 'alpha': 0.07979819085895129, 'max_depth': 6, 'gamma': 0.14271732920694105, 'lambda': 1.212350804702256} # PF
        #hyper_params = {'colsample_bytree': 0.6607323533198513, 'subsample': 0.8549783548086116, 'eta': 0.12721398718375884, 'alpha': 0.08311984221421874, 'max_depth': 7, 'gamma': 0.571135630090849, 'lambda': 1.2855493741184907} # Mix
        #hyper_params = {'colsample_bytree': 0.8020747383215419, 'subsample': 0.7014327644533827, 'eta': 0.02973077790685988, 'alpha': 0.0015234991615051378, 'max_depth': 6, 'gamma': 0.9183332340428476, 'lambda': 1.2443558028940713}
        hyper_params = {'colsample_bytree': 0.7265861505610647, 'subsample': 0.5726850720377014, 'eta': 0.02777478451843462, 'alpha': 0.04311540168298377, 'max_depth': 4, 'gamma': 0.6948236527284698, 'lambda': 1.1769525359431465}

        #model['XGB'] = build_custom_model(hyper_params, 'XGB')
        model['XGB'] = train(None, 'XGB', hyper_params=hyper_params)
        # save model to file
        #cwd = os.getcwd()
        #joblib.dump(model['XGB'], os.path.join(cwd, "xgb_fulldata_{}.joblib.dat".format(args.suffix)))
        model['XGB'].save_model('xgb_fulldata_{}.model'.format(args.suffix))

    if use_classifiers['SVM']:
        print('Training Support Vector Machine...')
        hyper_params = {'C': 1.0, 'gamma': 1.0/input_dim}

        model['SVM'] = build_custom_model(hyper_params, 'SVM')
        model['SVM'] = train(model['SVM'], 'SVM')
        # save model to file
        joblib.dump(model['SVM'], "svm_fulldata_{}.joblib.dat".format(args.suffix))


    import matplotlib as mpl
    #mpl.use('Agg')
    mpl.use('pdf')
    from matplotlib import pyplot as plt
    from matplotlib import rc
    #.Allow for using TeX mode in matplotlib Figures
    rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern Roman']})
    rc('text', usetex=True)
    plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]

    ratio=6.0/8.0
    fig_width_pt = 3*246.0  # Get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    golden_mean = ratio if ratio != 0.0 else (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*golden_mean      # height in inches
    fig_size =  [fig_width,fig_height]

    params = {'text.usetex' : True,
            'axes.labelsize': 24,
            'font.size': 24,
            'legend.fontsize': 20,
            'xtick.labelsize': 24,
            'ytick.labelsize': 24,
            'font.family' : 'lmodern',
            'text.latex.unicode': True,
            'axes.grid' : True,
            'text.usetex': True,
            'figure.figsize': fig_size}
    plt.rcParams.update(params)


    if 'Keras' in model.keys():
        # plot loss vs epoch
        fig, (ax1, ax2) = plt.subplots(2,1,sharex=True)
        ax1.plot(history.history['loss'], label='loss')
        ax1.plot(history.history['val_loss'], label='val loss')
        ax1.legend(loc="upper right")
        ax1.set_ylabel('Loss')

        # plot accuracy vs epoch
        ax2.plot(history.history['acc'], label='acc')
        ax2.plot(history.history['val_acc'], label='val acc')
        ax2.legend(loc="lower right")
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuary')
        fig.savefig('training_results_fulldata_keras_{}.pdf'.format(args.suffix), bbox_inches='tight')



