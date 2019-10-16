#!/usr/bin/env python

from subprocess import call
from os.path import isfile
import sys

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Dropout
from keras.regularizers import l2
from keras import initializers
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import classification_report, roc_curve, auc, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import compute_class_weight, compute_sample_weight
import xgboost as xgb
from sklearn import svm
from scipy import interp

from sklearn.externals import joblib

import numpy as np
import uproot
import pandas as pd
import h5py
import time

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
        'legend.fontsize': 10,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'font.family' : 'lmodern',
        'text.latex.unicode': True,
        'axes.grid' : True,
        'text.usetex': True,
        'figure.figsize': fig_size}
plt.rcParams.update(params)


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



def train(X_train_val, Y_train_val, X_test, Y_test, model, classifier, classWeight=None, hyper_params=None):
    if classifier == 'Keras':
        history = model.fit(X_train_val, 
                        Y_train_val, 
                        epochs=400, 
                        batch_size=hyper_params['batch_size'], 
                        verbose=1,
                        class_weight=classWeight,
                        callbacks=[early_stopping, model_checkpoint], 
                        validation_split=0.25)
        Y_predict = model.predict(X_test)
        fpr, tpr, thresholds = roc_curve(Y_test, Y_predict, drop_intermediate=False)
        roc_auc = roc_auc_score(Y_test, Y_predict, average='weighted')
        #roc_auc = auc(fpr, tpr)
        print("Best auc: {}".format(roc_auc))
        print("Classification Report")
        print(classification_report(Y_test, np.argmax(Y_predict, axis=1)))
        #print(classification_report(Y_test, Y_predict, labels=[0, 1]))
        return model, history, fpr, tpr, thresholds, roc_auc 

    if classifier == 'GTB' or classifier == 'XGB' or classifier == 'SVM':
        model.fit(X_train_val, Y_train_val)
        if classifier == 'GTB' or classifier == 'SVM':
          Y_predict = model.decision_function(X_test)
        if classifier == 'XGB':
          Y_predict = model.predict_proba(X_test)[:,1]
        fpr, tpr, thresholds = roc_curve(Y_test, Y_predict)
        roc_auc = auc(fpr, tpr)
        Y_predict = model.predict(X_test)
        print("Best auc: {}".format(roc_auc))
        print("Classification Report")
        print(classification_report(Y_test, Y_predict))
        return model


if __name__ == "__main__":
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

    #branches = ['BToKEE_l1_normpt', 'BToKEE_l1_eta', 'BToKEE_l1_phi', 'BToKEE_l1_dxy_sig', 'BToKEE_l1_dz', 'BToKEE_l1_mvaId', 'BToKEE_l1_isPF', 'BToKEE_l2_normpt', 'BToKEE_l2_eta', 'BToKEE_l2_phi', 'BToKEE_l2_dxy_sig', 'BToKEE_l2_dz', 'BToKEE_l2_mvaId', 'BToKEE_l2_isPF', 'BToKEE_k_normpt', 'BToKEE_k_eta', 'BToKEE_k_phi', 'BToKEE_k_DCASig', 'BToKEE_normpt', 'BToKEE_svprob', 'BToKEE_cos2D', 'BToKEE_l_xy_sig']
    #branches = sorted(['BToKEE_l1_normpt', 'BToKEE_l1_eta', 'BToKEE_l1_phi', 'BToKEE_l1_dxy_sig', 'BToKEE_l1_dz', 'BToKEE_l1_mvaId', 'BToKEE_l2_normpt', 'BToKEE_l2_eta', 'BToKEE_l2_phi', 'BToKEE_l2_dxy_sig', 'BToKEE_l2_dz', 'BToKEE_l2_mvaId', 'BToKEE_k_normpt', 'BToKEE_k_eta', 'BToKEE_k_phi', 'BToKEE_k_DCASig', 'BToKEE_normpt', 'BToKEE_svprob', 'BToKEE_cos2D', 'BToKEE_l_xy_sig'])
    branches = sorted(['BToKEE_l1_normpt', 'BToKEE_l1_eta', 'BToKEE_l1_phi', 'BToKEE_l1_dxy_sig', 'BToKEE_l1_dz', 'BToKEE_l2_normpt', 'BToKEE_l2_eta', 'BToKEE_l2_phi', 'BToKEE_l2_dxy_sig', 'BToKEE_l2_dz', 'BToKEE_k_normpt', 'BToKEE_k_eta', 'BToKEE_k_phi', 'BToKEE_k_DCASig', 'BToKEE_normpt', 'BToKEE_svprob', 'BToKEE_cos2D', 'BToKEE_l_xy_sig'])
    #branches = ['BToKEE_l1_normpt', 'BToKEE_l1_eta', 'BToKEE_l1_phi', 'BToKEE_l1_dxy_sig', 'BToKEE_l1_dz', 'BToKEE_l2_normpt', 'BToKEE_l2_eta', 'BToKEE_l2_phi', 'BToKEE_l2_dxy_sig', 'BToKEE_l2_dz', 'BToKEE_k_normpt', 'BToKEE_k_eta', 'BToKEE_k_phi', 'BToKEE_k_DCASig', 'BToKEE_normpt', 'BToKEE_svprob', 'BToKEE_cos2D']
    #branches = sorted(['BToKEE_svprob', 'BToKEE_cos2D', 'BToKEE_l_xy_sig'])

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

    df['sig'] = df['sig'].sample(frac=1)#[:nData]
    df['bkg'] = df['bkg'].sample(frac=1)#[:nData]

    # add isSignal variable
    df['sig']['isSignal'] = np.ones(len(df['sig']))
    df['bkg']['isSignal'] = np.zeros(len(df['bkg']))

    df_all = pd.concat([df['sig'],df['bkg']]).sort_index(axis=1).sample(frac=1).reset_index(drop=True)
    dataset = df_all.values
    X = dataset[:,0:input_dim]
    Y = dataset[:,input_dim]
    #print(Y)

    use_classifiers = {'Keras': True, 'GTB': False, 'XGB': False, 'SVM': False}
    model = {}
    tprs = {}
    aucs = {}
    figs = {}
    axs = {}

    cv = StratifiedKFold(n_splits=10, shuffle=True)
    #cv = KFold(n_splits=3, shuffle=True)

    if use_classifiers['Keras']:
        print('Initializing Keras Neural Network...')
        early_stopping = EarlyStopping(monitor='val_loss', patience=100)

        model_checkpoint = ModelCheckpoint('dense_model_{}.h5'.format(args.suffix), monitor='val_loss', 
                                           verbose=1, save_best_only=True, 
                                           save_weights_only=False, mode='auto', 
                                           period=1)

        #hyper_params = {'hidden_layers': 3, 'initial_nodes': 64, 'l2_lambda': 10.0**-4, 'dropout': 0.25, 'batch_size': 512, 'learning_rate': 10.0**-3}
        hyper_params = {'hidden_layers': 4, 'initial_nodes': 256, 'l2_lambda': 1.0e-06, 'dropout': 0.0, 'batch_size': 788, 'learning_rate': 0.002220813809955833} # PF
        #hyper_params = {'hidden_layers': 3, 'initial_nodes': 471, 'l2_lambda': 0.0009291606742300548, 'dropout': 0.3082756432562473, 'batch_size': 492, 'learning_rate': 0.00033698412975590987} # Low
        model['Keras'] = build_custom_model(hyper_params, 'Keras') 
        model['Keras'].compile(optimizer=Adam(lr=hyper_params['learning_rate']), loss='binary_crossentropy', metrics=['accuracy'], weighted_metrics=['accuracy'])
        tprs['Keras'] = []
        aucs['Keras'] = []
        figs['Keras'], axs['Keras'] = plt.subplots()


    if use_classifiers['GTB']:
        print('Initializing Gradient Boosting Tree...')
        hyper_params = {'n_estimators': 200, 'max_depth': 3, 'min_samples_split': 0.001, 'min_samples_leaf': 2, 'learning_rate': 10.0**-3}
        model['GTB'] = build_custom_model(hyper_params, 'GTB')
        tprs['GTB'] = []
        aucs['GTB'] = []

    if use_classifiers['XGB']:
        print('Initializing XGBoost...')
        hyper_params = {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 10.0**-3}
        model['XGB'] = build_custom_model(hyper_params, 'XGB')
        tprs['XGB'] = []
        aucs['XGB'] = []

    if use_classifiers['SVM']:
        print('Initializing Support Vector Machine...')
        hyper_params = {'C': 1.0, 'gamma': 1.0/input_dim}
        model['SVM'] = build_custom_model(hyper_params, 'SVM')
        tprs['SVM'] = []
        aucs['SVM'] = []



    #mean_fpr = np.linspace(0, 1, 10000)
    mean_fpr = np.logspace(-5, 0, 100)

    iFold = 0
    for train_idx, test_idx in cv.split(X, Y):
      X_train_val = X[train_idx]
      X_test = X[test_idx]
      Y_train_val = Y[train_idx]
      Y_test = Y[test_idx]

      classWeight = compute_class_weight('balanced', np.unique(Y_train_val), Y_train_val) 
      classWeight = dict(enumerate(classWeight))
      print(classWeight)

      if use_classifiers['Keras']:
          print('Training Keras Neural Network...')
          model['Keras'].summary()
          model['Keras'], history, fpr, tpr, thresholds, roc_auc = train(X_train_val, Y_train_val, X_test, Y_test, model['Keras'], 'Keras', classWeight=classWeight, hyper_params=hyper_params)
          tprs['Keras'].append(interp(mean_fpr, fpr, tpr))
          tprs['Keras'][-1][0] = 0.0
          aucs['Keras'].append(roc_auc)
          axs['Keras'].plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (iFold, roc_auc))


      if use_classifiers['GTB']:
          print('Training Gradient Boosting Tree...')
          model['GTB'] = train(X_train_val, Y_train_val, X_test, Y_test, model['GTB'], 'GTB')
          # save model to file
          joblib.dump(model['GTB'], "gtb_{}.joblib.dat".format(args.suffix))

      if use_classifiers['XGB']:
          print('Training XGBoost...')
          model['XGB'] = train(X_train_val, Y_train_val, X_test, Y_test, model['XGB'], 'XGB')
          # save model to file
          joblib.dump(model['XGB'], "xgb_{}.joblib.dat".format(args.suffix))

      if use_classifiers['SVM']:
          print('Training Support Vector Machine...')
          model['SVM'] = train(X_train_val, Y_train_val, X_test, Y_test, model['SVM'], 'SVM')
          # save model to file
          joblib.dump(model['SVM'], "svm_{}.joblib.dat".format(args.suffix))


      iFold += 1


    for classifier in model.keys():
      mean_tpr = np.mean(tprs[classifier], axis=0)
      mean_tpr[-1] = 1.0
      mean_auc = auc(mean_fpr, mean_tpr)
      std_auc = np.std(aucs[classifier])
      axs[classifier].plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
      axs[classifier].plot(np.logspace(-3, 0, 1000), np.logspace(-4, 0, 1000), linestyle='--', lw=2, color='k', label='Random chance')

      std_tpr = np.std(tprs[classifier], axis=0)
      tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
      tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
      axs[classifier].fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
      axs[classifier].set_xscale('log')
      #axs[classifier].set_xlim([1.0e-3, 1.0])
      axs[classifier].set_ylim([0, 1.0])
      axs[classifier].set_xlabel('False Positive Rate')
      axs[classifier].set_ylabel('True Positive Rate')
      axs[classifier].set_title('Cross-validation Receiver Operating Curve: {}'.format(classifier))
      #axs[classifier].legend(loc="lower right") 
      axs[classifier].legend(loc="upper left") 
      figs[classifier].savefig('training_results_roc_cv_{}_{}.pdf'.format(classifier, args.suffix), bbox_inches='tight')





