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
                        epochs=100, 
                        batch_size=hyper_params['batch_size'], 
                        verbose=1,
                        class_weight=classWeight,
                        callbacks=[early_stopping, model_checkpoint], 
                        validation_split=0.25)
        Y_predict = model.predict(X_test)
        fpr, tpr, thresholds = roc_curve(Y_test, Y_predict, sample_weight=sampleWeight_test)
        roc_auc = roc_auc_score(Y_test, Y_predict, average='weighted')
        #roc_auc = auc(fpr, tpr)
        print("Best auc: {}".format(roc_auc))
        print("Classification Report")
        print(classification_report(Y_test, np.argmax(Y_predict, axis=1)))
        #print(classification_report(Y_test, Y_predict, sample_weight=sampleWeight_test, labels=[0, 1]))
        return model, history
        #best_acc = max(history.history['val_acc'])
        #return best_acc

    if classifier == 'GTB' or classifier == 'XGB' or classifier == 'SVM':
        model.fit(X_train_val, Y_train_val)
        if classifier == 'GTB' or classifier == 'SVM':
          Y_predict = model.decision_function(X_test)
        if classifier == 'XGB':
          Y_predict = model.predict_proba(X_test)[:,1]
        fpr, tpr, thresholds = roc_curve(Y_test, Y_predict, sample_weight=sampleWeight_test)
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

    #df['sig'] = df['sig'].sample(frac=1)[:nData]
    #df['bkg'] = df['bkg'].sample(frac=1)[:nData]

    # add isSignal variable
    df['sig']['isSignal'] = np.ones(len(df['sig']))
    df['bkg']['isSignal'] = np.zeros(len(df['bkg']))

    df_all = pd.concat([df['sig'],df['bkg']]).sort_index(axis=1).sample(frac=1).reset_index(drop=True)
    dataset = df_all.values
    X = dataset[:,0:input_dim]
    Y = dataset[:,input_dim]
    #print(Y)

    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.25, random_state=8)

    classWeight = compute_class_weight('balanced', np.unique(Y_train_val), Y_train_val) 
    classWeight = dict(enumerate(classWeight))
    print(classWeight)
    classWeight_test = compute_class_weight('balanced', np.unique(Y_test), Y_test) 
    classWeight_test = dict(enumerate(classWeight_test))
    sampleWeight_test = compute_sample_weight(classWeight_test, Y_test)

    use_classifiers = {'Keras': True, 'GTB': False, 'XGB': False, 'SVM': False}
    model = {}

    if use_classifiers['Keras']:
        print('Training Keras Neural Network...')
        early_stopping = EarlyStopping(monitor='val_loss', patience=100)

        model_checkpoint = ModelCheckpoint('dense_model_{}.h5'.format(args.suffix), monitor='val_loss', 
                                           verbose=1, save_best_only=True, 
                                           save_weights_only=False, mode='auto', 
                                           period=1)

        #hyper_params = {'hidden_layers': 3, 'initial_nodes': 64, 'l2_lambda': 10.0**-4, 'dropout': 0.25, 'batch_size': 512, 'learning_rate': 10.0**-3}
        hyper_params = {'hidden_layers': 2, 'initial_nodes': 32, 'l2_lambda': 1.3057169431397849e-05, 'dropout': 0.08351959164395392, 'batch_size': 256, 'learning_rate': 0.00039193778307251034} # PF
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
        joblib.dump(model['GTB'], "gtb_{}.joblib.dat".format(args.suffix))

    if use_classifiers['XGB']:
        print('Training XGBoost...')
        hyper_params = {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 10.0**-3}

        model['XGB'] = build_custom_model(hyper_params, 'XGB')
        model['XGB'] = train(model['XGB'], 'XGB')
        # save model to file
        joblib.dump(model['XGB'], "xgb_{}.joblib.dat".format(args.suffix))

    if use_classifiers['SVM']:
        print('Training Support Vector Machine...')
        hyper_params = {'C': 1.0, 'gamma': 1.0/input_dim}

        model['SVM'] = build_custom_model(hyper_params, 'SVM')
        model['SVM'] = train(model['SVM'], 'SVM')
        # save model to file
        joblib.dump(model['SVM'], "svm_{}.joblib.dat".format(args.suffix))


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

    fig_roc, ax_roc = plt.subplots()
    for classifier in model.keys():
        if classifier == 'Keras':
          Y_predict = model[classifier].predict(X_test)
        if classifier == 'GTB' or classifier == 'SVM':
          Y_predict = model[classifier].decision_function(X_test)
        if classifier == 'XGB':
          Y_predict = model[classifier].predict_proba(X_test)[:,1]

        fpr, tpr, thresholds = roc_curve(Y_test, Y_predict, sample_weight=sampleWeight_test, drop_intermediate=False)
        roc_auc = roc_auc_score(Y_test, Y_predict, average='weighted')
        #roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, lw=2, label='%s, AUC=%.3f'%(classifier, roc_auc))

        Y_predict = Y_predict.flatten()
        Y_output = {'ymin': Y_predict.min(), 'ymax': Y_predict.max()}

        fig_thres, ax_thres = plt.subplots()
        ax_thres.plot(thresholds, fpr, lw=2, label='%s'%(classifier))
        ax_thres.set_xlim([(Y_output['ymin']+Y_output['ymax'])/2.0, Y_output['ymax']])
        ax_thres.set_yscale('log')
        #ax_thres.set_xscale('log')
        ax_thres.set_xlabel('MVA cut')
        ax_thres.set_ylabel('False Positive Rate')
        ax_thres.set_title('FPR vs. threshold: {}'.format(classifier))
        ax_thres.legend(loc="lower left") 
        fig_thres.savefig('training_results_thres_{}_{}.pdf'.format(classifier, args.suffix), bbox_inches='tight')


        #if not (classifier == 'Keras'): continue
        fig_mvaId, ax_mvaId = plt.subplots()
        Y_sig = Y_predict[np.greater(Y_test, 0.5)]
        Y_bkg = Y_predict[np.less(Y_test, 0.5)]
        weights_sig = np.ones_like(Y_sig)/float(len(Y_sig))
        weights_bkg = np.ones_like(Y_bkg)/float(len(Y_bkg))
        ax_mvaId.hist(Y_sig, bins=np.linspace(Y_output['ymin'], Y_output['ymax'], 50), weights=weights_sig, color='b', log=True, label='Signal')
        ax_mvaId.hist(Y_bkg, bins=np.linspace(Y_output['ymin'], Y_output['ymax'], 50), weights=weights_bkg, color='r', alpha=0.5, log=True, label='Background')
        ax_mvaId.set_ylabel('a.u.')
        ax_mvaId.set_xlabel('Classifier output')
        ax_mvaId.set_title('Classifier output: {}'.format(classifier))
        ax_mvaId.legend(loc='upper center')
        fig_mvaId.savefig('training_results_mvaId_{}_{}.pdf'.format(classifier, args.suffix), bbox_inches='tight')


    ax_roc.plot(np.logspace(-3, 0, 1000), np.logspace(-4, 0, 1000), linestyle='--', lw=2, color='k', label='Random chance')
    ax_roc.set_xscale('log')
    #ax_roc.set_xlim([1.0e-3, 1.0])
    ax_roc.set_ylim([0, 1.0])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('Receiver Operating Curve')
    ax_roc.legend(loc="lower right") 
    fig_roc.savefig('training_results_roc_{}.pdf'.format(args.suffix), bbox_inches='tight')


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
        fig.savefig('training_results_keras_{}.pdf'.format(args.suffix), bbox_inches='tight')



