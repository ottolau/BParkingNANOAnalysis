#!/usr/bin/env python

from subprocess import call
from os.path import isfile


from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Dropout
from keras.regularizers import l2
from keras import initializers
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.ensemble import GradientBoostingClassifier
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
        model = svm.SVC(C=hyper_params['C'], gamma=hyper_params['gamma'], verbose=1)

    return model



def train(model, classifier, hyper_params=None):
    if classifier == 'Keras':
        history = model.fit(X_train_val, 
                        Y_train_val, 
                        epochs=200, 
                        batch_size=hyper_params['batch_size'], 
                        verbose=1,
                        callbacks=[early_stopping, model_checkpoint], 
                        validation_split=0.25)
        Y_predict = model.predict(X_test)
        fpr, tpr, thresholds = roc_curve(Y_test, Y_predict)
        roc_auc = auc(fpr, tpr)
        print("Best auc: {}".format(roc_auc))
        print("Classification Report")
        print(classification_report(Y_test, np.argmax(Y_predict, axis=1)))
        return model, history
        #best_acc = max(history.history['val_acc'])
        #return best_acc

    if classifier == 'GTB' or classifier == 'XGB' or classifier == 'SVM':
        model.fit(X_train_val, Y_train_val)
        Y_predict = model.predict(X_test)
        fpr, tpr, thresholds = roc_curve(Y_test, Y_predict)
        roc_auc = auc(fpr, tpr)
        print("Best auc: {}".format(roc_auc))
        print("Classification Report")
        print(classification_report(Y_test, Y_predict))
        return model


if __name__ == "__main__":

    filename = {}
    upfile = {}
    params = {}
    df = {}

    filename['bkg'] = "test_bkg.root"
    filename['sig'] = "test_sig.root"

    branches = ['BToKEE_l1_normpt', 'BToKEE_l1_eta', 'BToKEE_l1_phi', 'BToKEE_l1_dxy_sig', 'BToKEE_l1_dz', 'BToKEE_l1_mvaId', 'BToKEE_l2_normpt', 'BToKEE_l2_eta', 'BToKEE_l2_phi', 'BToKEE_l2_dxy_sig', 'BToKEE_l2_dz', 'BToKEE_l2_mvaId', 'BToKEE_k_normpt', 'BToKEE_k_eta', 'BToKEE_k_phi', 'BToKEE_k_DCASig', 'BToKEE_normpt', 'BToKEE_svprob', 'BToKEE_cos2D', 'BToKEE_l_xy_sig']

    input_dim = len(branches)

    upfile['bkg'] = uproot.open(filename['bkg'])
    upfile['sig'] = uproot.open(filename['sig'])

    params['bkg'] = upfile['bkg']['tree'].arrays(branches)
    params['sig'] = upfile['sig']['tree'].arrays(branches)

    df['sig'] = pd.DataFrame(params['sig'])#[:1000]
    df['bkg'] = pd.DataFrame(params['bkg'])#[:1000]

    # add isSignal variable
    df['sig']['isSignal'] = np.ones(len(df['sig']))
    df['bkg']['isSignal'] = np.zeros(len(df['bkg']))

    df_all = pd.concat([df['sig'],df['bkg']])
    dataset = df_all.values
    X = dataset[:,0:input_dim]
    Y = dataset[:,input_dim]

    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.25, random_state=7)

    use_classifiers = {'Keras': True, 'GTB': True, 'XGB': True, 'SVM': True}
    model = {}

    if use_classifiers['Keras']:
        print('Training Keras Neural Network...')
        early_stopping = EarlyStopping(monitor='val_loss', patience=30)

        model_checkpoint = ModelCheckpoint('dense_model.h5', monitor='val_loss', 
                                           verbose=1, save_best_only=True, 
                                           save_weights_only=False, mode='auto', 
                                           period=1)

        hyper_params = {'hidden_layers': 3, 'initial_nodes': 64, 'l2_lambda': 10.0**-4, 'dropout': 0.25, 'batch_size': 512, 'learning_rate': 10.0**-3}

        model['Keras'] = build_custom_model(hyper_params, 'Keras') 
        model['Keras'].compile(optimizer=Adam(lr=hyper_params['learning_rate']), loss='binary_crossentropy', metrics=['accuracy'])
        model['Keras'].summary()
        model['Keras'], history = train(model['Keras'], 'Keras', hyper_params)


    if use_classifiers['GTB']:
        print('Training Gradient Boosting Tree...')
        hyper_params = {'n_estimators': 200, 'max_depth': 3, 'min_samples_split': 0.001, 'min_samples_leaf': 2, 'learning_rate': 10.0**-3}

        model['GTB'] = build_custom_model(hyper_params, 'GTB')
        model['GTB'] = train(model['GTB'], 'GTB')
        # save model to file
        joblib.dump(model['GTB'], "gtb.joblib.dat")

    if use_classifiers['XGB']:
        print('Training XGBoost...')
        hyper_params = {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 10.0**-3}

        model['XGB'] = build_custom_model(hyper_params, 'XGB')
        model['XGB'] = train(model['XGB'], 'XGB')
        # save model to file
        joblib.dump(model['XGB'], "xgb.joblib.dat")

    if use_classifiers['SVM']:
        print('Training Support Vector Machine...')
        hyper_params = {'C': 1.0, 'gamma': 1.0/input_dim}

        model['SVM'] = build_custom_model(hyper_params, 'SVM')
        model['SVM'] = train(model['SVM'], 'SVM')
        # save model to file
        joblib.dump(model['SVM'], "svm.joblib.dat")


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

    plt.figure()
    for classifier in model.keys():
        Y_predict = model[classifier].predict(X_test)
        fpr, tpr, thresholds = roc_curve(Y_test, Y_predict)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label='%s, AUC=%.3f'%(classifier, roc_auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Random chance')
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Curve')
    plt.legend(loc="lower right") 
    plt.savefig('test_roc.pdf', bbox_inches='tight')

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
        fig.savefig('test_keras.pdf', bbox_inches='tight')



