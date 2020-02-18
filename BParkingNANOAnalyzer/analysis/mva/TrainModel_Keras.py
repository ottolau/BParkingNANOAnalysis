import matplotlib as mpl
mpl.use('pdf')
from matplotlib import pyplot as plt
from matplotlib import rc
#.Allow for using TeX mode in matplotlib Figures
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern Roman']})
rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]

ratio=5.0/7.0
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
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'font.family' : 'lmodern',
        'text.latex.unicode': True,
        'axes.grid' : True,
        'text.usetex': True,
        'figure.figsize': fig_size}
plt.rcParams.update(params)

import uproot
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import classification_report, roc_curve, auc, roc_auc_score
from sklearn import metrics
import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_evaluations, plot_objective
from scipy import interp
import h5py
import time
import ROOT
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
from sklearn.utils import compute_class_weight

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


def get_df(root_file_name, branches):
    f = uproot.open(root_file_name)
    if len(f.allkeys()) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(uproot.open(root_file_name)["tree"].arrays(branches))
    return df

def get_label(name):
    if name == 0:
        return "background"
    else:
        return "signal"

def plot_roc_curve(df, score_column, tpr_threshold=0.0, ax=None, color=None, linestyle='-', label=None):
    print('Plotting ROC...')
    if ax is None:
        ax = plt.gca()
    if label is None:
        label = score_column
    fpr, tpr, thresholds = roc_curve(df["isSignal"], df[score_column], drop_intermediate=True)
    roc_auc = roc_auc_score(df["isSignal"], df[score_column])
    roc_pauc = roc_auc_score(df["isSignal"], df[score_column], max_fpr=1.0e-2)
    print("auc: {}, pauc: {}".format(roc_auc, roc_pauc))
    mask = tpr > tpr_threshold
    fpr, tpr = fpr[mask], tpr[mask]
    ax.plot(fpr, tpr, label=label, color=color, linestyle=linestyle)

'''
space  = [Integer(2, 4, name='hidden_layers'),
          Integer(32, 256, name='initial_nodes'),
          Real(10**-5, 10**-1, "log-uniform", name='l2_lambda'),
          Real(0.15,0.5,name='dropout'),
          Integer(256,4096,name='batch_size'),
          Real(10**-5, 10**-1, "log-uniform", name='learning_rate'),
          ]
'''
space  = [Real(10**-5, 10**-1, "log-uniform", name='l2_lambda'),
          Real(0.05,0.5,name='dropout'),
          Integer(128,2048,name='batch_size'),
          Real(10**-5, 10**-1, "log-uniform", name='learning_rate'),
          ]

def pauc(predt, dtrain):
    y = dtrain.get_label()
    return 'pauc', roc_auc_score(y, predt, max_fpr=1.0e-2)

def build_custom_model(num_hiddens=3, initial_node=32, 
                          dropout=0.20, l2_lambda=1.e-5):
    inputs = Input(shape=(input_dim,), name = 'input')
    for i in range(num_hiddens):
        #hidden = Dense(units=int(round(initial_node/np.power(2,i))), kernel_initializer='glorot_normal', activation='relu', kernel_regularizer=l2(l2_lambda))(inputs if i==0 else hidden)
        hidden = Dense(units=int(initial_node), kernel_initializer='glorot_normal', activation='relu', kernel_regularizer=l2(l2_lambda))(inputs if i==0 else hidden)
        hidden = Dropout(np.float32(dropout))(hidden)
    outputs = Dense(1, name = 'output', kernel_initializer='normal', activation='sigmoid')(hidden)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def train(X_train_val, Y_train_val, X_test, Y_test, model, classWeight, batch_size=64):
    history = model.fit(X_train_val, Y_train_val, epochs=200, batch_size=batch_size, verbose=0, class_weight=classWeight, callbacks=[early_stopping, model_checkpoint], validation_split=0.25)
    Y_predict = model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(Y_test, Y_predict)
    roc_auc = roc_auc_score(Y_test, Y_predict, max_fpr=1.0e-2)
    return model, fpr, tpr, thresholds, roc_auc, history

def train_cv(X_train, y_train, model, batch_size=64):
    fprs = []
    tprs = []
    aucs = []
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    for train_idx, test_idx in cv.split(X_train, y_train):
      X_train_val = X_train.iloc[train_idx]
      X_test = X_train.iloc[test_idx]
      Y_train_val = y_train.iloc[train_idx]
      Y_test = y_train.iloc[test_idx]
      classWeight = compute_class_weight('balanced', np.unique(Y_train_val), Y_train_val) 
      classWeight = dict(enumerate(classWeight))
      _, fpr, tpr, _, auc, _ = train(X_train_val, Y_train_val, X_test, Y_test, model, classWeight, batch_size=batch_size)
      fprs.append(fpr)
      tprs.append(tpr)
      aucs.append(auc)
    return fprs, tprs, aucs

@use_named_args(space)
def objective(**X):
    global best_auc, best_auc_std, best_params
    print("New configuration: {}".format(X))
    #model = build_custom_model(num_hiddens=X['hidden_layers'], initial_node=X['initial_nodes'], dropout=X['dropout'], l2_lambda=X['l2_lambda'])
    model = build_custom_model(num_hiddens=num_hiddens, initial_node=initial_node, dropout=X['dropout'], l2_lambda=X['l2_lambda'])
    model.compile(optimizer=Adam(lr=X['learning_rate']), loss='binary_crossentropy', metrics=['accuracy'], weighted_metrics=['accuracy'])
    model.summary()
    _, _, aucs = train_cv(X_train, y_train, model, X['batch_size'])
    ave_auc = sum(aucs)/float(len(aucs))
    ave_auc_std = np.std(aucs)
    if ave_auc > best_auc:
      best_auc = ave_auc
      best_auc_std = ave_auc_std
      best_params = X.copy()
    print("Best pauc: {}+-{}, Best configuration: {}".format(best_auc, best_auc_std, best_params))
    return -ave_auc

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="A simple ttree plotter")
    parser.add_argument("-s", "--signal", dest="signal", default="RootTree_2020Jan16_BuToKee_BToKEEAnalyzer_2020Feb14_fullq2_EB_pf.root", help="Signal file")
    parser.add_argument("-b", "--background", dest="background", default="RootTree_2020Jan16_Run2018ABCDpartial_BToKEEAnalyzer_2020Feb14_fullq2_EB_upSB_pf_partial.root", help="Background file")
    parser.add_argument("-f", "--suffix", dest="suffix", default=None, help="Suffix of the output name")
    parser.add_argument("-o", "--optimization", dest="optimization", action='store_true', help="Perform Bayesian optimization")
    args = parser.parse_args()

    
    features = ['BToKEE_fit_l1_normpt', 'BToKEE_fit_l1_eta', 'BToKEE_l1_dxy_sig', 'BToKEE_l1_dz',
                'BToKEE_fit_l2_normpt', 'BToKEE_fit_l2_eta', 'BToKEE_l2_dxy_sig', 'BToKEE_l2_dz', 
                'BToKEE_fit_k_normpt', 'BToKEE_fit_k_eta', 'BToKEE_k_DCASig', 'BToKEE_k_dz',
                'BToKEE_fit_normpt', 'BToKEE_fit_eta', 'BToKEE_svprob', 'BToKEE_fit_cos2D', 'BToKEE_l_xy_sig',
                ]
    #features += ['BToKEE_fit_l1_phi', 'BToKEE_fit_l2_phi', 'BToKEE_fit_k_phi', 'BToKEE_fit_phi']
    features += ['BToKEE_fit_l1_dphi', 'BToKEE_fit_l2_dphi', 'BToKEE_fit_k_dphi', 'BToKEE_fit_dphi']
    features += ['BToKEE_l1_iso04_rel', 'BToKEE_l2_iso04_rel', 'BToKEE_k_iso04_rel', 'BToKEE_b_iso04_rel']
    #features += ['BToKEE_l1_iso03_rel', 'BToKEE_l2_iso03_rel', 'BToKEE_k_iso03_rel', 'BToKEE_b_iso03_rel']
    features += ['BToKEE_l1_pfmvaId_lowPt', 'BToKEE_l2_pfmvaId_lowPt', 'BToKEE_l1_pfmvaId_highPt', 'BToKEE_l2_pfmvaId_highPt']
    #features += ['BToKEE_ptImbalance']
    #features += ['BToKEE_trg_eta']
    #features += ['BToKEE_l1_mvaId', 'BToKEE_l2_mvaId']
   

    features = sorted(features)
    input_dim = len(features)
    branches = features + ['BToKEE_fit_massErr']

    ddf = {}
    ddf['sig'] = get_df(args.signal, branches)
    ddf['bkg'] = get_df(args.background, branches)

    ddf['sig'].replace([np.inf, -np.inf], 10.0**+10, inplace=True)
    ddf['bkg'].replace([np.inf, -np.inf], 10.0**+10, inplace=True)

    nSig = ddf['sig'].shape[0]
    nBkg = 100000
    #nSig = 10000
    #nBkg = 10000
    ddf['sig'] = ddf['sig'].sample(frac=1)[:nSig]
    ddf['bkg'] = ddf['bkg'].sample(frac=1)[:nBkg]

    # add isSignal variable
    ddf['sig']['isSignal'] = 1
    ddf['bkg']['isSignal'] = 0

    df = pd.concat([ddf['sig'],ddf['bkg']]).sort_index(axis=1).sample(frac=1).reset_index(drop=True)
    #df['weights'] = np.where(df['isSignal'], 1.0/df['BToKEE_fit_massErr'].replace(np.nan, 1.0), 1.0)
    df['weights'] = 1.0

    X = df[features]
    y = df['isSignal']
    W = df['weights']

    suffix = args.suffix
    n_calls = 120
    n_random_starts = 60
    num_hiddens = 4
    initial_node = 128
    do_bo = args.optimization
    do_cv = True
    best_params = {'l2_lambda': 2.2415890894627853e-05, 'dropout': 0.3725777915006973, 'learning_rate': 0.028325266337641073, 'batch_size': 2048}

    early_stopping = EarlyStopping(monitor='val_loss', patience=75)

    model_checkpoint = ModelCheckpoint('dense_model_{}.h5'.format(suffix), monitor='val_loss', 
                                       verbose=0, save_best_only=True, 
                                       save_weights_only=False, mode='auto', 
                                       period=1)

    # split X and y up in train and test samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20)

    # Get the number of positive and nevative training examples in this category
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)

    print("training on {0} signal and {1} background".format(n_pos, n_neg))
   
    # get the indices correspondng to the testing and training samples
    idx_train = X_train.index
    idx_test = X_test.index

    w_train = W.loc[idx_train]

    # Bayesian optimization
    if do_bo:
        begt = time.time()
        print("Begin Bayesian optimization")
        best_auc = 0.0
        best_auc_std = 0.0
        best_params = {}
        res_gp = gp_minimize(objective, space, n_calls=n_calls, n_random_starts=n_random_starts, verbose=True, random_state=36)
        print("Finish optimization in {}s".format(time.time()-begt))
        plt.figure()
        plot_convergence(res_gp)
        plt.savefig('training_resultis_bo_convergencePlot_xgb_{}.pdf'.format(suffix))

    # Get the cv plots with the best hyper-parameters
    if do_bo or do_cv:
        print("Get the cv plots with the best hyper-parameters")
        model = build_custom_model(num_hiddens=num_hiddens, initial_node=initial_node, dropout=best_params['dropout'], l2_lambda=best_params['l2_lambda'])
        model.compile(optimizer=Adam(lr=best_params['learning_rate']), loss='binary_crossentropy', metrics=['accuracy'], weighted_metrics=['accuracy'])
        model.summary()
        fpr, tpr, aucs = train_cv(X_train, y_train, model, best_params['batch_size'])
        tprs = []
        figs, axs = plt.subplots()
        mean_fpr = np.logspace(-6, 0, 100)
        for iFold in range(5):
            tprs.append(interp(mean_fpr, fpr[iFold], tpr[iFold]))
            tprs[-1][0] = 0.0
            axs.plot(fpr[iFold], tpr[iFold], lw=1, alpha=0.3, label='ROC fold %d (PAUC = %0.2f)' % (iFold, aucs[iFold]))

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        axs.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
        axs.plot(np.logspace(-5, 0, 1000), np.logspace(-5, 0, 1000), linestyle='--', lw=2, color='k', label='Random chance')

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        axs.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
        axs.set_xscale('log')
        axs.set_xlim([1.0e-5, 1.0])
        axs.set_ylim([0.0, 1.0])
        axs.set_xlabel('False Alarm Rate')
        axs.set_ylabel('Signal Efficiency')
        axs.set_title('Cross-validation Receiver Operating Curve')
        axs.legend(loc="lower right") 
        figs.savefig('training_results_roc_cv_{}.pdf'.format(suffix), bbox_inches='tight')

    keras_params = best_params.copy()

    # Re-train the whole dataset with the best hyper-parameters (without doing any cross validation)
    print('Training full model...')
    model = build_custom_model(num_hiddens=num_hiddens, initial_node=initial_node, dropout=keras_params['dropout'], l2_lambda=keras_params['l2_lambda'])
    model.compile(optimizer=Adam(lr=keras_params['learning_rate']), loss='binary_crossentropy', metrics=['accuracy'], weighted_metrics=['accuracy'])
    model.summary()
    classWeight = compute_class_weight('balanced', np.unique(y_train), y_train) 
    classWeight = dict(enumerate(classWeight))

    model, fpr, tpr, thresholds, roc_auc, history  = train(X_train, y_train, X_test, y_test, model, classWeight, batch_size=keras_params['batch_size'])
  
    print("")
    
    df.loc[idx_train, "score"] = model.predict(X_train)
    df.loc[idx_test, "score"] = model.predict(X_test)
   
    df.loc[idx_train, "test"] = False
    df.loc[idx_test, "test"] = True

    print("")
    print("Final model: Best hyper-parameters: {}".format(best_params))
    print("")

    df_train = df[np.logical_not(df['test'])]
    df_test = df[df['test']]

    fpr, tpr, thresholds = roc_curve(df_test["isSignal"], df_test["score"], drop_intermediate=True)
    roc_dict = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
    roc_df = pd.DataFrame(data=roc_dict)
    roc_df.to_csv('training_results_roc_csv_{}.csv'.format(suffix))

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
    fig.savefig('training_results_learning_curve_{}.pdf'.format(args.suffix), bbox_inches='tight')


    fig, ax = plt.subplots()
    plot_roc_curve(df_test, "score", ax=ax, label="Keras")
    ax.plot(np.logspace(-5, 0, 1000), np.logspace(-5, 0, 1000), linestyle='--', color='k')
    ax.set_xlim([1.0e-5, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xscale('log')
    ax.set_xlabel("False Alarm Rate")
    ax.set_ylabel("Signal Efficiency")
    ax.set_title('Receiver Operating Curve')
    ax.legend(loc='lower right')
    fig.savefig('training_results_roc_curve_{}.pdf'.format(suffix), bbox_inches='tight')

