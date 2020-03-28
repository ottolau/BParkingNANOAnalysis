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
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import classification_report, roc_curve, auc, roc_auc_score
import lightgbm as lgb
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
import os, sys
from collections import OrderedDict
import PyPDF2

def get_df(root_file_name, branches):
    f = uproot.open(root_file_name)
    if len(f.allkeys()) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(uproot.open(root_file_name)["tree"].arrays(branches))
    #df = pd.DataFrame(uproot.open(root_file_name)["tree"].arrays(branches, namedecode="utf-8"))
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
    #ax.semilogx(fpr, tpr, label=label, color=color, linestyle=linestyle)
    ax.plot(fpr, tpr, label=label, color=color, linestyle=linestyle)

def get_bins_center(column, bins):
    n_bins = bins.shape[0] 
    bin_centers = (bins[:-1] + bins[1:])/2.
    bin_indices = np.digitize(column, bins) - 1
    bin_indices[bin_indices == (n_bins-1)] = n_bins - 2
    return bin_centers[bin_indices]

def get_working_points(df, score_column):
    
    working_points = {}

    df_test = df[df['test']]
    signal_mask = df_test["isSignal"] == 1
    df_sig = df_test[signal_mask]
    df_bkg = df_test[~signal_mask]

    wp60, wp80, wp90 = np.percentile(df_sig[score_column], [40., 20., 10.])

    wp60_bkg_eff = 1.*len(df_bkg[df_bkg[score_column] >= wp60])/len(df_bkg)
    wp80_bkg_eff = 1.*len(df_bkg[df_bkg[score_column] >= wp80])/len(df_bkg)
    wp90_bkg_eff = 1.*len(df_bkg[df_bkg[score_column] >= wp90])/len(df_bkg)
   
    working_points["wp60"] = wp60
    working_points["wp80"] = wp80
    working_points["wp90"] = wp90
   
    print("sig. efficiency at 60 % false alarm rate: {0:.2e} %. WP: {1}".format(wp60_bkg_eff * 100, wp60))
    print("sig. efficiency at 80 % false alarm rate: {0:.2e} %. WP: {1}".format(wp80_bkg_eff * 100, wp80))
    print("sig. efficiency at 90 % false alarm rate: {0:.2e} %. WP: {1}".format(wp90_bkg_eff * 100, wp90))
    return working_points

def get_efficiency(df, score_column, working_point, isSignal):
    df = df.query("isSignal == 1") if isSignal else df.query("isSignal == 0")
    k = len(df[df[score_column] >= working_point])
    n = len(df)
    return 1.*k/n if n != 0 else np.nan

def get_efficiency_unc(df, score_column, working_point, isSignal, bUpper):
    df = df.query("isSignal == 1") if isSignal else df.query("isSignal == 0")
    k = len(df[df[score_column] >= working_point])
    n = len(df)
    teff = ROOT.TEfficiency()
    return teff.Bayesian(n, k, 0.683, 1.0, 1.0, bUpper, True) if n != 0 else np.nan

def plot_turnon_curve(df_group, group, working_points, label, isSignal=True, ax=None):
    if ax is None:
        ax = plt.gca()
    cmap = {'wp60': 'b', 'wp80': 'g', 'wp90': 'r'}
    for name, wp in OrderedDict(sorted(working_points.items())).iteritems():
      wp_eff = df_group.groupby(group).apply(lambda df : get_efficiency(df, "score", wp, isSignal))
      wp_eff_unc_upper = df_group.groupby(group).apply(lambda df : get_efficiency_unc(df, "score", wp, isSignal=isSignal, bUpper=True))
      wp_eff_unc_lower = df_group.groupby(group).apply(lambda df : get_efficiency_unc(df, "score", wp, isSignal=isSignal, bUpper=False))
      wp_eff.plot(label=name, ax=ax, marker='.', color=cmap[name])
      ax.fill_between(wp_eff_unc_upper.index.values, wp_eff_unc_lower.values, wp_eff_unc_upper.values, alpha=0.3, color=cmap[name])
    if isSignal:
      ax.set_ylabel("Signal Efficiency", fontsize=12)
      ax.set_xlabel("")
      ax.xaxis.set_tick_params(bottom=False, labelbottom=False)
      ax.legend(loc='lower right')
    else:
      ax.set_yscale('log')
      ax.set_ylabel("False Alarm Rate", fontsize=12)
      ax.set_xlabel(label)
      ax.set_ylim(ymax=0.9)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.grid(True)

def pdf_combine(pdf_list, outputfile):
    merger = PyPDF2.PdfFileMerger()
    for pdf in pdf_list:
        merger.append(pdf)
    merger.write(outputfile)

def learning_rate_010_decay_power_099(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate  * np.power(.99, current_iter)
    return lr if lr > 1e-3 else 1e-3

def learning_rate_010_decay_power_0995(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate  * np.power(.995, current_iter)
    return lr if lr > 1e-3 else 1e-3

def learning_rate_005_decay_power_099(current_iter):
    base_learning_rate = 0.05
    lr = base_learning_rate  * np.power(.99, current_iter)
    return lr if lr > 1e-3 else 1e-3

def fpreproc(dtrain, dtest, param):
    label = dtrain.get_label()
    ratio = float(np.sum(label == 0)) / np.sum(label == 1)
    param['scale_pos_weight'] = ratio
    return (dtrain, dtest, param)

def pauc(predt, dtrain):
    y = dtrain.get_label()
    return 'pauc', roc_auc_score(y, predt, max_fpr=1.0e-2), True

space  = [Integer(100, 500, name='num_leaves'),
         Real(10**-5, 10**+3, "log-uniform", name='min_child_weight'),
         Integer(0.0, 500.0, name='min_child_samples'),
         Real(0.5, 1.0, name='subsample'),
         Real(0.1, 1.0, name='colsample_bytree'),
         Real(0.0, 10.0, name='lambda_l1'),
         Real(0.0, 10.0, name='lambda_l2'),
         ]

@use_named_args(space)
def objective(**X):
    global best_auc, best_auc_std, best_params
    print("New configuration: {}".format(X))
    params = X.copy()
    params['learning_rate'] = 0.05
    params['objective'] = 'binary'
    #params['eval_metric'] = 'auc'
    params['nthread'] = 6
    params['verbose'] = -1
    #cv_result = lgb.cv(params, dmatrix_train, num_boost_round=n_boost_rounds, nfold=5, shuffle=True, stratified=True, early_stopping_rounds=75, fpreproc=fpreproc, feval=pauc)
    cv_result = lgb.cv(params, dmatrix_train, num_boost_round=n_boost_rounds, nfold=5, shuffle=True, stratified=True, early_stopping_rounds=100, feval=pauc)
    ave_auc = cv_result['pauc-mean'][-1]
    ave_auc_std = cv_result['pauc-stdv'][-1]
    print("Average pauc: {}+-{}".format(ave_auc, ave_auc_std))
    if ave_auc > best_auc:
      best_auc = ave_auc
      best_auc_std = ave_auc_std
      best_params = X.copy()
    print("Best pauc: {}+-{}, Best configuration: {}".format(best_auc, best_auc_std, best_params))
    return -ave_auc

def train(xgtrain, xgtest, hyper_params=None, cv=False):
    params = hyper_params.copy()
    label = xgtrain.get_label()
    ratio = float(np.sum(label == 0)) / np.sum(label == 1)
    params['scale_pos_weight'] = ratio
    params['objective'] = 'binary'
    #params['eval_metric'] = 'auc'
    params['nthread'] = 10
    params['verbose'] = -1
    results = {}
    if cv:
      params['learning_rate'] = 0.05
      model = lgb.train(params, xgtrain, num_boost_round=n_boost_rounds, valid_sets=xgtest, evals_result=results, early_stopping_rounds=100, verbose_eval=False, feval=pauc)
    else:
      model = lgb.train(params, xgtrain, num_boost_round=n_boost_rounds, valid_sets=[xgtest, xgtrain], valid_names=['eval', 'train'], evals_result=results, early_stopping_rounds=100, verbose_eval=False, feval=pauc, callbacks=[lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_0995)])
    #model = lgb.train(params, xgtrain, num_boost_round=n_boost_rounds, early_stopping_rounds=75, verbose_eval=False, feval=pauc)
    best_iteration = model.best_iteration + 1
    if best_iteration < n_boost_rounds:
        print("early stopping after {0} boosting rounds".format(best_iteration))
    return model, results

def train_cv(X_train_val, Y_train_val, X_test, Y_test, w_train_val, hyper_params=None):
    xgtrain = lgb.Dataset(X_train_val, label=Y_train_val, weight=w_train_val)
    xgtest  = lgb.Dataset(X_test , label=Y_test, reference=xgtrain)
    model, results = train(xgtrain, xgtest, hyper_params=hyper_params, cv=True)
    Y_predict = model.predict(X_test, raw_score=True)
    fpr, tpr, thresholds = roc_curve(Y_test, Y_predict, drop_intermediate=True)
    roc_auc = roc_auc_score(Y_test, Y_predict, max_fpr=1.0e-2)
    print("Best pauc: {}".format(roc_auc))
    return model, fpr, tpr, thresholds, roc_auc, results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="A simple ttree plotter")
    parser.add_argument("-s", "--signal", dest="signal", default="RootTree_2020Jan16_BuToKee_all_BToKEEAnalyzer_2020Mar06_fullq2_low.root", help="Signal file")
    parser.add_argument("-b", "--background", dest="background", default="RootTree_2020Jan16_Run2018ABCD_random30_BToKEEAnalyzer_2020Mar22_fullq2_upSB_low.root", help="Background file")
    parser.add_argument("-f", "--suffix", dest="suffix", default=None, help="Suffix of the output name")
    parser.add_argument("-o", "--optimization", dest="optimization", action='store_true', help="Perform Bayesian optimization")
    args = parser.parse_args()

    
    features = ['BToKEE_fit_l1_normpt', 'BToKEE_l1_dxy_sig',
                'BToKEE_fit_l2_normpt', 'BToKEE_l2_dxy_sig',
                'BToKEE_fit_k_normpt', 'BToKEE_k_DCASig',
                'BToKEE_fit_normpt', 'BToKEE_svprob', 'BToKEE_fit_cos2D', 'BToKEE_l_xy_sig', 'BToKEE_dz',
                ]
    features += ['BToKEE_minDR', 'BToKEE_maxDR']
    features += ['BToKEE_l1_iso04_rel', 'BToKEE_l2_iso04_rel', 'BToKEE_k_iso04_rel', 'BToKEE_b_iso04_rel']
    #features += ['BToKEE_l1_pfmvaId_lowPt', 'BToKEE_l2_pfmvaId_lowPt', 'BToKEE_l1_pfmvaId_highPt', 'BToKEE_l2_pfmvaId_highPt']
    features += ['BToKEE_ptImbalance']
    features += ['BToKEE_l1_mvaId', 'BToKEE_l2_mvaId']
   

    features = sorted(features)
    branches = features + ['BToKEE_fit_massErr', 'BToKEE_fit_pt', 'BToKEE_fit_eta', 'BToKEE_q2']

    ddf = {}
    ddf['sig'] = get_df(args.signal, branches)
    ddf['bkg'] = get_df(args.background, branches)

    ddf['sig'].replace([np.inf, -np.inf], 0.0, inplace=True)
    ddf['bkg'].replace([np.inf, -np.inf], 0.0, inplace=True)

    #nSig = ddf['sig'].shape[0]
    #nBkg = 300000
    nSig = 300000
    nBkg = 300000
    ddf['sig'] = ddf['sig'].sample(frac=1)[:nSig]
    ddf['bkg'] = ddf['bkg'].sample(frac=1)[:nBkg]

    # add isSignal variable
    ddf['sig']['isSignal'] = 1
    ddf['bkg']['isSignal'] = 0

    df = pd.concat([ddf['sig'],ddf['bkg']]).sort_index(axis=1).sample(frac=1).reset_index(drop=True)
    df['weights'] = np.where(df['isSignal'], 1.0/df['BToKEE_fit_massErr'].replace(np.nan, 1.0), 1.0)
    #df['weights'] = 1.0

    X = df[features]
    y = df['isSignal']
    W = df['weights']

    suffix = args.suffix
    n_boost_rounds = 5000
    n_calls = 80
    n_random_starts = 40
    do_bo = args.optimization
    do_cv = True
    best_params = {'num_leaves': 119, 'colsample_bytree': 0.462895782036834, 'subsample': 0.9565005845248696, 'lambda_l1': 0.642535538842698, 'min_child_samples': 43, 'lambda_l2': 4.4071882417829045, 'min_child_weight': 182.12727208918622}

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

    dmatrix_train = lgb.Dataset(X_train.copy(), label=np.copy(y_train), feature_name=[f.replace('_','-') for f in features], weight=np.copy(w_train))
    dmatrix_test  = lgb.Dataset(X_test.copy(), label=np.copy(y_test), reference=dmatrix_train, feature_name=[f.replace('_','-') for f in features])

    # Bayesian optimization
    if do_bo:
        begt = time.time()
        print("Begin Bayesian optimization")
        best_auc = 0.0
        best_auc_std = 0.0
        best_params = {}
        res_gp = gp_minimize(objective, space, n_calls=n_calls, n_random_starts=n_random_starts, verbose=True, random_state=36)
        print("Finish optimization in {}s".format(time.time()-begt))
        #plt.figure()
        #plot_convergence(res_gp)
        #plt.savefig('lgb_training_resultis_bo_convergencePlot_lgb_{}.pdf'.format(suffix))

    # Get the cv plots with the best hyper-parameters
    if do_bo or do_cv:
        print("Get the cv plots with the best hyper-parameters")
        tprs = []
        aucs = []
        figs, axs = plt.subplots()
        #cv = KFold(n_splits=5, shuffle=True)
        cv = StratifiedKFold(n_splits=5, shuffle=True)
        mean_fpr = np.logspace(-7, 0, 100)

        iFold = 0
        for train_idx, test_idx in cv.split(X_train, y_train):
            X_train_cv = X_train.iloc[train_idx]
            X_test_cv = X_train.iloc[test_idx]
            Y_train_cv = y_train.iloc[train_idx]
            Y_test_cv = y_train.iloc[test_idx]
            w_train_cv = w_train.loc[X_train_cv.index]

            model, fpr, tpr, thresholds, roc_auc, results = train_cv(X_train_cv, Y_train_cv, X_test_cv, Y_test_cv, w_train_cv, hyper_params=best_params)

            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            aucs.append(roc_auc)
            axs.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (PAUC = %0.2f)' % (iFold, roc_auc))
            iFold += 1

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        axs.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
        axs.plot(np.logspace(-6, 0, 1000), np.logspace(-6, 0, 1000), linestyle='--', lw=2, color='k', label='Random chance')

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        axs.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
        axs.set_xscale('log')
        axs.set_xlim([1.0e-6, 1.0])
        axs.set_ylim([0.0, 1.0])
        axs.set_xlabel('False Alarm Rate')
        axs.set_ylabel('Signal Efficiency')
        axs.set_title('Cross-validation Receiver Operating Curve')
        axs.legend(loc="lower right") 
        figs.savefig('lgb_training_results_roc_cv_{}.pdf'.format(suffix), bbox_inches='tight')


    xgboost_params = best_params.copy()

    # Re-train the whole dataset with the best hyper-parameters (without doing any cross validation)
    print('Training full model...')
    model, results = train(dmatrix_train, dmatrix_test, hyper_params=xgboost_params)
  
    # We want to know if and when the training was early stopped.
    # `best_iteration` counts the first iteration as zero, so we increase by one.
    best_iteration = model.best_iteration + 1
    if best_iteration < n_boost_rounds:
        print("Final model: early stopping after {0} boosting rounds".format(best_iteration))
    print("")
    
    model.save_model("lgb_fulldata_{}.model".format(suffix))

    df.loc[idx_train, "score"] = model.predict(X_train, raw_score=True)
    df.loc[idx_test, "score"] = model.predict(X_test, raw_score=True)
   
    df.loc[idx_train, "test"] = False
    df.loc[idx_test, "test"] = True

    print("")
    print("Final model: Best hyper-parameters: {}, ntree_limit: {}".format(best_params, model.best_iteration))
    print("")

    #df_train = df.query("not test")
    #df_test = df.query("test")
    df_train = df[np.logical_not(df['test'])]
    df_test = df[df['test']]
    #df_test.to_csv('training_results_testdf_{}.csv'.format(suffix))

    fpr, tpr, thresholds = roc_curve(df_test["isSignal"], df_test["score"], drop_intermediate=True)
    roc_dict = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
    roc_df = pd.DataFrame(data=roc_dict)
    roc_df.to_csv('lgb_training_results_roc_csv_{}.csv'.format(suffix))


    fig, ax = plt.subplots()
    plot_roc_curve(df_test, "score", ax=ax, label="LightGBM")
    ax.plot(np.logspace(-6, 0, 1000), np.logspace(-6, 0, 1000), linestyle='--', color='k')
    ax.set_xlim([1.0e-6, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xscale('log')
    ax.set_xlabel("False Alarm Rate")
    ax.set_ylabel("Signal Efficiency")
    ax.set_title('Receiver Operating Curve')
    ax.legend(loc='lower right')
    fig.savefig('lgb_training_results_roc_curve_{}.pdf'.format(suffix), bbox_inches='tight')

    plt.figure()
    lgb.plot_importance(model)
    plt.savefig('lgb_training_results_feature_importance_{}.pdf'.format(suffix), bbox_inches='tight')

    plt.figure()
    lgb.plot_metric(results, metric='pauc', ylabel='PACU')
    plt.savefig('lgb_training_results_learning_curve_{}.pdf'.format(suffix), bbox_inches='tight')

    split_value_hist = []
    plt.figure()
    for feature in features:
      feature = feature.replace('_','-')
      outputname = 'lgb_training_results_split_value_histogram_{}_{}.pdf'.format(feature, suffix)
      try:
        lgb.plot_split_value_histogram(model, feature)
        plt.savefig(outputname, bbox_inches='tight')
        plt.cla()
        split_value_hist.append(outputname)
      except ValueError:
        pass

    pdf_combine(split_value_hist, 'lgb_training_results_split_value_histogram_{}.pdf'.format(suffix))
    map(lambda x: os.system('rm {}'.format(x)), split_value_hist)


    print("Finding working points for new training:")
    working_points = get_working_points(df, "score")
    print("")

    n_pt_bins = 100
    pt_bins = np.linspace(3, 60, n_pt_bins)
    #pt_bins = np.concatenate((np.linspace(2, 5, n_pt_bins/2, endpoint=False), np.linspace(5, 20, n_pt_bins/2)))
    df["pt_binned"] = get_bins_center(df["BToKEE_fit_pt"], pt_bins)

    n_eta_bins = 100
    eta_bins = np.linspace(-2.5, 2.5, n_eta_bins)
    df["eta_binned"] = get_bins_center(df["BToKEE_fit_eta"], eta_bins)

    n_q2_bins = 100
    q2_bins = np.linspace(0.045, 14.8, n_q2_bins)
    df["q2_binned"] = get_bins_center(df["BToKEE_q2"], q2_bins)

    n_mvaId_bins = 100
    mvaId_bins = np.linspace(0.0, 10.0, n_mvaId_bins)
    df["mvaId_binned"] = get_bins_center(df["BToKEE_l2_mvaId"], mvaId_bins)

    df_test = df[df["test"]]

    fig_pt, axes_pt = plt.subplots(2, 1)
    plot_turnon_curve(df_test, 'pt_binned', working_points, r'$p_{T}(B) [GeV]$', isSignal=True, ax=axes_pt[0])
    plot_turnon_curve(df_test, 'pt_binned', working_points, r'$p_{T}(B) [GeV]$', isSignal=False, ax=axes_pt[1])
    fig_pt.subplots_adjust(hspace=0.05)      
    fig_pt.savefig('lgb_training_results_eff_trunon_pt_{}.pdf'.format(suffix), bbox_inches='tight')

    fig_eta, axes_eta = plt.subplots(2, 1)
    plot_turnon_curve(df_test, 'eta_binned', working_points, r'$\eta(B)$', isSignal=True, ax=axes_eta[0])
    plot_turnon_curve(df_test, 'eta_binned', working_points, r'$\eta(B)$', isSignal=False, ax=axes_eta[1])
    fig_eta.subplots_adjust(hspace=0.05)      
    fig_eta.savefig('lgb_training_results_eff_trunon_eta_{}.pdf'.format(suffix), bbox_inches='tight')

    fig_q2, axes_q2 = plt.subplots(2, 1)
    plot_turnon_curve(df_test, 'q2_binned', working_points, r'$q^{2} [GeV^{2}]$', isSignal=True, ax=axes_q2[0])
    plot_turnon_curve(df_test, 'q2_binned', working_points, r'$q^{2} [GeV^{2}]$', isSignal=False, ax=axes_q2[1])
    fig_q2.subplots_adjust(hspace=0.05)      
    fig_q2.savefig('lgb_training_results_eff_trunon_q2_{}.pdf'.format(suffix), bbox_inches='tight')

    fig_mvaId, axes_mvaId = plt.subplots(2, 1)
    plot_turnon_curve(df_test, 'mvaId_binned', working_points, r'Sub-leading electron mvaId', isSignal=True, ax=axes_mvaId[0])
    plot_turnon_curve(df_test, 'mvaId_binned', working_points, r'Sub-leading electron mvaId', isSignal=False, ax=axes_mvaId[1])
    fig_mvaId.subplots_adjust(hspace=0.05)      
    fig_mvaId.savefig('lgb_training_results_eff_trunon_mvaId_{}.pdf'.format(suffix), bbox_inches='tight')

    df = df.drop("pt_binned", axis=1)
    df = df.drop("eta_binned", axis=1)
    df = df.drop("q2_binned", axis=1)
    df = df.drop("mvaId_binned", axis=1)




