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
import xgboost as xgb
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

def plot_electrons(df, column, bins, logscale=False, ax=None, title=None):
    if ax is None:
        ax = plt.gca()
    for name, group in df.groupby("matchedToGenEle"):
        group[column].hist(bins=bins, histtype="step", label=get_label(name), ax=ax, density=True)
    ax.set_ylabel("density")
    ax.set_xlabel(column)
    ax.legend()
    ax.set_title(title)
    if logscale:
        ax.set_yscale("log", nonposy='clip')

def plot_roc_curve(df, score_column, tpr_threshold=0.0, ax=None, color=None, linestyle='-', label=None):
    print('Plotting ROC...')
    if ax is None:
        ax = plt.gca()
    if label is None:
        label = score_column
    fpr, tpr, thresholds = roc_curve(df["isSignal"], df[score_column], drop_intermediate=True)
    roc_auc = roc_auc_score(df["isSignal"], df[score_column])
    print("auc: {}".format(roc_auc))
    mask = tpr > tpr_threshold
    fpr, tpr = fpr[mask], tpr[mask]
    #ax.semilogx(fpr, tpr, label=label, color=color, linestyle=linestyle)
    ax.plot(fpr, tpr, label=label, color=color, linestyle=linestyle)

def get_working_points(df, score_column):
    
    working_points = {}

    df_test = df[df['test']]

    signal_mask = df_test["matchedToGenEle"] == 1

    df_sig = df_test[signal_mask]
    df_bkg = df_test[~signal_mask]

    # Little detail here: as the signal efficiency has a significant turnon before pT around 5 GeV,
    # we tune the working point to have a given signal efficiency for pT > 5 GeV.
    wp80, wp90 = np.percentile(df_sig.query("ele_pt > 2")[score_column], [20., 10.])

    wp80_bkg_eff = 1.*len(df_bkg[df_bkg[score_column] >= wp80])/len(df_bkg)
    wp90_bkg_eff = 1.*len(df_bkg[df_bkg[score_column] >= wp90])/len(df_bkg)
    
    working_points["wp80"] = wp80
    working_points["wp90"] = wp90
    
    print("")
    print("sig. efficiency at 80 % false alarm rate: {0:.2f} %. WP: {1}".format(wp80_bkg_eff * 100, wp80))
    print("sig. efficiency at 90 % false alarm rate: {0:.2f} %. WP: {1}".format(wp90_bkg_eff * 100, wp90))
    return working_points

def get_working_points_eleMVACats(df, score_column):
    
    working_points = {}

    for i, df_group in df.groupby("EleMVACats_customized"):
        if i > 2:
            continue

        # get the category name
        category = category_titles[i]
        
        working_points[category] = {}

        #df_test = df_group.query("test")
        df_test = df_group[df_group['test']]

        signal_mask = df_test["matchedToGenEle"] == 1

        df_sig = df_test[signal_mask]
        df_bkg = df_test[~signal_mask]

        # Little detail here: as the signal efficiency has a significant turnon before pT around 5 GeV,
        # we tune the working point to have a given signal efficiency for pT > 5 GeV.
        wp80, wp90 = np.percentile(df_sig.query("ele_pt > 2")[score_column], [20., 10.])

        wp80_bkg_eff = 1.*len(df_bkg[df_bkg[score_column] >= wp80])/len(df_bkg)
        wp90_bkg_eff = 1.*len(df_bkg[df_bkg[score_column] >= wp90])/len(df_bkg)
        
        working_points[category]["wp80"] = wp80
        working_points[category]["wp90"] = wp90
        
        print("")
        print(category)
        print("sig. efficiency at 80 % false alarm rate: {0:.2f} %. WP: {1}".format(wp80_bkg_eff * 100, wp80))
        print("sig. efficiency at 90 % false alarm rate: {0:.2f} %. WP: {1}".format(wp90_bkg_eff * 100, wp90))
        
    return working_points

def get_bayesian_var(k, n):
    k = float(k)
    n = float(n)
    return ((k + 1.0) * (k + 2.0)) / ((n + 2.0) * (n + 3.0)) - (((k + 1.0)**2) / ((n + 2.0)**2))

def get_signal_efficiency(df, score_column, working_point):
    df_sig = df.query("matchedToGenEle == 1")
    k = len(df_sig[df_sig[score_column] >= working_point])
    n = len(df_sig)
    return 1.*k/n if n != 0 else np.nan

def get_background_efficiency(df, score_column, working_point):
    df_bkg = df.query("matchedToGenEle == 0")
    k = len(df_bkg[df_bkg[score_column] >= working_point])
    n = len(df_bkg)
    return 1.*k/n if n != 0 else np.nan

def get_signal_efficiency_unc(df, score_column, working_point, bUpper):
    df_sig = df.query("matchedToGenEle == 1")
    k = len(df_sig[df_sig[score_column] >= working_point])
    n = len(df_sig)
    teff = ROOT.TEfficiency()
    return teff.Bayesian(n, k, 0.683, 1.0, 1.0, bUpper, True) if n != 0 else np.nan
    #return np.sqrt(get_bayesian_var(k, n)) if n != 0 else None

def get_background_efficiency_unc(df, score_column, working_point, bUpper):
    df_bkg = df.query("matchedToGenEle == 0")
    k = len(df_bkg[df_bkg[score_column] >= working_point])
    n = len(df_bkg)
    teff = ROOT.TEfficiency()
    return teff.Bayesian(n, k, 0.683, 1.0, 1.0, bUpper, True) if n != 0 else np.nan
    #return np.sqrt(get_bayesian_var(k, n)) if n != 0 else None

def EleMVACats(row):
    if abs(row['scl_eta']) < 0.8: return 0
    #elif 0.8 < abs(row['scl_eta']) < 1.44: return 1
    elif 0.8 < abs(row['scl_eta']) < 1.57: return 1
    elif 1.57 < abs(row['scl_eta']) < 2.5: return 2
    else: return 3

def fpreproc(dtrain, dtest, param):
    label = dtrain.get_label()
    ratio = float(np.sum(label == 0)) / np.sum(label == 1)
    param['scale_pos_weight'] = ratio
    return (dtrain, dtest, param)

space  = [Integer(3, 7, name='max_depth'),
         Real(0.01, 0.2, name='eta'),
         Real(0.0, 1.0, name='gamma'),
         Real(0.5, 1.0, name='subsample'),
         Real(0.5, 1.0, name='colsample_bytree'),
         Real(0.0, 0.2, name='alpha'),
         Real(1.0, 1.5, name='lambda'),
         ]

@use_named_args(space)
def objective(**X):
    global best_auc, best_params
    print("New configuration: {}".format(X))
    params = X.copy()
    params['objective'] = 'binary:logitraw'
    params['eval_metric'] = 'auc'
    params['nthread'] = 6
    params['silent'] = 1
    cv_result = xgb.cv(params, dmatrix_train, num_boost_round=n_boost_rounds, nfold=10, shuffle=True, stratified=True, early_stopping_rounds=75, fpreproc=fpreproc)
    ave_auc = cv_result['test-auc-mean'].iloc[-1]
    print("Average auc: {}".format(ave_auc))
    if ave_auc > best_auc:
      best_auc = ave_auc
      best_params = X
    print("Best auc: {}, Best configuration: {}".format(best_auc, best_params))
    return -ave_auc

def train(xgtrain, xgtest, hyper_params=None):
    watchlist = [(xgtrain, 'train'), (xgtest, 'eval')]
    params = hyper_params.copy()
    label = xgtrain.get_label()
    ratio = float(np.sum(label == 0)) / np.sum(label == 1)
    params['scale_pos_weight'] = ratio
    params['objective'] = 'binary:logitraw'
    params['eval_metric'] = 'auc'
    params['nthread'] = 10
    params['silent'] = 1
    results = {}
    model = xgb.train(params, xgtrain, num_boost_round=n_boost_rounds, evals=watchlist, evals_result=results, early_stopping_rounds=75, verbose_eval=False)
    best_iteration = model.best_iteration + 1
    if best_iteration < n_boost_rounds:
        print("early stopping after {0} boosting rounds".format(best_iteration))
    return model, results

def train_cv(X_train_val, Y_train_val, X_test, Y_test, hyper_params=None):
    xgtrain = xgb.DMatrix(X_train_val, label=Y_train_val)
    xgtest  = xgb.DMatrix(X_test , label=Y_test )
    model, results = train(xgtrain, xgtest, hyper_params=hyper_params)
    Y_predict = model.predict(xgb.DMatrix(X_test), ntree_limit=model.best_ntree_limit)
    fpr, tpr, thresholds = roc_curve(Y_test, Y_predict, drop_intermediate=True)
    roc_auc = roc_auc_score(Y_test, Y_predict)
    print("Best auc: {}".format(roc_auc))
    return model, fpr, tpr, thresholds, roc_auc, results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="A simple ttree plotter")
    parser.add_argument("-s", "--signal", dest="signal", default="RootTree_BParkingNANO_2019Sep12_BuToKJpsi_Toee_mvaTraining_sig_training_pf.root", help="Signal file")
    parser.add_argument("-b", "--background", dest="background", default="RootTree_BParkingNANO_2019Sep12_Run2018A2A3B2B3C2C3D2_mvaTraining_bkg_training_pf.root", help="Background file")
    parser.add_argument("-f", "--suffix", dest="suffix", default=None, help="Suffix of the output name")
    parser.add_argument("-o", "--optimization", dest="optimization", action='store_true', help="Perform Bayesian optimization")
    args = parser.parse_args()

    features = ['BToKEE_fit_l1_normpt', 'BToKEE_fit_l1_eta', 'BToKEE_fit_l1_phi', 'BToKEE_l1_dxy_sig', 'BToKEE_l1_dz', 'BToKEE_fit_l2_normpt', 'BToKEE_fit_l2_eta', 'BToKEE_fit_l2_phi', 'BToKEE_l2_dxy_sig', 'BToKEE_l2_dz', 'BToKEE_fit_k_normpt', 'BToKEE_fit_k_eta', 'BToKEE_fit_k_phi', 'BToKEE_k_DCASig', 'BToKEE_fit_normpt', 'BToKEE_svprob', 'BToKEE_fit_cos2D', 'BToKEE_l_xy_sig']

    '''
    features = ['BToKEE_fit_l1_normpt', 'BToKEE_fit_l1_eta', 'BToKEE_fit_l1_phi', 'BToKEE_l1_dxy_sig', 'BToKEE_l1_dz', 'BToKEE_l1_iso03_rel',
                'BToKEE_fit_l2_normpt', 'BToKEE_fit_l2_eta', 'BToKEE_fit_l2_phi', 'BToKEE_l2_dxy_sig', 'BToKEE_l2_dz', 'BToKEE_l2_iso03_rel',
                'BToKEE_fit_k_normpt', 'BToKEE_fit_k_eta', 'BToKEE_fit_k_phi', 'BToKEE_k_DCASig', 'BToKEE_k_dz', 'BToKEE_k_nValidHits', 'BToKEE_k_iso03_rel',
                'BToKEE_fit_normpt', 'BToKEE_b_iso03_rel', 'BToKEE_svprob', 'BToKEE_fit_cos2D', 'BToKEE_l_xy_sig',
                ]

    features += ['BToKEE_l1_pfmvaId', 'BToKEE_l2_pfmvaId']
    features += ['BToKEE_l1_pfmvaCats', 'BToKEE_l2_pfmvaCats']
    '''

    features = sorted(features)

    ddf = {}
    ddf['sig'] = get_df(args.signal, features)
    ddf['bkg'] = get_df(args.background, features)

    ddf['sig'].replace([np.inf, -np.inf], 10.0**+10, inplace=True)
    ddf['bkg'].replace([np.inf, -np.inf], 10.0**+10, inplace=True)

    nSig = ddf['sig'].shape[0]
    nBkg = 80000
    #nSig = 1000
    #nBkg = 1000
    ddf['sig'] = ddf['sig'].sample(frac=1)[:nSig]
    ddf['bkg'] = ddf['bkg'].sample(frac=1)[:nBkg]

    # add isSignal variable
    ddf['sig']['isSignal'] = 1
    ddf['bkg']['isSignal'] = 0

    df = pd.concat([ddf['sig'],ddf['bkg']]).sort_index(axis=1).sample(frac=1).reset_index(drop=True)
    X = df[features]
    y = df['isSignal']

    suffix = args.suffix
    n_boost_rounds = 800
    n_calls = 120
    n_random_starts = 70
    do_bo = args.optimization
    do_cv = True
    best_params = {'colsample_bytree': 0.8380017432637168, 'subsample': 0.7771020436861611, 'eta': 0.043554653675279234, 'alpha': 0.13978587730419964, 'max_depth': 5, 'gamma': 0.5966218064835417, 'lambda': 1.380893119219306}

    # split X and y up in train and test samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20, random_state=47)

    # Get the number of positive and nevative training examples in this category
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)

    print("training on {0} signal and {1} background".format(n_pos, n_neg))
   
    # get the indices correspondng to the testing and training samples
    idx_train = X_train.index
    idx_test = X_test.index

    dmatrix_train = xgb.DMatrix(X_train.copy(), label=np.copy(y_train), feature_names=[f.replace('_','-') for f in features])
    dmatrix_test  = xgb.DMatrix(X_test.copy(), label=np.copy(y_test), feature_names=[f.replace('_','-') for f in features])

    # Bayesian optimization
    if do_bo:
        begt = time.time()
        print("Begin Bayesian optimization")
        best_auc = 0.0
        best_params = {}
        res_gp = gp_minimize(objective, space, n_calls=n_calls, n_random_starts=n_random_starts, random_state=3, verbose=True)
        print("Finish optimization in {}s".format(time.time()-begt))
        plt.figure()
        plot_convergence(res_gp)
        plt.savefig('training_resultis_bo_convergencePlot_xgb_{}.pdf'.format(suffix))
        plt.figure()
        plot_evaluations(res_gp)
        plt.savefig('training_resultis_bo_evaluationsPlot_xgb_{}.pdf'.format(suffix))
        plt.figure()
        plot_objective(res_gp)
        plt.savefig('training_resultis_bo_objectivePlot_xgb_{}.pdf'.format(suffix))

    # Get the cv plots with the best hyper-parameters
    if do_bo or do_cv:
        print("Get the cv plots with the best hyper-parameters")
        tprs = []
        aucs = []
        figs, axs = plt.subplots()
        #cv = KFold(n_splits=5, shuffle=True)
        cv = StratifiedKFold(n_splits=10, shuffle=True)
        mean_fpr = np.logspace(-4, 0, 100)

        iFold = 0
        for train_idx, test_idx in cv.split(X_train, y_train):
            X_train_cv = X_train.iloc[train_idx]
            X_test_cv = X_train.iloc[test_idx]
            Y_train_cv = y_train.iloc[train_idx]
            Y_test_cv = y_train.iloc[test_idx]

            model, fpr, tpr, thresholds, roc_auc, results = train_cv(X_train_cv, Y_train_cv, X_test_cv, Y_test_cv, hyper_params=best_params)
            epochs = len(results['train']['auc'])
            x_axis = range(0, epochs)
            fig, ax = plt.subplots()
            ax.plot(x_axis, results['train']['auc'], label='Train')
            ax.plot(x_axis, results['eval']['auc'], label='Test')
            ax.legend()
            plt.ylabel('AUC')
            plt.xlabel('Epoch')
            plt.title('Fold: {}'.format(iFold))
            fig.savefig('training_results_learning_curve_cv_fold_{}_{}.pdf'.format(suffix,iFold), bbox_inches='tight')

            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            aucs.append(roc_auc)
            axs.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (iFold, roc_auc))
            iFold += 1

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        axs.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
        axs.plot(np.logspace(-4, 0, 1000), np.logspace(-4, 0, 1000), linestyle='--', lw=2, color='k', label='Random chance')

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        axs.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
        axs.set_xscale('log')
        axs.set_ylim([0, 1.0])
        axs.set_xlabel('False Alarm Rate')
        axs.set_ylabel('True Positive Rate')
        axs.set_title('Cross-validation Receiver Operating Curve')
        axs.legend(loc="lower right") 
        figs.savefig('training_results_roc_cv_{}.pdf'.format(suffix), bbox_inches='tight')


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
    
    model.save_model("xgb_fulldata_{}.model".format(suffix))

    df.loc[idx_train, "score"] = model.predict(dmatrix_train, ntree_limit=model.best_ntree_limit)
    df.loc[idx_test, "score"] = model.predict(dmatrix_test, ntree_limit=model.best_ntree_limit)
   
    df.loc[idx_train, "test"] = False
    df.loc[idx_test, "test"] = True

    print("")
    print("Final model: Best hyper-parameters: {}, ntree_limit: {}".format(best_params, model.best_ntree_limit))
    print("")

    #df_train = df.query("not test")
    #df_test = df.query("test")
    df_train = df[np.logical_not(df['test'])]
    df_test = df[df['test']]
    #df_test.to_csv('training_results_testdf_{}.csv'.format(suffix))

    fpr, tpr, thresholds = roc_curve(df_test["isSignal"], df_test["score"], drop_intermediate=True)
    roc_dict = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
    roc_df = pd.DataFrame(data=roc_dict)
    roc_df.to_csv('training_results_roc_csv_{}.csv'.format(suffix))

    epochs = len(results['train']['auc'])
    x_axis = range(0, epochs)
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['train']['auc'], label='Train')
    ax.plot(x_axis, results['eval']['auc'], label='Test')
    ax.legend()
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    fig.savefig('training_results_learning_curve_{}.pdf'.format(suffix), bbox_inches='tight')


    fig, ax = plt.subplots()
    plot_roc_curve(df_test, "score", ax=ax, label="XGB")
    ax.plot(np.logspace(-4, 0, 1000), np.logspace(-4, 0, 1000), linestyle='--', color='k')
    '''
    mvaCut = np.linspace(0.0, 4.0, 10)
    wp_fpr = interp(mvaCut, thresholds[::-1], fpr[::-1])
    wp_tpr = interp(mvaCut, thresholds[::-1], tpr[::-1])
    ax.scatter(wp_fpr, wp_tpr, c='r', label="Working point")
    for i, mva in enumerate(mvaCut):
      ax.annotate(round(mva,2), (wp_fpr[i], wp_tpr[i]), fontsize=10, xytext=(10,-20), textcoords="offset points", arrowprops=dict(arrowstyle="->"))
    '''
    ax.set_xlim([1.0e-4, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xscale('log')
    ax.set_xlabel("False Alarm Rate")
    ax.set_ylabel("Signal Efficiency")
    ax.set_title('Receiver Operating Curve')
    ax.legend(loc='upper left')
    fig.savefig('training_results_roc_curve_{}.pdf'.format(suffix), bbox_inches='tight')

    plt.figure()
    xgb.plot_importance(model)
    plt.savefig('training_results_feature_importance_{}.pdf'.format(suffix), bbox_inches='tight')

