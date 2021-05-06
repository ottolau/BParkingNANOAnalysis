import matplotlib as mpl
mpl.use('pdf')
from matplotlib import pyplot as plt
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
from collections import OrderedDict

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

def get_weights(sig, bkg, suffix):
    xedges = np.linspace(0.045, 20, 50)
    q2_min, q2_max = min(xedges), max(xedges)
    n     = len(bkg)
    n_sig = len(sig)
    n_bkg = len(bkg)

    H_sig, xedges = np.histogram(sig, bins=xedges)
    H_bkg, xedges = np.histogram(bkg, bins=xedges)

    W = H_sig.astype(float) / H_bkg.astype(float)

    W[W == np.inf] = 0
    W = np.nan_to_num(W)
    bkg_in_bin = np.logical_and.reduce([bkg > q2_min, bkg < q2_max])
    bkg_in_bins = bkg[bkg_in_bin]

    in_bin = np.logical_and.reduce([sig > q2_min, sig < q2_max])
    sig_in_bins = sig[in_bin]
    xinds = np.digitize(bkg_in_bins, xedges) - 1

    n_bkg_in_bins = len(bkg_in_bins)
    n_bkg_overflow = n_bkg - n_bkg_in_bins

    weights = np.ones(n)
    bkg_weights = np.zeros(n_bkg)
    bkg_weights[bkg_in_bin] = W[xinds]

    weights = bkg_weights

    density=True
    var_name, edges = r'$q^{2} [GeV^{2}]$', xedges

    bin_centers = edges[:-1] + np.diff(edges)/2.

    hist, _, _ = plt.hist(bkg_in_bins, edges, histtype='step', label="background")
    hist_reweighted_density, _, _ = plt.hist(bkg_in_bins, edges, normed=density, histtype='step', weights=weights[bkg_in_bin])

    plt.figure()
    plt.hist(sig_in_bins, edges, normed=density, histtype='step', label="signal")
    hist_density, _, _ = plt.hist(bkg_in_bins, edges, normed=density, histtype='step', label="background")

    hist_err = np.sqrt(hist)
    hist_reweighted_density_err = hist_err * hist_reweighted_density / hist

    plt.errorbar(bin_centers, hist_reweighted_density, yerr=hist_reweighted_density_err, fmt='o', color='k', label="background reweighted", markersize="3")
    plt.xlabel(var_name)
    plt.ylabel('density')
    plt.legend()
    plt.savefig('training_results_q2weighting_{}.pdf'.format(suffix), bbox_inches='tight')
    plt.close()

    return weights

def plot_roc_curve(df, score_column, tpr_threshold=0.0, ax=None, color=None, linestyle='-', label=None):
    print('Plotting ROC...')
    if ax is None:
        ax = plt.gca()
    if label is None:
        label = score_column
    fpr, tpr, thresholds = roc_curve(df["isSignal"], df[score_column], drop_intermediate=True)
    roc_auc = roc_auc_score(df["isSignal"], df[score_column])
    roc_pauc = roc_auc_score(df["isSignal"], df[score_column], max_fpr=max_fpr)
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


def fpreproc(dtrain, dtest, param):
    label = dtrain.get_label()
    ratio = float(np.sum(label == 0)) / np.sum(label == 1)
    param['scale_pos_weight'] = ratio
    return (dtrain, dtest, param)

def pauc(predt, dtrain):
    y = dtrain.get_label()
    return 'pauc', roc_auc_score(y, predt, max_fpr=max_fpr)

space  = [Integer(6, 10, name='max_depth'),
         Real(0.001, 0.05, name='eta'),
         Real(0.0, 3.0, name='gamma'),
         #Integer(1.0, 10.0, name='min_child_weight'),
         Real(0.0, 1.0, name='min_child_weight'),
         Real(0.5, 1.0, name='subsample'),
         Real(0.1, 1.0, name='colsample_bytree'),
         Real(0.0, 10.0, name='alpha'),
         Real(0.1, 10.0, name='lambda'),
         ]

@use_named_args(space)
def objective(**X):
    global best_auc, best_auc_std, best_params
    print("New configuration: {}".format(X))
    params = X.copy()
    params['objective'] = 'binary:logitraw'
    params['tree_method'] = 'hist'
    #params['eval_metric'] = 'auc'
    params['nthread'] = 6
    params['silent'] = 1
    cv_result = xgb.cv(params, dmatrix_train, num_boost_round=n_boost_rounds, nfold=5, shuffle=True, stratified=True, maximize=True, early_stopping_rounds=100, fpreproc=fpreproc, feval=pauc)
    ave_auc = cv_result['test-pauc-mean'].iloc[-1]
    ave_auc_std = cv_result['test-pauc-std'].iloc[-1]
    print("Average pauc: {}+-{}".format(ave_auc, ave_auc_std))
    if ave_auc > best_auc:
      best_auc = ave_auc
      best_auc_std = ave_auc_std
      best_params = X.copy()
    print("Best pauc: {}+-{}, Best configuration: {}".format(best_auc, best_auc_std, best_params))
    return -ave_auc

def train(xgtrain, xgtest, hyper_params=None):
    watchlist = [(xgtrain, 'train'), (xgtest, 'eval')]
    params = hyper_params.copy()
    label = xgtrain.get_label()
    ratio = float(np.sum(label == 0)) / np.sum(label == 1)
    params['scale_pos_weight'] = ratio
    params['objective'] = 'binary:logitraw'
    params['tree_method'] = 'hist'
    #params['eval_metric'] = 'auc'
    #params['nthread'] = 10
    params['silent'] = 1
    results = {}
    model = xgb.train(params, xgtrain, num_boost_round=n_boost_rounds, evals=watchlist, evals_result=results, maximize=True, early_stopping_rounds=100, verbose_eval=False, feval=pauc)
    best_iteration = model.best_iteration + 1
    if best_iteration < n_boost_rounds:
        print("early stopping after {0} boosting rounds".format(best_iteration))
    return model, results

def train_cv(X_train_val, Y_train_val, X_test, Y_test, w_train_val, hyper_params=None):
    xgtrain = xgb.DMatrix(X_train_val, label=Y_train_val, weight=w_train_val)
    xgtest  = xgb.DMatrix(X_test , label=Y_test )
    model, results = train(xgtrain, xgtest, hyper_params=hyper_params)
    Y_predict = model.predict(xgtest, ntree_limit=model.best_ntree_limit)
    fpr, tpr, thresholds = roc_curve(Y_test, Y_predict, drop_intermediate=True)
    roc_auc = roc_auc_score(Y_test, Y_predict, max_fpr=max_fpr)
    print("Best pauc: {}".format(roc_auc))
    return model, fpr, tpr, thresholds, roc_auc, results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="A simple ttree plotter")
    parser.add_argument("-s", "--signal", dest="signal", default="RootTree_2020Jan16_BuToKee_all_BToKEEAnalyzer_2020May03_newVar_mc_pf.root", help="Signal file")
    parser.add_argument("-b", "--background", dest="background", default="BParkingNANO_2020Jan16_Run2018ABCD_BToKEEAnalyzer_2020May03_allq2_newVar_upSB_pf_random_subsample.root", help="Background file")
    parser.add_argument("-f", "--suffix", dest="suffix", default=None, help="Suffix of the output name")
    parser.add_argument("-o", "--optimization", dest="optimization", action='store_true', help="Perform Bayesian optimization")
    args = parser.parse_args()

    
    suffix = args.suffix
    max_fpr = 1.0e-2
    features = ['BToKEE_fit_l1_normpt', 'BToKEE_fit_l2_normpt',
                'BToKEE_l1_dxy_sig', 'BToKEE_l2_dxy_sig',
                'BToKEE_fit_k_normpt', 'BToKEE_k_DCASig',
                'BToKEE_fit_normpt', 'BToKEE_svprob', 'BToKEE_fit_cos2D', 'BToKEE_l_xy_sig',
                ]
    features += ['BToKEE_eleDR', 'BToKEE_llkDR']
    features += ['BToKEE_l1_iso04_rel', 'BToKEE_l2_iso04_rel', 'BToKEE_k_iso04_rel', 'BToKEE_b_iso04_rel']
    features += ['BToKEE_ptAsym']
    #features += ['BToKEE_Dmass', 'BToKEE_Dmass_flip']
    features += ['BToKEE_l1_pfmvaId_lowPt', 'BToKEE_l2_pfmvaId_lowPt', 'BToKEE_l1_pfmvaId_highPt', 'BToKEE_l2_pfmvaId_highPt']
    #features += ['BToKEE_l1_mvaId', 'BToKEE_l2_mvaId']
    features += ['BToKEE_l1_dzTrg', 'BToKEE_l2_dzTrg', 'BToKEE_k_dzTrg']
    features += ['BToKEE_k_svip2d', 'BToKEE_k_svip3d']
    #features += ['BToKEE_l1_iso04_dca_rel', 'BToKEE_l2_iso04_dca_rel', 'BToKEE_k_iso04_dca_rel', 'BToKEE_b_iso04_dca_rel']
    #features += ['BToKEE_l1_iso04_dca_tight_rel', 'BToKEE_l2_iso04_dca_tight_rel', 'BToKEE_k_iso04_dca_tight_rel', 'BToKEE_b_iso04_dca_tight_rel']
    #features += ['BToKEE_l1_n_isotrk', 'BToKEE_l2_n_isotrk', 'BToKEE_k_n_isotrk', 'BToKEE_b_n_isotrk']
    #features += ['BToKEE_l1_n_isotrk_dca', 'BToKEE_l2_n_isotrk_dca', 'BToKEE_k_n_isotrk_dca', 'BToKEE_b_n_isotrk_dca']
    #features += ['BToKEE_l1_n_isotrk_dca_tight', 'BToKEE_l2_n_isotrk_dca_tight', 'BToKEE_k_n_isotrk_dca_tight', 'BToKEE_b_n_isotrk_dca_tight']

    features = sorted(features)
    branches = features + ['BToKEE_fit_mass', 'BToKEE_fit_massErr', 'BToKEE_fit_pt', 'BToKEE_fit_eta', 'BToKEE_q2']

    ddf = {}
    ddf['sig'] = get_df(args.signal, branches)
    ddf['bkg'] = get_df(args.background, branches)

    ddf['sig'].replace([np.inf, -np.inf], 0.0, inplace=True)
    ddf['bkg'].replace([np.inf, -np.inf], 0.0, inplace=True)

    '''
    selection = '(BToKEE_fit_l1_normpt*BToKEE_fit_mass > 2.0) and (BToKEE_fit_l2_normpt*BToKEE_fit_mass > 2.0)'
    ddf['sig'].query(selection, inplace=True)
    ddf['bkg'].query(selection, inplace=True)
    '''

    #nSig = ddf['sig'].shape[0]
    #nBkg = 25000
    nSig = 300000
    nBkg = 300000
    ddf['sig'] = ddf['sig'].sample(frac=1)[:nSig]
    ddf['bkg'] = ddf['bkg'].sample(frac=1)[:nBkg]

    # add isSignal variable
    ddf['sig']['isSignal'] = 1
    ddf['bkg']['isSignal'] = 0

    # add weights
    #ddf['sig']['weights'] = 1.0/ddf['sig']['BToKEE_fit_massErr'].replace(np.nan, 1.0)
    #ddf['bkg']['weights'] = get_weights(ddf['sig']['BToKEE_q2'], ddf['bkg']['BToKEE_q2'], suffix)
    ddf['sig']['weights'] = 1.0
    ddf['bkg']['weights'] = 1.0

    df = pd.concat([ddf['sig'],ddf['bkg']]).sort_index(axis=1).sample(frac=1).reset_index(drop=True)

    X = df[features]
    y = df['isSignal']
    W = df['weights']

    n_boost_rounds = 800
    n_calls = 80
    n_random_starts = 40
    do_bo = args.optimization
    do_cv = False
    #best_params = {'colsample_bytree': 0.5825650686239838, 'min_child_weight': 0.8004306182129557, 'subsample': 0.8363871434049193, 'eta': 0.03806335273785261, 'alpha': 6.813338261152275, 'max_depth': 9, 'gamma': 0.16767944414770025, 'lambda': 9.582893111434032}
    #best_params = {'colsample_bytree': 0.49090035832528456, 'min_child_weight': 0.5554583843109246, 'subsample': 0.9256932381586518, 'eta': 0.03723487005901925, 'alpha': 8.766606138603317, 'max_depth': 9, 'gamma': 2.066453710120974, 'lambda': 6.425723447952634}
    #best_params = {'colsample_bytree': 0.3983963024147342, 'min_child_weight': 0.9649953160358865, 'subsample': 0.9960147806640621, 'eta': 0.04736950401062728, 'alpha': 0.18133633301551935, 'max_depth': 9, 'gamma': 2.099735486419789, 'lambda': 9.6853903960718}
    #best_params = {'colsample_bytree': 0.49090035832528456, 'min_child_weight': 0.5554583843109246, 'subsample': 0.9256932381586518, 'eta': 0.03723487005901925, 'alpha': 8.766606138603317, 'max_depth': 9, 'gamma': 2.066453710120974, 'lambda': 6.425723447952634}
    #best_params = {'colsample_bytree': 0.33494788911748086, 'min_child_weight': 0.9401647699443538, 'subsample': 0.582774071495416, 'eta': 0.04972190783672264, 'alpha': 0.5463715009170179, 'max_depth': 8, 'gamma': 2.9111764103158935, 'lambda': 0.5566585617286808}
    #best_params = {'colsample_bytree': 0.45134538658651024, 'min_child_weight': 0.944927618484819, 'subsample': 0.5121326042318723, 'eta': 0.03226800105641119, 'alpha': 0.06607299498327614, 'max_depth': 8, 'gamma': 2.734455156885833, 'lambda': 0.6026281920548624}
    #best_params = {'colsample_bytree': 0.907543878419449, 'min_child_weight': 0.012247747026221115, 'subsample': 0.9720367108801046, 'eta': 0.034584090868681104, 'alpha': 0.4992270733917138, 'max_depth': 9, 'gamma': 2.969479938139431, 'lambda': 0.13158392260368543}
    #best_params = {'subsample': 0.7964473633389513, 'eta': 0.04324038245866539, 'colsample_bytree': 0.9997097256893213, 'gamma': 0.41930662741996866, 'alpha': 0.3595590194583754, 'max_depth': 10, 'min_child_weight': 10, 'lambda': 9.137209334601707}
    #best_params = {'subsample': 0.7964473633389513, 'eta': 0.04324038245866539, 'colsample_bytree': 0.9997097256893213, 'gamma': 0.41930662741996866, 'alpha': 0.3595590194583754, 'max_depth': 10, 'min_child_weight': 10, 'lambda': 9.137209334601707}
    #best_params = {'colsample_bytree': 0.6985151799554887, 'min_child_weight': 0.1880992226152179, 'subsample': 0.5059776752154034, 'eta': 0.025091014670069068, 'alpha': 0.5825875179922492, 'max_depth': 10, 'gamma': 0.15761992682454778, 'lambda': 4.552028057728982}
    #best_params = {'colsample_bytree': 0.6417405650302768, 'min_child_weight': 0.7998463694760696, 'subsample': 0.5570470959335826, 'eta': 0.02867563834178777, 'alpha': 2.4126097151465116, 'max_depth': 11, 'gamma': 1.491648232476693, 'lambda': 5.956681761649099}
    #best_params = {'colsample_bytree': 0.45134538658651024, 'min_child_weight': 0.944927618484819, 'subsample': 0.5121326042318723, 'eta': 0.03226800105641119, 'alpha': 0.06607299498327614, 'max_depth': 8, 'gamma': 2.734455156885833, 'lambda': 0.6026281920548624}
    #best_params = {'colsample_bytree': 0.7878804118099031, 'min_child_weight': 0.04738616608180214, 'subsample': 0.7959243848808419, 'eta': 0.04561652677069882, 'alpha': 4.648974389014093, 'max_depth': 11, 'gamma': 2.2517449536394687, 'lambda': 7.969893573507295}
    best_params = {'colsample_bytree': 0.37983170319692516, 'min_child_weight': 0.9657673185693753, 'subsample': 0.9896553723307563, 'eta': 0.04914400734945899, 'alpha': 0.7246914042028753, 'max_depth': 12, 'gamma': 2.866440307554156, 'lambda': 3.7932151643303103}

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

    dmatrix_train = xgb.DMatrix(X_train.copy(), label=np.copy(y_train), feature_names=[f.replace('_','-') for f in features], weight=np.copy(w_train))
    dmatrix_test  = xgb.DMatrix(X_test.copy(), label=np.copy(y_test), feature_names=[f.replace('_','-') for f in features])

    # Bayesian optimization
    if do_bo:
        begt = time.time()
        print("Begin Bayesian optimization")
        best_auc = 0.0
        best_auc_std = 0.0
        best_params = {}
        res_gp = gp_minimize(objective, space, n_calls=n_calls, n_random_starts=n_random_starts, verbose=True, random_state=36)
        print("Finish optimization in {}s".format(time.time()-begt))

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
            epochs = len(results['train']['pauc'])
            x_axis = range(0, epochs)
            fig, ax = plt.subplots()
            ax.plot(x_axis, results['train']['pauc'], label='Train')
            ax.plot(x_axis, results['eval']['pauc'], label='Test')
            ax.legend()
            plt.ylabel('PAUC')
            plt.xlabel('Epoch')
            plt.title('Fold: {}'.format(iFold))
            fig.savefig('training_results_learning_curve_cv_fold_{}_{}.pdf'.format(suffix,iFold), bbox_inches='tight')

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
        axs.set_xlim([1.0e-5, 1.0])
        axs.set_ylim([0.0, 1.0])
        axs.set_xlabel('False Alarm Rate')
        axs.set_ylabel('Signal Efficiency')
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

    epochs = len(results['train']['pauc'])
    x_axis = range(0, epochs)
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['train']['pauc'], label='Train')
    ax.plot(x_axis, results['eval']['pauc'], label='Test')
    ax.legend()
    plt.ylabel('PAUC')
    plt.xlabel('Epoch')
    fig.savefig('training_results_learning_curve_{}.pdf'.format(suffix), bbox_inches='tight')


    fig, ax = plt.subplots()
    plot_roc_curve(df_test, "score", ax=ax, label="XGB")
    ax.plot(np.logspace(-6, 0, 1000), np.logspace(-6, 0, 1000), linestyle='--', color='k')
    ax.set_xlim([1.0e-6, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xscale('log')
    ax.set_xlabel("False Alarm Rate")
    ax.set_ylabel("Signal Efficiency")
    ax.set_title('Receiver Operating Curve')
    ax.legend(loc='lower right')
    fig.savefig('training_results_roc_curve_{}.pdf'.format(suffix), bbox_inches='tight')

    plt.figure()
    xgb.plot_importance(model)
    plt.savefig('training_results_feature_importance_{}.pdf'.format(suffix), bbox_inches='tight')



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
    q2_bins = np.linspace(0.045, 25.0, n_q2_bins)
    df["q2_binned"] = get_bins_center(df["BToKEE_q2"], q2_bins)


    df_test = df[df["test"]]

    fig_pt, axes_pt = plt.subplots(2, 1)
    plot_turnon_curve(df_test, 'pt_binned', working_points, r'$p_{T}(B) [GeV]$', isSignal=True, ax=axes_pt[0])
    plot_turnon_curve(df_test, 'pt_binned', working_points, r'$p_{T}(B) [GeV]$', isSignal=False, ax=axes_pt[1])
    fig_pt.subplots_adjust(hspace=0.05)      
    fig_pt.savefig('training_results_eff_trunon_pt_{}.pdf'.format(suffix), bbox_inches='tight')

    fig_eta, axes_eta = plt.subplots(2, 1)
    plot_turnon_curve(df_test, 'eta_binned', working_points, r'$\eta(B)$', isSignal=True, ax=axes_eta[0])
    plot_turnon_curve(df_test, 'eta_binned', working_points, r'$\eta(B)$', isSignal=False, ax=axes_eta[1])
    fig_eta.subplots_adjust(hspace=0.05)      
    fig_eta.savefig('training_results_eff_trunon_eta_{}.pdf'.format(suffix), bbox_inches='tight')

    fig_q2, axes_q2 = plt.subplots(2, 1)
    plot_turnon_curve(df_test, 'q2_binned', working_points, r'$q^{2} [GeV^{2}]$', isSignal=True, ax=axes_q2[0])
    plot_turnon_curve(df_test, 'q2_binned', working_points, r'$q^{2} [GeV^{2}]$', isSignal=False, ax=axes_q2[1])
    fig_q2.subplots_adjust(hspace=0.05)      
    fig_q2.savefig('training_results_eff_trunon_q2_{}.pdf'.format(suffix), bbox_inches='tight')

    df = df.drop("pt_binned", axis=1)
    df = df.drop("eta_binned", axis=1)
    df = df.drop("q2_binned", axis=1)
   
    if 'BToKEE_l2_mvaId' in features:
      n_mvaId_bins = 100
      mvaId_bins = np.linspace(0.0, 10.0, n_mvaId_bins)
      df["mvaId_binned"] = get_bins_center(df["BToKEE_l2_mvaId"], mvaId_bins)
      df_test = df[df["test"]]

      fig_mvaId, axes_mvaId = plt.subplots(2, 1)
      plot_turnon_curve(df_test, 'mvaId_binned', working_points, r'Sub-leading electron mvaId', isSignal=True, ax=axes_mvaId[0])
      plot_turnon_curve(df_test, 'mvaId_binned', working_points, r'Sub-leading electron mvaId', isSignal=False, ax=axes_mvaId[1])
      fig_mvaId.subplots_adjust(hspace=0.05)      
      fig_mvaId.savefig('training_results_eff_trunon_mvaId_{}.pdf'.format(suffix), bbox_inches='tight')

      df = df.drop("mvaId_binned", axis=1)

