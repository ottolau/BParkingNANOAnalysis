import matplotlib as mpl
mpl.use('agg')
from matplotlib import pyplot as plt
from matplotlib import rc
#.Allow for using TeX mode in matplotlib Figures
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern Roman']})
rc('text', usetex=False)
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]

ratio=5.0/7.0
fig_width_pt = 3*246.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = ratio if ratio != 0.0 else (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]

params = {'text.usetex' : False,
        'axes.labelsize': 24,
        'font.size': 24,
        'legend.fontsize': 12,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'font.family' : 'lmodern',
        'text.latex.unicode': False,
        'axes.grid' : True,
        'figure.figsize': fig_size}
plt.rcParams.update(params)

import uproot
import glob
import pandas as pd
from tqdm import tqdm
import numpy as np
import h5py
import time
import ROOT
import sys
np.set_printoptions(threshold=sys.maxsize)
from matplotlib.backends.backend_pdf import PdfPages
from collections import OrderedDict
import seaborn as sns
from scipy.interpolate import interp1d

unneccesary_columns = []

def get_df(root_file_name):
    print('Opening file {}...'.format(root_file_name))
    f = uproot.open(root_file_name)
    if len(f.allkeys()) == 0:
        return pd.DataFrame()
    #df = uproot.open(root_file_name)["tree"].pandas.df()
    #df = pd.DataFrame(uproot.open(root_file_name)["tree"].arrays(namedecode="utf-8"))
    df = pd.DataFrame(uproot.open(root_file_name)["tree"].arrays())
    print('Finished opening file {}...'.format(root_file_name))
    return df.drop(unneccesary_columns, axis=1)

def get_label(name):
    if name == -1:
        return "Data bkg"
    if name == 0:
      return "MC: B+ to K+ ee"
        #return "EB" #"B+ to K+ ee"
    if name == 1:
      return "data" #"B0 to K* ee"
    if name == 2:
        return "EBEE" #"B+ to K+ J/psi(ee)"

def plot_hist(df, column, bins=None, logscale=False, ax=None, title=None):
    if ax is None:
        ax = plt.gca()
    #ax.hist(np.clip(df[column], bins[0], bins[-1]), bins=bins, histtype='step', normed=True)
    for name, group in df.groupby("Category"):
        #ax.hist(np.clip(group[column], bins[0], bins[-1]), bins=bins, histtype='step', label=get_label(name), normed=True)
        ax.hist(group[column], bins=bins, histtype='step', label=get_label(name), normed=True)
        #ax.hist(group[column], bins=bins, histtype='step', label='{0}, Mean: {1:.2f}'.format(get_label(name), np.mean(group[column])), normed=True)
    ax.set_ylabel("density")
    ax.set_xlabel(column)
    ax.legend()
    ax.set_title(title)
    if logscale:
        ax.set_yscale("log", nonposy='clip')

def plot_duplicates(df, column, bins=None, logscale=False, ax=None, title=None):
    if ax is None:
        ax = plt.gca()
    #x = df.groupby('BToKEE_event').count()[column]
    #ax.hist(np.clip(x, bins[0], bins[-1]), bins=bins, histtype='step', label='Mean: {}'.format(np.mean(x)), normed=True)
    for name, group in df.groupby("Category"):
        x = group.groupby('BToKEE_event').count()[column]
        #ax.hist(np.clip(x, bins[0], bins[-1]), bins=bins, histtype='step', label='{}, Mean: {}'.format(get_label(name), np.mean(x)), normed=True)
        ax.hist(x, bins=bins, histtype='step', label='{0}, Mean: {1:.2f}'.format(get_label(name), np.mean(x)), normed=True)
    ax.set_ylabel("density")
    ax.set_xlabel("Duplicates")
    ax.legend()
    ax.set_title(title)
    if logscale:
        ax.set_yscale("log", nonposy='clip')

def plot_corr(df, outputfile):
    print("Plotting correlation...")
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    fig, ax = plt.subplots(figsize=(110, 90))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1.0, vmax=1.0, center=0, square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .5})
    fig.savefig(outputfile, bbox_inches='tight')

def hide_current_axis(*args, **kwds):
    plt.gca().set_visible(False)

def plot_pairgrid(df, outputfile):
    print("Plotting pair grid...")
    '''
    g = sns.PairGrid(df, hue="Category")
    g = g.map_diag(plt.hist, histtype="step", linewidth=3)
    g = g.map_upper(hide_current_axis)
    g = g.map_lower(plt.scatter)
    g = g.add_legend() 
    g.savefig(outputfile)
    '''
    g = sns.pairplot(df, hue='Category', corner=True)
    g.savefig(outputfile)
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="A simple ttree plotter")
    parser.add_argument("-s", "--signal", dest="signal", default="BuToKJpsi_Toee_pfretrain_ntuple.root", help="Signal file")
    parser.add_argument("-b", "--background", dest="background", default="BuToKJpsi_Toee_pfretrain_ntuple.root", help="Background file")
    parser.add_argument("-o", "--outputfile", dest="outputfile", default="test.pdf", help="Output file")
    parser.add_argument("-f", "--suffix", dest="suffix", default=None, help="Suffix of the output name")
    args = parser.parse_args()


    features = OrderedDict([#('BToKEE_mll_raw', np.linspace(0.0, 5.0, 100)),
                            ('BToKEE_mll_fullfit', np.linspace(0.0, 5.0, 100)),
                            #('BToKEE_mll_llfit', np.linspace(0.0, 5.0, 100)),
                            ('BToKEE_q2', np.linspace(1.0, 25.0, 100)),
                            ('BToKEE_fit_mass', np.linspace(4.5, 6.0, 100)),
                            ('BToKEE_fit_massErr', np.linspace(0.0, 0.5, 100)),
                            ('BToKEE_fit_pt', np.linspace(0.0, 50.0, 100)),
                            ('BToKEE_fit_normpt', np.linspace(0.0, 10.0, 100)),
                            ('BToKEE_fit_eta', np.linspace(-3.0, 3.0, 100)),
                            ('BToKEE_fit_phi', np.linspace(-np.pi, np.pi, 100)),
                            ('BToKEE_dz', np.linspace(-1.0, 1.0, 100)),
                            #('BToKEE_b_iso03_rel', np.linspace(0.0, 10.0, 100)),
                            ('BToKEE_b_iso04_rel', np.linspace(0.0, 10.0, 100)),
                            ('BToKEE_fit_l1_pt', np.linspace(0.0, 20.0, 100)),
                            ('BToKEE_fit_l2_pt', np.linspace(0.0, 10.0, 100)),
                            ('BToKEE_fit_l1_normpt', np.linspace(0.0, 10.0, 100)),
                            ('BToKEE_fit_l2_normpt', np.linspace(0.0, 5.0, 100)),
                            ('BToKEE_fit_l1_eta', np.linspace(-3.0, 3.0, 100)),
                            ('BToKEE_fit_l2_eta', np.linspace(-3.0, 3.0, 100)),
                            ('BToKEE_fit_l1_phi', np.linspace(-np.pi, np.pi, 100)),
                            ('BToKEE_fit_l2_phi', np.linspace(-np.pi, np.pi, 100)),
                            ('BToKEE_l1_dxy_sig', np.linspace(-30.0, 30.0, 100)),
                            ('BToKEE_l2_dxy_sig', np.linspace(-30.0, 30.0, 100)),
                            #('BToKEE_l1_dz', np.linspace(-0.5, 0.5, 100)),
                            #('BToKEE_l2_dz', np.linspace(-0.5, 0.5, 100)),
                            ('BToKEE_l1_mvaId', np.linspace(-10.0, 20.0, 100)),
                            ('BToKEE_l2_mvaId', np.linspace(-10.0, 20.0, 100)),
                            ('BToKEE_l1_pfmvaId', np.linspace(-10.0, 20.0, 100)),
                            ('BToKEE_l2_pfmvaId', np.linspace(-10.0, 20.0, 100)),
                            #('BToKEE_l1_pfmvaId_lowPt', np.linspace(-10.0, 10.0, 100)),
                            #('BToKEE_l2_pfmvaId_lowPt', np.linspace(-10.0, 10.0, 100)),
                            #('BToKEE_l1_pfmvaId_highPt', np.linspace(-10.0, 10.0, 100)),
                            #('BToKEE_l2_pfmvaId_highPt', np.linspace(-10.0, 10.0, 100)),
                            #('BToKEE_l1_iso03_rel', np.linspace(0.0, 10.0, 100)),
                            #('BToKEE_l2_iso03_rel', np.linspace(0.0, 10.0, 100)),
                            ('BToKEE_l1_iso04_rel', np.linspace(0.0, 10.0, 100)),
                            ('BToKEE_l2_iso04_rel', np.linspace(0.0, 10.0, 100)),
                            ('BToKEE_fit_k_pt', np.linspace(0.0, 10.0, 100)),
                            ('BToKEE_fit_k_normpt', np.linspace(0.0, 2.0, 100)),
                            ('BToKEE_fit_k_eta', np.linspace(-3.0, 3.0, 100)),
                            ('BToKEE_fit_k_phi', np.linspace(-np.pi, np.pi, 100)),
                            ('BToKEE_k_DCASig', np.linspace(0.0, 10.0, 100)),
                            #('BToKEE_k_dz', np.linspace(-0.5, 0.5, 100)),
                            ('BToKEE_k_nValidHits', np.linspace(0.0, 40.0, 40)),
                            #('BToKEE_k_iso03_rel', np.linspace(0.0, 10.0, 100)),
                            ('BToKEE_k_iso04_rel', np.linspace(0.0, 10.0, 100)),
                            ('BToKEE_svprob', np.linspace(0.0, 1.0, 100)),
                            ('BToKEE_fit_cos2D', np.linspace(0.999, 1.0, 100)),
                            ('BToKEE_l_xy_sig', np.linspace(0.0, 40.0, 100)),
                            ('BToKEE_Dmass', np.linspace(0.5, 5.0, 100)),
                            ('BToKEE_maxDR', np.linspace(0.0, 5.0, 100)),
                            ('BToKEE_minDR', np.linspace(0.0, 3.0, 100)),
                            ])
    
    #n_files = 20
    #root_files = glob.glob("/eos/user/r/rembserj/ntuples/electron_mva_run3/*.root")[:n_files]
    #df = pd.concat((get_df(f) for f in  tqdm(root_files)), ignore_index=True)
    df1 = get_df(args.signal)
    df2 = get_df(args.background)
    #df2 = df2.sample(frac=1)[:300000]
    print('variables in ntuples: {}'.format(df1.columns))
    df1['Category'] = 0
    df2['Category'] = -1
    
    #df1.drop(columns=['BToKEE_decay'])
    #df1 = df1[np.logical_not(df1['BToKEE_l1_isGen'] | df1['BToKEE_l2_isGen'] | df1['BToKEE_k_isGen'])]
    #drop_columns = ['BToKEE_decay', 'BToKEE_l1_isGen', 'BToKEE_l2_isGen', 'BToKEE_k_isGen', 'BToKEE_l1_genPdgId', 'BToKEE_l2_genPdgId', 'BToKEE_k_genPdgId']
    #df1 = df1[(df1['BToKEE_decay'] != 0)].drop(columns=drop_columns)
    #df2 = df2.drop(columns=['BToKEE_decay'])
    #df1.drop(columns=['BToKEE_decay'])
    df1 = df1[:5000000]
    df2 = df2[:5000000]
    df = pd.concat((df1, df2), ignore_index=True).replace([np.inf, -np.inf], 0.0)
    #df = df1.copy()

   
    df = df[(df['BToKEE_svprob'] > 0.1) & (df['BToKEE_fit_cos2D'] > 0.999)]

    l1_pf_selection = (df['BToKEE_l1_isPF'])
    l2_pf_selection = (df['BToKEE_l2_isPF'])
    l1_low_selection = (df['BToKEE_l1_isLowPt']) 
    l2_low_selection = (df['BToKEE_l2_isLowPt']) 

    pf_selection = l1_pf_selection & l2_pf_selection # & (df['BToKEE_k_pt'] > 1.5) & (df['BToKEE_pt'] > 10.0)
    low_selection = l1_low_selection & l2_low_selection
    #overlap_veto_selection = np.logical_not(df['BToKEE_l1_isPFoverlap']) & np.logical_not(df['BToKEE_l2_isPFoverlap'])
    #mix_selection = ((l1_pf_selection & l2_low_selection) | (l2_pf_selection & l1_low_selection))
    #low_pfveto_selection = low_selection & overlap_veto_selection
    #mix_net_selection = overlap_veto_selection & np.logical_not(pf_selection | low_selection)
    #all_selection = pf_selection | low_pfveto_selection | mix_net_selection 
    
    #trigger_selection = (df['BToKMuMu_l1_isTriggering'] == 1) | (df['BToKMuMu_l2_isTriggering'] == 1)
    
    #pu_selection = (df['BToKEE_PV_npvsGood'] < 15)

    '''
    df_pf = df[pf_selection].sort_values('BToKEE_fit_pt', ascending=False).groupby('BToKEE_event').filter(lambda g: any(g.BToKEE_decay == 0))
    df_pf['cumcount'] = df_pf.groupby('BToKEE_event').cumcount()
    df_pf_loc = df_pf[df_pf['BToKEE_decay'] == 0]['cumcount']

    df_low = df[low_selection].sort_values('BToKEE_fit_pt', ascending=False).groupby('BToKEE_event').filter(lambda g: any(g.BToKEE_decay == 0))
    df_low['cumcount'] = df_low.groupby('BToKEE_event').cumcount()
    df_low_loc = df_low[df_low['BToKEE_decay'] == 0]['cumcount']

    bins = np.linspace(0, 10, 10)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].hist(df_pf_loc, bins=bins, histtype='step', label='{0}, Mean: {1:.2f}'.format(get_label(0), np.mean(df_pf_loc)), normed=True)
    axes[0].set_ylabel("density")
    axes[0].set_xlabel("Position of truth combination of Kee")
    axes[0].legend()
    axes[0].set_title("PF-PF")
    axes[0].set_yscale("log", nonposy='clip')
    axes[1].hist(df_low_loc, bins=bins, histtype='step', label='{0}, Mean: {1:.2f}'.format(get_label(0), np.mean(df_low_loc)), normed=True)
    axes[1].set_ylabel("density")
    axes[1].set_xlabel("Position of truth combination of Kee")
    axes[1].legend()
    axes[1].set_title("LowPt-LwoPt")
    axes[1].set_yscale("log", nonposy='clip')
    fig.savefig('test.pdf', bbox_inches='tight')
    '''

    #print(df[trigger_selection].groupby('BToKMuMu_event').count()['BToKMuMu_fit_mass'])
    #print(df[trigger_selection])
    bins = np.linspace(0.0, 50.0, 50)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    #fig, axes = plt.subplots()
    #plot_duplicates(df[trigger_selection], 'BToKMuMu_fit_mass', bins=bins, ax=axes[0], title='Tag side')
    #plot_duplicates(df[np.logical_not(trigger_selection)], 'BToKMuMu_fit_mass', bins=bins, ax=axes[1], title='Probe side')
    #plot_duplicates(df[(df['BToKMuMu_nTriggerMuon'] - df['BToKMuMu_l1_isTriggering'] - df['BToKMuMu_l2_isTriggering']) > 0.0], 'BToKMuMu_fit_mass', bins=bins, ax=axes[1], title='Probe side')
    
    
    
    plot_duplicates(df, 'BToKEE_q2', bins=bins, ax=axes[0], title='All')
    print('plotting PF')
    plot_duplicates(df[pf_selection], 'BToKEE_q2', bins=bins, ax=axes[1], title='PF-PF')
    print('plotting lowPt')
    plot_duplicates(df[low_selection], 'BToKEE_q2', bins=bins, ax=axes[2], title='LowPt-LowPt')
    #plot_duplicates(df, 'BToKEE_fit_mass', bins=bins, ax=axes, title='LowPt-LowPt', logscale=False)
    #plot_duplicates(df[mix_net_selection], 'BToKEE_fit_mass', bins=bins, ax=axes[1], title='PF-LowPt')
    #plot_duplicates(df[low_pfveto_selection], 'BToKEE_fit_mass', bins=bins, ax=axes[2], title='LowPt-LowPt')
    fig.savefig('test.pdf', bbox_inches='tight')
    

    '''
    fig, axes = plt.subplots()
    df_pu = df.drop_duplicates(['BToKEE_event'], keep='first')
    plot_hist(df_pu, 'BToKEE_PV_npvsGood', bins=bins, ax=axes, title='')
    fig.savefig('test.pdf', bbox_inches='tight')
    '''

    '''
    bins_interp = np.linspace(0, 51, 51)
    weights = np.histogram(df_pu[(df_pu['Category'] == -1)], bins=bins_interp, density=True)[0] / np.histogram(df_pu[(df_pu['Category'] == 0)], bins=bins_interp, density=True)[0]
    print(weights, len(weights))
    f_weights = interp1d(bins, weights, fill_value="extrapolate")
    df_pu['weights'] = np.where((df_pu['Category'] == 0), f_weights(df_pu['BToKEE_PV_npvsGood']), 1.0)

    print(df_pu['weights'])
    fig, ax = plt.subplots()
    ax.hist(df_pu[(df_pu['Category'] == -1)]['BToKEE_PV_npvsGood'], bins=bins, histtype='step', label='{0}, Mean: {1:.2f}'.format('Data bkg', np.mean(df_pu[(df_pu['Category'] == -1)]['BToKEE_PV_npvsGood'])), normed=True)
    ax.hist(df_pu[(df_pu['Category'] == 0)]['BToKEE_PV_npvsGood'], bins=bins, histtype='step', label='{0}, Mean: {1:.2f}'.format('MC bkg: B+ to K+ ee', np.average(df_pu[(df_pu['Category'] == 0)]['BToKEE_PV_npvsGood'], weights=df_pu[(df_pu['Category'] == 0)]['weights'])), normed=True, weights=df_pu[(df_pu['Category'] == 0)]['weights'])
    ax.set_ylabel("density")
    ax.set_xlabel('BToKEE_PV_npvsGood')
    ax.legend()
    fig.savefig('test_2.pdf', bbox_inches='tight')
    '''


    '''
    with PdfPages(args.outputfile.replace('.pdf','')+'.pdf') as pdf:
      mvaCutList = np.linspace(9.0, 14.0, 20)
      for mvaCut in mvaCutList:
        print("plotting {}...".format(mvaCut))
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        plot_duplicates(df[(df['Category'] == 0) & (df['BToKEE_xgb'] > mvaCut)], 'BToKEE_fit_mass', bins=np.linspace(0.0, 5.0, 5), ax=axes[0], title='MVA: {0:.2f}, PF-PF'.format(mvaCut))
        plot_duplicates(df[(df['Category'] == 1) & (df['BToKEE_xgb'] > mvaCut)], 'BToKEE_fit_mass', bins=np.linspace(0.0, 5.0, 5), ax=axes[1], title='MVA: {0:.2f}, PF-LowPt'.format(mvaCut))
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    '''
    '''    
    with PdfPages(args.outputfile.replace('.pdf','')+'.pdf') as pdf:
      for var, bins in features.items():
        #if var != "ele_pt": continue
        print("plotting {}...".format(var))
        #fig, axes = plt.subplots()
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        plot_hist(df.sample(n=1000000), var, bins=bins, ax=axes[0], title='All')
        plot_hist(df[pf_selection], var, bins=bins, ax=axes[1], title='PF-PF')
        plot_hist(df[low_selection].sample(n=1000000), var, bins=bins, ax=axes[2], title='LowPt-LowPt')
        #fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        #plot_hist(df.query("abs(BToKEE_fit_eta) <= 1.57"), var, bins=bins, ax=axes[0], title="Barrel")
        #plot_hist(df, var, bins=bins, ax=axes[1], title="All")
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("finished plotting {}...".format(var))
    '''
    '''    
    training_features = ['BToKEE_fit_l1_normpt', 'BToKEE_fit_l1_eta', 'BToKEE_l1_dxy_sig',
              'BToKEE_fit_l2_normpt', 'BToKEE_fit_l2_eta', 'BToKEE_l2_dxy_sig',
              'BToKEE_fit_k_normpt', 'BToKEE_fit_k_eta', 'BToKEE_k_DCASig',
              'BToKEE_fit_normpt', 'BToKEE_fit_eta', 'BToKEE_svprob', 'BToKEE_fit_cos2D', 'BToKEE_l_xy_sig', 'BToKEE_dz',
              ]
    training_features += ['BToKEE_minDR', 'BToKEE_maxDR']
    training_features += ['BToKEE_l1_iso04_rel', 'BToKEE_l2_iso04_rel', 'BToKEE_k_iso04_rel', 'BToKEE_b_iso04_rel']
    training_features += ['BToKEE_l1_pfmvaId_lowPt', 'BToKEE_l2_pfmvaId_lowPt', 'BToKEE_l1_pfmvaId_highPt', 'BToKEE_l2_pfmvaId_highPt']
    training_features += ['BToKEE_ptImbalance']
    #training_features += ['BToKEE_l1_mvaId', 'BToKEE_l2_mvaId']

    plot_corr(df.query('Category == 0')[training_features], args.outputfile.replace('.pdf','') + '_corr_sig.pdf')
    plot_corr(df.query('Category == 1')[training_features], args.outputfile.replace('.pdf','') + '_corr_bkg.pdf')
    
    plot_pairgrid(df[training_features+['Category']].sample(n=1000), args.outputfile.replace('.pdf','') + '_pairgrid.pdf')
    '''



