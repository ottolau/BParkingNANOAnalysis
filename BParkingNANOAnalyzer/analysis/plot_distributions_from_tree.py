import matplotlib as mpl
mpl.use('pdf')
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

unneccesary_columns = []

def get_df(root_file_name):
    f = uproot.open(root_file_name)
    if len(f.allkeys()) == 0:
        return pd.DataFrame()
    df = uproot.open(root_file_name)["tree"].pandas.df()
    return df.drop(unneccesary_columns, axis=1)

def get_label(name):
    if name == -1:
        return "background"
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
    ax.set_ylabel("density")
    ax.set_xlabel(column)
    ax.legend()
    ax.set_title(title)
    if logscale:
        ax.set_yscale("log", nonposy='clip')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="A simple ttree plotter")
    parser.add_argument("-s", "--signal", dest="signal", default="BuToKJpsi_Toee_pfretrain_ntuple.root", help="Signal file")
    parser.add_argument("-b", "--background", dest="background", default="BuToKJpsi_Toee_pfretrain_ntuple.root", help="Background file")
    parser.add_argument("-o", "--outputfile", dest="outputfile", default="test.pdf", help="Output file")
    parser.add_argument("-f", "--suffix", dest="suffix", default=None, help="Suffix of the output name")
    args = parser.parse_args()


    features = OrderedDict([('BToKEE_mll_raw', np.linspace(0.0, 5.0, 100)),
                            ('BToKEE_mll_fullfit', np.linspace(0.0, 5.0, 100)),
                            ('BToKEE_mll_llfit', np.linspace(0.0, 5.0, 100)),
                            ('BToKEE_fit_mass', np.linspace(4.5, 6.0, 100)),
                            ('BToKEE_fit_massErr', np.linspace(0.0, 0.5, 100)),
                            ('BToKEE_fit_pt', np.linspace(0.0, 70.0, 100)),
                            ('BToKEE_fit_normpt', np.linspace(0.0, 30.0, 100)),
                            ('BToKEE_fit_eta', np.linspace(-3.0, 3.0, 100)),
                            ('BToKEE_fit_phi', np.linspace(-np.pi, np.pi, 100)),
                            ('BToKEE_b_iso03_rel', np.linspace(0.0, 10.0, 100)),
                            ('BToKEE_b_iso04_rel', np.linspace(0.0, 10.0, 100)),
                            ('BToKEE_fit_l1_pt', np.linspace(0.0, 40.0, 100)),
                            ('BToKEE_fit_l2_pt', np.linspace(0.0, 30.0, 100)),
                            ('BToKEE_fit_l1_normpt', np.linspace(0.0, 30.0, 100)),
                            ('BToKEE_fit_l2_normpt', np.linspace(0.0, 30.0, 100)),
                            ('BToKEE_fit_l1_eta', np.linspace(-3.0, 3.0, 100)),
                            ('BToKEE_fit_l2_eta', np.linspace(-3.0, 3.0, 100)),
                            ('BToKEE_fit_l1_phi', np.linspace(-np.pi, np.pi, 100)),
                            ('BToKEE_fit_l2_phi', np.linspace(-np.pi, np.pi, 100)),
                            ('BToKEE_l1_dxy_sig', np.linspace(-30.0, 30.0, 100)),
                            ('BToKEE_l2_dxy_sig', np.linspace(-30.0, 30.0, 100)),
                            ('BToKEE_l1_dz', np.linspace(-0.5, 0.5, 100)),
                            ('BToKEE_l2_dz', np.linspace(-0.5, 0.5, 100)),
                            #('BToKEE_l1_mvaId', np.linspace(-2.0, 10.0, 100)),
                            #('BToKEE_l2_mvaId', np.linspace(-2.0, 10.0, 100)),
                            ('BToKEE_l1_pfmvaId', np.linspace(-10.0, 10.0, 100)),
                            ('BToKEE_l2_pfmvaId', np.linspace(-10.0, 10.0, 100)),
                            ('BToKEE_l1_pfmvaId_lowPt', np.linspace(-10.0, 10.0, 100)),
                            ('BToKEE_l2_pfmvaId_lowPt', np.linspace(-10.0, 10.0, 100)),
                            ('BToKEE_l1_pfmvaId_highPt', np.linspace(-10.0, 10.0, 100)),
                            ('BToKEE_l2_pfmvaId_highPt', np.linspace(-10.0, 10.0, 100)),
                            ('BToKEE_l1_iso03_rel', np.linspace(0.0, 10.0, 100)),
                            ('BToKEE_l2_iso03_rel', np.linspace(0.0, 10.0, 100)),
                            ('BToKEE_l1_iso04_rel', np.linspace(0.0, 10.0, 100)),
                            ('BToKEE_l2_iso04_rel', np.linspace(0.0, 10.0, 100)),
                            ('BToKEE_fit_k_pt', np.linspace(0.0, 25.0, 100)),
                            ('BToKEE_fit_k_normpt', np.linspace(0.0, 10.0, 100)),
                            ('BToKEE_fit_k_eta', np.linspace(-3.0, 3.0, 100)),
                            ('BToKEE_fit_k_phi', np.linspace(-np.pi, np.pi, 100)),
                            ('BToKEE_k_DCASig', np.linspace(0.0, 30.0, 100)),
                            ('BToKEE_k_dz', np.linspace(-0.5, 0.5, 100)),
                            ('BToKEE_k_nValidHits', np.linspace(0.0, 40.0, 40)),
                            ('BToKEE_k_iso03_rel', np.linspace(0.0, 10.0, 100)),
                            ('BToKEE_k_iso04_rel', np.linspace(0.0, 10.0, 100)),
                            ('BToKEE_svprob', np.linspace(0.0, 1.0, 100)),
                            ('BToKEE_fit_cos2D', np.linspace(0.999, 1.0, 100)),
                            ('BToKEE_l_xy_sig', np.linspace(0.0, 100.0, 100)),
                            ('BToKEE_Dmass', np.linspace(0.5, 5.0, 100)),
                            ('BToKEE_maxDR', np.linspace(0.0, 3.0, 100)),
                            ('BToKEE_minDR', np.linspace(0.0, 3.0, 100)),
                            ])
    
    #n_files = 20
    #root_files = glob.glob("/eos/user/r/rembserj/ntuples/electron_mva_run3/*.root")[:n_files]
    #df = pd.concat((get_df(f) for f in  tqdm(root_files)), ignore_index=True)
    df1 = get_df(args.signal)
    df2 = get_df(args.background)
    #df2 = df2.sample(frac=1)[:50000]
    print('variables in ntuples: {}'.format(df1.columns))
    df1['Category'] = 0
    df2['Category'] = 1
    
    df = pd.concat((df1, df2), ignore_index=True)
    #df = df1.copy()

    with PdfPages(args.outputfile) as pdf:
      for var, bins in features.items():
        #if var != "ele_pt": continue
        print("plotting {}...".format(var))
        fig, axes = plt.subplots()
        plot_hist(df, var, bins=bins, ax=axes, title='')
        #fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        #plot_hist(df.query("abs(BToKEE_fit_eta) <= 1.57"), var, bins=bins, ax=axes[0], title="Barrel")
        #plot_hist(df, var, bins=bins, ax=axes[1], title="All")
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("finished plotting {}...".format(var))



