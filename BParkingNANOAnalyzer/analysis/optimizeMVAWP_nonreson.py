import uproot
import pandas as pd
import numpy as np
from scipy import interp
from rootpy.io import root_open
from rootpy.plotting import Hist
from root_numpy import fill_hist, array2root, array2tree
from root_pandas import to_root
import ROOT
from ROOT import RooFit
import makePlot_fitPeak_unbinned as fit_unbinned
import sys
sys.path.append('../')
from scripts.helper import *

ROOT.gErrorIgnoreLevel=ROOT.kError
ROOT.RooMsgService.instance().setGlobalKillBelow(RooFit.FATAL)

import matplotlib as mpl
mpl.use('agg')
import matplotlib.font_manager
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
        'legend.fontsize': 20,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'font.family' : 'lmodern',
        'text.latex.unicode': True,
        'axes.grid' : True,
        'text.usetex': True,
        'figure.figsize': fig_size}
plt.rcParams.update(params)


import argparse
parser = argparse.ArgumentParser(description="A simple ttree plotter")
parser.add_argument("-i", "--inputfile", dest="inputfile", default="DoubleMuonNtu_Run2016B.list", help="List of input ggNtuplizer files")
parser.add_argument("-o", "--outputfile", dest="outputfile", default="test", help="Output file containing plots")
parser.add_argument("-m", "--model", dest="model", default="dense_model_pf.h5", help="Trainned model")
parser.add_argument("-s", "--hist", dest="hist", action='store_true', help="Store histograms or tree")
args = parser.parse_args()


outputbranches = {'BToKEE_mll_raw': {'nbins': 50, 'xmin': 0.0, 'xmax': 5.0},
                  'BToKEE_mll_fullfit': {'nbins': 30, 'xmin': 2.6, 'xmax': 3.6},
                  'BToKEE_mll_llfit': {'nbins': 30, 'xmin': 2.6, 'xmax': 3.6},
                  'BToKEE_mass': {'nbins': 30, 'xmin': 4.7, 'xmax': 6.0},
                  'BToKEE_l1_pt': {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0},
                  'BToKEE_l2_pt': {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0},
                  'BToKEE_l1_normpt': {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0},
                  'BToKEE_l2_normpt': {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0},
                  'BToKEE_l1_eta': {'nbins': 50, 'xmin': -3.0, 'xmax': 3.0},
                  'BToKEE_l2_eta': {'nbins': 50, 'xmin': -3.0, 'xmax': 3.0},
                  'BToKEE_l1_phi': {'nbins': 50, 'xmin': -4.0, 'xmax': 4.0},
                  'BToKEE_l2_phi': {'nbins': 50, 'xmin': -4.0, 'xmax': 4.0},
                  'BToKEE_l1_dxy_sig': {'nbins': 50, 'xmin': -30.0, 'xmax': 30.0},
                  'BToKEE_l2_dxy_sig': {'nbins': 50, 'xmin': -30.0, 'xmax': 30.0},
                  'BToKEE_l1_dz': {'nbins': 50, 'xmin': -1.0, 'xmax': 1.0},
                  'BToKEE_l2_dz': {'nbins': 50, 'xmin': -1.0, 'xmax': 1.0},
                  'BToKEE_l1_unBiased': {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0},
                  'BToKEE_l2_unBiased': {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0},
                  'BToKEE_l1_ptBiased': {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0},
                  'BToKEE_l2_ptBiased': {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0},
                  'BToKEE_l1_mvaId': {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0},
                  'BToKEE_l2_mvaId': {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0},
                  'BToKEE_l1_isPF': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                  'BToKEE_l2_isPF': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                  'BToKEE_l1_isLowPt': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                  'BToKEE_l2_isLowPt': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                  'BToKEE_l1_isPFoverlap': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                  'BToKEE_l2_isPFoverlap': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                  'BToKEE_k_pt': {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0},
                  'BToKEE_k_normpt': {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0},
                  'BToKEE_k_eta': {'nbins': 50, 'xmin': -3.0, 'xmax': 3.0},
                  'BToKEE_k_phi': {'nbins': 50, 'xmin': -4.0, 'xmax': 4.0},
                  'BToKEE_k_DCASig': {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0},
                  'BToKEE_pt': {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0},
                  'BToKEE_normpt': {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0},
                  'BToKEE_svprob': {'nbins': 50, 'xmin': 0.0, 'xmax': 1.0},
                  'BToKEE_cos2D': {'nbins': 50, 'xmin': 0.999, 'xmax': 1.0},
                  'BToKEE_l_xy_sig': {'nbins': 50, 'xmin': 0.0, 'xmax': 50.0},
                  'BToKEE_keras_pf': {'nbins': 50, 'xmin': 0.0, 'xmax': 1.0},
                  }

def plotSNR(cut, sig, sigError, bkg, CutBasedWP):
    '''
    fig, ax1 = plt.subplots()
    #plt.grid(linestyle='--')
    snr_line, = ax1.plot(cut, sig/np.sqrt(sig+bkg), 'b-', label=r'$S/\sqrt{S+B}$')
    snr_cb_line = ax1.axhline(y=CutBasedWP['SNR'], color='g', linestyle='-.')
    ax1.set_xlabel(r'MVA Cut')
    ax1.set_ylabel(r'$S/\sqrt{S+B}$')
    #ax1.set_ylim(ymin=0)
    for t1 in ax1.get_yticklabels():
        t1.set_color('b')

    lower_bound = [s-serror for (s, serror) in zip(sig, sigError)]
    upper_bound = [s+serror for (s, serror) in zip(sig, sigError)]

    ax2 = ax1.twinx()
    s_line, = ax2.plot(cut, sig, 'r-', label=r'Number of signals')
    s_cb_line = ax2.axhline(y=CutBasedWP['S'], color='c', linestyle='-.')
    ax2.fill_between(cut, lower_bound, upper_bound, facecolor='yellow', alpha=0.5)
    ax2.set_ylabel(r'Number of signals')
    ax2.set_ylim(ymin=0)
    for t1 in ax2.get_yticklabels():
        t1.set_color('r')
    #ax2.legend(loc='upper left')
    #lns = lns1+lns2
    #labs = [l.get_label() for l in lns]
    #ax2.legend(lns, labs, loc='upper left')
    #fig.legend(loc=2, bbox_to_anchor=(0,1), bbox_transform=ax2.transAxes)
    handles = [snr_line, s_line, snr_cb_line, s_cb_line]
    labels = [r'Keras: $S/\sqrt{S+B}$', r'Keras: Number of signals', r'Cut-based: $S/\sqrt{S+B}$', r'Cut-based: Number of signals']
    #fig.legend(handles=handles, labels=labels, loc=1, bbox_to_anchor=(1,1), bbox_transform=ax2.transAxes)
    #fig.legend(handles=handles, labels=labels, loc=2, bbox_to_anchor=(0,1), bbox_transform=ax2.transAxes)
    fig.legend(handles=handles, labels=labels, loc=3, bbox_to_anchor=(0,0), bbox_transform=ax2.transAxes)
    fig.savefig('{}_SNRPlot.pdf'.format(args.outputfile), bbox_inches='tight')
    '''
    
    fig3, ax3 = plt.subplots()
    plt.grid(linestyle='--')
    ax3.plot(sig, sig/np.sqrt(sig+bkg), 'bo', label='XGB')
    #ax3.plot(CutBasedWP['S'], CutBasedWP['SNR'], 'r*', label='Cut-based')
    ax3.set_xlabel(r'S')
    ax3.set_ylabel(r'$S/\sqrt{S+B}$')
    ax3.legend(loc=2)
    fig3.savefig('{}_S_SNRPlot.pdf'.format(args.outputfile), bbox_inches='tight')


if __name__ == "__main__":
  inputfile = args.inputfile.replace('.root','').replace('.h5','')+'.root'
  outputfile = args.outputfile.replace('.root','').replace('.h5','')

  partial_resonant = 'part_workspace_resonant_low.root'
  partial_nonresonant = 'part_workspace_nonresonant_low.root' 
  drawSNR = True

  events = uproot.open(inputfile)['tree']
  params = events.arrays()
  branches = pd.DataFrame(params).sort_index(axis=1)

  output_branches = {}

  SList_R = []
  SErrList_R = []
  BList_R = []
  SList_NR = []
  SErrList_NR = []
  BList_NR = []

  mvaCutList = np.linspace(12.0, 17.0, 20)
  for mvaCut in mvaCutList:
    # mva selection
    mva_selection = (branches['BToKEE_xgb'] > mvaCut) #& (branches['BToKEE_l1_mvaId'] > 4.24) & (branches['BToKEE_l2_mvaId'] > 4.24)#& (branches['BToKEE_Dmass'] > 1.9)
    resonant_selection = (branches['BToKEE_mll_fullfit'] > JPSI_LOW) & (branches['BToKEE_mll_fullfit'] < JPSI_UP) #& ((branches['BToKEE_fit_mass'] < BLIND_LOW) | (branches['BToKEE_fit_mass'] > BLIND_UP))
    resonant_branches = np.array(branches[mva_selection & resonant_selection]['BToKEE_fit_mass'], dtype=[('BToKEE_fit_mass', 'f4')])
    resonant_tree = array2tree(resonant_branches)
    S_R, SErr_R, B_R = fit_unbinned.fit(resonant_tree, outputfile + '_resonant_mva_{0:.3f}'.format(mvaCut).replace('.','-') + '.pdf', doPartial=True, partialinputfile=partial_resonant, drawSNR=drawSNR, mvaCut=mvaCut, blinded=False)
    expS = S_R * BR_BToKLL / (BR_BToKJpsi * BR_JpsiToLL) * (12091.0/44024)

    nonresonant_selection = (branches['BToKEE_mll_fullfit'] > NR_LOW) & (branches['BToKEE_mll_fullfit'] < JPSI_LOW) #& ((branches['BToKEE_fit_mass'] < BLIND_LOW) | (branches['BToKEE_fit_mass'] > BLIND_UP))
    #nonresonant_selection = (branches['BToKEE_mll_fullfit'] > JPSI_LOW) & (branches['BToKEE_mll_fullfit'] < JPSI_UP) & ((branches['BToKEE_fit_mass'] < BLIND_LOW) | (branches['BToKEE_fit_mass'] > BLIND_UP))

    nonresonant_branches = np.array(branches[mva_selection & nonresonant_selection]['BToKEE_fit_mass'], dtype=[('BToKEE_fit_mass', 'f4')])
    nonresonant_tree = array2tree(nonresonant_branches)
    S_NR, SErr_NR, B_NR = fit_unbinned.fit(nonresonant_tree, outputfile + '_nonresonant_mva_{0:.3f}'.format(mvaCut).replace('.','-') + '.pdf', doPartial=True, partialinputfile=partial_nonresonant, drawSNR=drawSNR, mvaCut=mvaCut, blinded=True, expS=expS)
    #S_NR, SErr_NR, B_NR = fit_unbinned.fit(nonresonant_tree, outputfile + '_nonresonant_mva_{0:.3f}'.format(mvaCut).replace('.','-') + '.pdf', doPartial=True, partialinputfile=partial_resonant, drawSNR=True, mvaCut=mvaCut, blinded=True, expS=expS)


    print('MVA: {}\n\tResonant - S: {}, B: {}, S/sqrt(S+B): {}\n\tNon-resonant - S: {}, B: {}, S/sqrt(S+B): {}'.format(mvaCut, S_R, B_R, S_R/np.sqrt(S_R+B_R), S_NR, B_NR, S_NR/np.sqrt(S_NR+B_NR)))
    SList_R.append(S_R)
    SErrList_R.append(SErr_R)
    BList_R.append(B_R)

  SList_R = np.array(SList_R)
  SErrList_R = np.array(SErrList_R)
  BList_R = np.array(BList_R)
  SNR_R = SList_R / np.sqrt(SList_R + BList_R)

  df_roc = pd.read_csv('training_results_roc_csv_18Mar2020_fullq2_isoMVADRptImb_weighted_pauc02_low.csv')
  fpr = df_roc['fpr'].values
  tpr = df_roc['tpr'].values
  thresholds = df_roc['thresholds'].values
  wp_fpr = interp(mvaCutList, thresholds[::-1], fpr[::-1])
  wp_tpr = interp(mvaCutList, thresholds[::-1], tpr[::-1])

  fig, ax = plt.subplots()
  ax.plot(fpr, tpr, label="XGB")
  ax.plot(np.logspace(-5, 0, 1000), np.logspace(-5, 0, 1000), linestyle='--', color='k')
  ax.scatter(wp_fpr, wp_tpr, c='r', label="Working point")
  for i, mva in enumerate(mvaCutList):
    ax.annotate("{0:.2f}, SNR: {1:.1f}".format(mva, SNR_R[i]), (wp_fpr[i], wp_tpr[i]), fontsize=10, xytext=(10,-20), textcoords="offset points", arrowprops=dict(arrowstyle="->"))
  ax.set_xlim([1.0e-5, 1.0])
  ax.set_ylim([0.0, 1.0])
  ax.set_xscale('log')
  ax.set_xlabel("False Alarm Rate")
  ax.set_ylabel("Signal Efficiency")
  ax.set_title('Receiver Operating Curve')
  ax.legend(loc='lower right', fontsize=10)
  fig.savefig('{}_roc_curve.pdf'.format(args.outputfile), bbox_inches='tight')

  argmax_SNR_R = np.argmax(SNR_R)
  print('Best SNR: {}, Best cut: {}'.format(np.max(SNR_R), mvaCutList[argmax_SNR_R]))
  plotSNR(mvaCutList, SList_R, SErrList_R, BList_R, {})

  '''
  if args.hist:
    file_out = root_open('{}.root'.format(outputfile), 'recreate')
    hist_list = {hist_name: Hist(hist_bins['nbins'], hist_bins['xmin'], hist_bins['xmax'], name=hist_name, title='', type='F') for hist_name, hist_bins in sorted(outputbranches.items())}
    for hist_name, hist_bins in sorted(outputbranches.items()):
      if hist_name in branches.keys():
        branch_np = output_branches[hist_name].values
        fill_hist(hist_list[hist_name], branch_np[np.isfinite(branch_np)])
        hist_list[hist_name].write()
    file_out.close()

  else:
    output_branches[outputbranches.keys()].to_root('{}.root'.format(outputfile), key='tree')
  '''


