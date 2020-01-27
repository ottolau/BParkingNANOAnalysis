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
ROOT.gErrorIgnoreLevel=ROOT.kError
ROOT.RooMsgService.instance().setGlobalKillBelow(RooFit.FATAL)

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
        'legend.fontsize': 20,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'font.family' : 'lmodern',
        'text.latex.unicode': True,
        'axes.grid' : False,
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

    '''
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
  events = uproot.open(inputfile)['tree']
  params = events.arrays()
  branches = pd.DataFrame(params).sort_index(axis=1)

  output_branches = {}

  SList = []
  SErrList = []
  BList = []
  
  mvaCutList = np.linspace(2.0, 8.0, 20)
  for mvaCut in mvaCutList:
    # mva selection
    mva_selection = (branches['BToKEE_xgb'] > mvaCut)
    selected_branches = np.array(branches[mva_selection]['BToKEE_fit_mass'], dtype=[('BToKEE_fit_mass', 'f4')])
    tree = array2tree(selected_branches)
    S, SErr, B = fit_unbinned.fit(tree, outputfile + '_mva_{0:.3f}'.format(mvaCut).replace('.','-') + '.pdf', sigPDF=5, bkgPDF=2, doPartial=True, drawSNR=True, mvaCut=mvaCut)
    #S, SErr, B = fit_unbinned.fit(tree, outputfile + '_mva_{0:.3f}'.format(mvaCut).replace('.','-') + '.pdf', sigPDF=5, bkgPDF=2, isMC=True, drawSNR=True, mvaCut=mvaCut)
    print('MVA: {}, S: {}, B: {}, S/sqrt(S+B): {}'.format(mvaCut, S, B, S/np.sqrt(S+B)))
    SList.append(S)
    SErrList.append(SErr)
    BList.append(B)

  df_roc = pd.read_csv('pfretrain_results_testdf_reweighted_unBiased.csv')
  fpr = df_roc['fpr'].values
  tpr = df_roc['tpr'].values
  thresholds = df_roc['thresholds'].values
  mvaCut = np.linspace(0.0, 4.0, 10)
  wp_fpr = interp(mvaCut, thresholds[::-1], fpr[::-1])
  wp_tpr = interp(mvaCut, thresholds[::-1], tpr[::-1])

  fig, ax = plt.subplots()
  ax.plot(fpr, tpr, label="XGB")
  ax.plot(np.logspace(-4, 0, 1000), np.logspace(-4, 0, 1000), linestyle='--', color='k')
  ax.scatter(wp_fpr, wp_tpr, c='r', label="Working point")
  for i, mva in enumerate(mvaCut):
    ax.annotate(round(mva,2), (wp_fpr[i], wp_tpr[i]), fontsize=10, xytext=(10,-20), textcoords="offset points", arrowprops=dict(arrowstyle="->"))
  ax.set_xlim([1.0e-4, 1.0])
  ax.set_ylim([0.0, 1.0])
  ax.set_xscale('log')
  ax.set_xlabel("False Alarm Rate")
  ax.set_ylabel("Signal Efficiency")
  ax.set_title('Receiver Operating Curve')
  ax.legend(loc='upper left')
  fig.savefig('{}_roc_curve.pdf'.format(args.outputfile), bbox_inches='tight')

  SList = np.array(SList)
  SErrList = np.array(SErrList)
  BList = np.array(BList)
  CutBasedWP = {'S': 1561, 'B': 1097, 'SNR': 30.2} # PF
  #CutBasedWP = {'S': 759, 'B': 1394, 'SNR': 16.3} # Mix
  #CutBasedWP = {'S': 140, 'B': 285, 'SNR': 6.8} # Low

  SNRList = SList/np.sqrt(SList + BList)
  argmax_SNR = np.argmax(SNRList)
  print('Best SNR: {}, Best cut: {}'.format(np.max(SNRList), mvaCutList[argmax_SNR]))
  plotSNR(mvaCutList, SList, SErrList, BList, CutBasedWP)

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


