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
import itertools
import PyPDF2
import makePlot_fitPeak_unbinned as fit_unbinned
import os, sys, copy
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

def plotSNR(cut, results):
    fig, ax1 = plt.subplots()
    snr_jpsi, = ax1.plot(cut, results['SNR_jpsi'], 'bo', label=r'$J/\psi: S/\sqrt{S+B}$')
    ax1.set_xlabel(r'MVA Cut')
    ax1.set_ylabel(r'$J/\psi: S/\sqrt{S+B}$')
    ax1.yaxis.label.set_color('b')
    for t1 in ax1.get_yticklabels():
        t1.set_color('b')
    ax2 = ax1.twinx()
    snr_nonresonant, = ax2.plot(cut, results['SNR_nonresonant'], 'ro', label=r'Non-resonant: $S/\sqrt{S+B}$')
    #ax2.fill_between(cut, lower_bound, upper_bound, facecolor='yellow', alpha=0.5)
    ax2.set_ylabel(r'Non-resonant: $S/\sqrt{S+B}$')
    ax2.yaxis.label.set_color('r')
    for t1 in ax2.get_yticklabels():
        t1.set_color('r')
    handles = [snr_jpsi, snr_nonresonant]
    labels = [r'$J/\psi: S/\sqrt{S+B}$', r'Non-resonant: $S/\sqrt{S+B}$']
    #fig.legend(handles=handles, labels=labels, loc=1, bbox_to_anchor=(1,1), bbox_transform=ax2.transAxes)
    #fig.legend(handles=handles, labels=labels, loc=2, bbox_to_anchor=(0,1), bbox_transform=ax2.transAxes)
    fig.legend(handles=handles, labels=labels, loc=3, bbox_to_anchor=(0,0), bbox_transform=ax2.transAxes)
    fig.savefig('{}_SNRPlot.pdf'.format(args.outputfile), bbox_inches='tight')

    fig, ax1 = plt.subplots()
    lower_bound_jpsi = [s-serror for (s, serror) in zip(results['S_jpsi'], results['SErr_jpsi'])]
    upper_bound_jpsi = [s+serror for (s, serror) in zip(results['S_jpsi'], results['SErr_jpsi'])]
    lower_bound_nonresonant = [s-serror for (s, serror) in zip(results['S_nonresonant'], results['SErr_nonresonant'])]
    upper_bound_nonresonant = [s+serror for (s, serror) in zip(results['S_nonresonant'], results['SErr_nonresonant'])]
    s_jpsi, = ax1.plot(cut, results['S_jpsi'], 'b-', label=r'$J/\psi$: Number of signals')
    ax1.fill_between(cut, lower_bound_jpsi, upper_bound_jpsi, facecolor='blue', alpha=0.5)
    ax1.set_xlabel(r'MVA Cut')
    ax1.set_ylabel(r'$J/\psi$: Number of signals')
    ax1.yaxis.label.set_color('b')
    for t1 in ax1.get_yticklabels():
        t1.set_color('b')
    ax2 = ax1.twinx()
    s_nonresonant, = ax2.plot(cut, results['S_nonresonant'], 'r-', label=r'Non-resonant: Number of signals')
    ax2.fill_between(cut, lower_bound_nonresonant, upper_bound_nonresonant, facecolor='red', alpha=0.5)
    ax2.set_ylabel(r'Non-resonant: Number of signals')
    ax2.yaxis.label.set_color('r')
    for t1 in ax2.get_yticklabels():
        t1.set_color('r')
    handles = [s_jpsi, s_nonresonant]
    labels = [r'$J/\psi$: Number of signals', r'Non-resonant: Number of signals']
    fig.legend(handles=handles, labels=labels, loc=3, bbox_to_anchor=(0,0), bbox_transform=ax2.transAxes)
    fig.savefig('{}_SPlot.pdf'.format(args.outputfile), bbox_inches='tight')

    fig, ax1 = plt.subplots()
    b_jpsi, = ax1.plot(cut, results['B_jpsi'], 'b-', label=r'$J/\psi$: Number of background')
    ax1.set_xlabel(r'MVA Cut')
    ax1.set_ylabel(r'$J/\psi$: Number of background')
    ax1.yaxis.label.set_color('b')
    for t1 in ax1.get_yticklabels():
        t1.set_color('b')
    ax2 = ax1.twinx()
    b_nonresonant, = ax2.plot(cut, results['B_nonresonant'], 'r-', label=r'Non-resonant: Number of background')
    ax2.set_ylabel(r'Non-resonant: Number of background')
    ax2.yaxis.label.set_color('r')
    for t1 in ax2.get_yticklabels():
        t1.set_color('r')
    handles = [b_jpsi, b_nonresonant]
    labels = [r'$J/\psi$: Number of background', r'Non-resonant: Number of background']
    fig.legend(handles=handles, labels=labels, loc=1, bbox_to_anchor=(1,1), bbox_transform=ax2.transAxes)
    fig.savefig('{}_BPlot.pdf'.format(args.outputfile), bbox_inches='tight')

    fig, ax1 = plt.subplots()
    ax1.plot(cut, results['S_jpsi']/results['S_psi2s'])
    ax1.set_xlabel(r'MVA Cut')
    ax1.set_ylabel(r'$N(J/\psi) / N(\psi (2S))$')
    fig.savefig('{}_Jpsi2Psi2SRatio.pdf'.format(args.outputfile), bbox_inches='tight')

def pdf_combine(pdf_list, outputfile):
    merger = PyPDF2.PdfFileMerger()
    for pdf in pdf_list:
        merger.append(pdf)
    merger.write(outputfile)

def pdf_sidebyside(out, inputfile):
    output = PyPDF2.PdfFileWriter()
    reader = [PyPDF2.PdfFileReader(file(in1, "rb")) for in1 in inputfile]
    m = min([in1.getNumPages() for in1 in reader])
    print("common pages",m)
    for i in range(0,m):
        print "adding page common",i
        p = [in1.getPage(i) for in1 in reader]
        nPages = len(p)
        p1 = p[0]
        offset_x = 0.0
        for i, p2 in enumerate(p[1:]):
            offset_y = -(i+1)*p1.cropBox[1] + (i+1)*p1.cropBox[3]
            p1.mergeTranslatedPage(p2, offset_x, offset_y, expand=True)
        bounding_box = copy.deepcopy(p1.cropBox)
        p1.trimBox.lowerLeft = (bounding_box[0], bounding_box[1])
        p1.trimBox.upperRight = (bounding_box[2], bounding_box[1] + nPages*(bounding_box[3] - bounding_box[1]))
        p1.cropBox.lowerLeft = (bounding_box[0], bounding_box[1])
        p1.cropBox.upperRight = (bounding_box[2], bounding_box[1] + nPages*(bounding_box[3] - bounding_box[1]))
        output.addPage(p1)
    outputStream = file(out, "wb")
    output.write(outputStream)
    outputStream.close()

if __name__ == "__main__":
  inputfile = args.inputfile.replace('.root','').replace('.h5','')+'.root'
  outputfile = args.outputfile.replace('.root','').replace('.h5','')

  eleType = 'low'
  partial_jpsi = 'part_workspace_jpsi_{}.root'.format(eleType)
  partial_psi2s = 'part_workspace_psi2s_{}.root'.format(eleType)
  partial_nonresonant = 'part_workspace_nonresonant_{}.root'.format(eleType)
  psi2s_jpsi = 'psi2s_workspace_jpsi_{}.root'.format(eleType)
  params_jpsi = eval('params_jpsi_{}'.format(eleType))
  params_psi2s = eval('params_psi2s_{}'.format(eleType))
  params_nonresonant = eval('params_jpsi_{}'.format(eleType))

  drawSNR = True

  events = uproot.open(inputfile)['tree']
  params = events.arrays()
  branches = pd.DataFrame(params).sort_index(axis=1)

  output_branches = {}

  results = {'{}_{}'.format(quantity, region): [] for quantity, region in itertools.product(['S', 'SErr', 'B', 'SNR'], ['nonresonant', 'jpsi', 'psi2s'])}
  outputplots = {'jpsi': [],
                 'psi2s': [],
                 'nonresonant': [],
                 }

  mvaCutList = np.linspace(12.0, 15.0, 20)
  for mvaCut in mvaCutList:
    # mva selection
    mva_selection = (branches['BToKEE_xgb'] > mvaCut)

    # j/psi selection
    fit_params_jpsi = {'doPartial': True,
                       'partialinputfile': partial_jpsi,
                       'drawSNR': drawSNR,
                       'mvaCut': mvaCut,
                       'blinded': False,
                       'params': params_jpsi,
                       }

    jpsi_selection = (branches['BToKEE_mll_fullfit'] > JPSI_LOW) & (branches['BToKEE_mll_fullfit'] < JPSI_UP)
    jpsi_branches = np.array(branches[mva_selection & jpsi_selection]['BToKEE_fit_mass'], dtype=[('BToKEE_fit_mass', 'f4')])
    jpsi_tree = array2tree(jpsi_branches)

    outputname_jpsi = outputfile + '_jpsi_mva_{0:.3f}'.format(mvaCut).replace('.','-') + '.pdf'
    outputplots['jpsi'].append(outputname_jpsi)
    S_jpsi, SErr_jpsi, B_jpsi = fit_unbinned.fit(jpsi_tree, outputname_jpsi, **fit_params_jpsi)
    results['S_jpsi'].append(S_jpsi)
    results['SErr_jpsi'].append(SErr_jpsi)
    results['B_jpsi'].append(B_jpsi)
    results['SNR_jpsi'].append(S_jpsi/np.sqrt(S_jpsi + B_jpsi))

    expS = S_jpsi * BR_BToKLL / (BR_BToKJpsi * BR_JpsiToLL) * (517790.0/1678742) #(12091.0/44024)

    # psi(2s) selection
    fit_params_psi2s = {'doPartial': True,
                        'partialinputfile': partial_psi2s,
                        'drawSNR': drawSNR,
                        'mvaCut': mvaCut,
                        'blinded': False,
                        'params': params_jpsi,
                        'sigName': "B^{+}#rightarrow K^{+} #psi (2S)(#rightarrow e^{+}e^{-})",
                        }
   
    psi2s_selection = (branches['BToKEE_mll_fullfit'] > JPSI_UP) & (branches['BToKEE_mll_fullfit'] < PSI2S_UP)
    psi2s_branches = np.array(branches[mva_selection & psi2s_selection]['BToKEE_fit_mass'], dtype=[('BToKEE_fit_mass', 'f4')])
    psi2s_tree = array2tree(psi2s_branches)

    outputname_psi2s = outputfile + '_psi2s_mva_{0:.3f}'.format(mvaCut).replace('.','-') + '.pdf'
    outputplots['psi2s'].append(outputname_psi2s)
    S_psi2s, SErr_psi2s, B_psi2s = fit_unbinned.fit(psi2s_tree, outputname_psi2s, **fit_params_psi2s)
    results['S_psi2s'].append(S_psi2s)
    results['SErr_psi2s'].append(SErr_psi2s)
    results['B_psi2s'].append(B_psi2s)
    results['SNR_psi2s'].append(S_psi2s/np.sqrt(S_psi2s + B_psi2s))
    

    fit_params_nonresonant = {'doPartial': True,
                              'partialinputfile': partial_nonresonant,
                              'drawSNR': drawSNR,
                              'mvaCut': mvaCut,
                              'blinded': True,
                              'expS': expS,
                              'params': params_nonresonant,
                              'sigName': "B^{+}#rightarrow K^{+} e^{+}e^{-}",
                              }

    nonresonant_selection = (branches['BToKEE_mll_fullfit'] > NR_LOW) & (branches['BToKEE_mll_fullfit'] < JPSI_LOW)
    nonresonant_branches = np.array(branches[mva_selection & nonresonant_selection]['BToKEE_fit_mass'], dtype=[('BToKEE_fit_mass', 'f4')])
    nonresonant_tree = array2tree(nonresonant_branches)

    outputname_nonresonant = outputfile + '_nonresonant_mva_{0:.3f}'.format(mvaCut).replace('.','-') + '.pdf'
    outputplots['nonresonant'].append(outputname_nonresonant)
    S_NR, SErr_NR, B_NR = fit_unbinned.fit(nonresonant_tree, outputname_nonresonant, **fit_params_nonresonant)
    results['S_nonresonant'].append(S_NR)
    results['SErr_nonresonant'].append(SErr_NR)
    results['B_nonresonant'].append(B_NR)
    results['SNR_nonresonant'].append(S_NR/np.sqrt(S_NR + B_NR))

    print('MVA: {}\n\tJ/psi - S: {}, B: {}, S/sqrt(S+B): {}\n\tNon-resonant - S: {}, B: {}, S/sqrt(S+B): {}'.format(mvaCut, S_jpsi, B_jpsi, S_jpsi/np.sqrt(S_jpsi+B_jpsi), S_NR, B_NR, S_NR/np.sqrt(S_NR+B_NR)))

  
  outputname_jpsi = '{}_jpsi_{}.pdf'.format(outputfile, eleType)
  outputname_psi2s = '{}_psi2s_{}.pdf'.format(outputfile, eleType)
  outputname_nonresonant = '{}_nonresonant_{}.pdf'.format(outputfile, eleType)
  pdf_combine(outputplots['jpsi'], outputname_jpsi)
  pdf_combine(outputplots['psi2s'], outputname_psi2s)
  pdf_combine(outputplots['nonresonant'], outputname_nonresonant)
  pdf_sidebyside('{}_combined_{}.pdf'.format(outputfile, eleType), [outputname_jpsi, outputname_psi2s, outputname_nonresonant])

  map(lambda x: os.system('rm {}'.format(x)), outputplots['jpsi'])
  map(lambda x: os.system('rm {}'.format(x)), outputplots['psi2s'])
  map(lambda x: os.system('rm {}'.format(x)), outputplots['nonresonant'])

  results = {key: np.array(value) for key, value in results.items()}

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
    ax.annotate("{0:.2f}, SNR: {1:.1f}".format(mva, results['SNR_jpsi'][i]), (wp_fpr[i], wp_tpr[i]), fontsize=10, xytext=(10,-20), textcoords="offset points", arrowprops=dict(arrowstyle="->"))
  ax.set_xlim([1.0e-5, 1.0])
  ax.set_ylim([0.0, 1.0])
  ax.set_xscale('log')
  ax.set_xlabel("False Alarm Rate")
  ax.set_ylabel("Signal Efficiency")
  ax.set_title('Receiver Operating Curve')
  ax.legend(loc='lower right', fontsize=10)
  fig.savefig('{}_roc_curve.pdf'.format(args.outputfile), bbox_inches='tight')

  #argmax_SNR_R = np.argmax(SNR_R)
  #print('Best SNR: {}, Best cut: {}'.format(np.max(SNR_R), mvaCutList[argmax_SNR_R]))
  plotSNR(mvaCutList, results)

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


