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
import xgboost as xgb
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


def plotSNR(cut, results):
    fig, ax1 = plt.subplots()
    snr_jpsi, = ax1.plot(cut, results['SNR_jpsi'], 'bo', label=r'$J/\psi: S/\sqrt{S+B}$')
    ax1.set_xlabel(r'MVA Cut')
    ax1.set_ylabel(r'$J/\psi: S/\sqrt{S+B}$')
    ax1.yaxis.label.set_color('b')
    for t1 in ax1.get_yticklabels():
        t1.set_color('b')
    ax2 = ax1.twinx()
    snr_nonresonant_lowq2, = ax2.plot(cut, results['SNR_nonresonant_lowq2'], 'ro', label=r'Non-resonant: $S/\sqrt{S+B}$')
    #ax2.fill_between(cut, lower_bound, upper_bound, facecolor='yellow', alpha=0.5)
    ax2.set_ylabel(r'Non-resonant: $S/\sqrt{S+B}$')
    ax2.yaxis.label.set_color('r')
    for t1 in ax2.get_yticklabels():
        t1.set_color('r')
    handles = [snr_jpsi, snr_nonresonant_lowq2]
    labels = [r'$J/\psi: S/\sqrt{S+B}$', r'Non-resonant: $S/\sqrt{S+B}$']
    #fig.legend(handles=handles, labels=labels, loc=1, bbox_to_anchor=(1,1), bbox_transform=ax2.transAxes)
    #fig.legend(handles=handles, labels=labels, loc=2, bbox_to_anchor=(0,1), bbox_transform=ax2.transAxes)
    fig.legend(handles=handles, labels=labels, loc=3, bbox_to_anchor=(0,0), bbox_transform=ax2.transAxes)
    fig.savefig('{}_SNRPlot.pdf'.format(args.outputfile), bbox_inches='tight')

    fig, ax1 = plt.subplots()
    lower_bound_jpsi = [s-serror for (s, serror) in zip(results['S_jpsi'], results['SErr_jpsi'])]
    upper_bound_jpsi = [s+serror for (s, serror) in zip(results['S_jpsi'], results['SErr_jpsi'])]
    lower_bound_nonresonant = [s-serror for (s, serror) in zip(results['S_nonresonant_lowq2'], results['SErr_nonresonant_lowq2'])]
    upper_bound_nonresonant = [s+serror for (s, serror) in zip(results['S_nonresonant_lowq2'], results['SErr_nonresonant_lowq2'])]
    s_jpsi, = ax1.plot(cut, results['S_jpsi'], 'b-', label=r'$J/\psi$: Number of signals')
    ax1.fill_between(cut, lower_bound_jpsi, upper_bound_jpsi, facecolor='blue', alpha=0.5)
    ax1.set_xlabel(r'MVA Cut')
    ax1.set_ylabel(r'$J/\psi$: Number of signals')
    ax1.yaxis.label.set_color('b')
    for t1 in ax1.get_yticklabels():
        t1.set_color('b')
    ax2 = ax1.twinx()
    s_nonresonant_lowq2, = ax2.plot(cut, results['S_nonresonant_lowq2'], 'r-', label=r'Non-resonant: Number of signals')
    ax2.fill_between(cut, lower_bound_nonresonant, upper_bound_nonresonant, facecolor='red', alpha=0.5)
    ax2.set_ylabel(r'Non-resonant: Number of signals')
    ax2.yaxis.label.set_color('r')
    for t1 in ax2.get_yticklabels():
        t1.set_color('r')
    handles = [s_jpsi, s_nonresonant_lowq2]
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
    b_nonresonant_lowq2, = ax2.plot(cut, results['B_nonresonant_lowq2'], 'r-', label=r'Non-resonant: Number of background')
    ax2.set_ylabel(r'Non-resonant: Number of background')
    ax2.yaxis.label.set_color('r')
    for t1 in ax2.get_yticklabels():
        t1.set_color('r')
    handles = [b_jpsi, b_nonresonant_lowq2]
    labels = [r'$J/\psi$: Number of background', r'Non-resonant: Number of background']
    fig.legend(handles=handles, labels=labels, loc=1, bbox_to_anchor=(1,1), bbox_transform=ax2.transAxes)
    fig.savefig('{}_BPlot.pdf'.format(args.outputfile), bbox_inches='tight')

    fig, ax1 = plt.subplots()
    ax1.plot(cut, results['S_jpsi']/results['S_psi2s'])
    ax1.set_xlabel(r'MVA Cut')
    ax1.set_ylabel(r'$N(J/\psi) / N(\psi (2S))$')
    fig.savefig('{}_Jpsi2Psi2SRatio.pdf'.format(args.outputfile), bbox_inches='tight')
    
    fig, ax1 = plt.subplots()
    ax1.plot(cut, results['eff_jpsi'], 'b-', label=r'$J/\psi$')
    ax1.plot(cut, results['eff_nonresonant_lowq2'], 'r-', label=r'Non-resonant')
    ax1.set_xlabel(r'MVA Cut')
    ax1.set_ylabel(r'MVA efficiency')
    ax1.legend(loc='upper right')
    fig.savefig('{}_mvaEfficiency.pdf'.format(args.outputfile), bbox_inches='tight')

    fig, ax1 = plt.subplots()
    scl_jpsi, = ax1.plot(cut, results['S_jpsi']/results['eff_jpsi'], 'b-', label=r'$J/\psi$')
    ax1.set_xlabel(r'MVA Cut')
    ax1.set_ylabel(r'$J/\psi: S/\epsilon_{\rm MVA}$')
    ax1.yaxis.label.set_color('b')
    for t1 in ax1.get_yticklabels():
        t1.set_color('b')
    ax2 = ax1.twinx()
    scl_nonresonant_lowq2, = ax2.plot(cut, results['S_nonresonant_lowq2']/results['eff_nonresonant_lowq2'], 'r-', label=r'Non-resonant')
    ax2.set_ylabel(r'Non-resonant: $S/\epsilon_{\rm MVA}$')
    ax2.yaxis.label.set_color('r')
    for t1 in ax2.get_yticklabels():
        t1.set_color('r')
    handles = [scl_jpsi, scl_nonresonant_lowq2]
    labels = [r'$J/\psi$', r'Non-resonant']
    fig.legend(handles=handles, labels=labels, loc=1, bbox_to_anchor=(1,1), bbox_transform=ax2.transAxes)
    fig.savefig('{}_scaledS.pdf'.format(args.outputfile), bbox_inches='tight')


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

  eleType = 'pf'
  partial_jpsi = 'part_workspace_jpsi_{}.root'.format(eleType)
  partial_psi2s = 'part_workspace_psi2s_{}.root'.format(eleType)
  partial_nonresonant_lowq2 = 'part_workspace_nonresonant_{}.root'.format(eleType)
  partial_nonresonant_highq2 = 'part_workspace_nonresonant_{}.root'.format(eleType)

  #psi2s_jpsi = 'psi2s_workspace_jpsi_{}.root'.format(eleType)
  jpsi_psi2s = 'jpsi_workspace_psi2s_{}.root'.format(eleType)
  #jpsi_nonresonant = 'jpsi_workspace_nonresonant_{}.root'.format(eleType)

  params_jpsi = eval('params_jpsi_{}'.format(eleType))
  params_jpsitri = eval('params_jpsitri_{}'.format(eleType))
  params_psi2s = eval('params_psi2s_{}'.format(eleType))
  params_psi2stri = eval('params_psi2stri_{}'.format(eleType))
  params_nonresonant_lowq2 = eval('params_jpsi_{}'.format(eleType))
  params_nonresonant_highq2 = eval('params_jpsi_{}'.format(eleType))

  #jpsi_mc = 'RootTree_2020Jan16_BuToKJpsi_Toee_BToKEEAnalyzer_2020Apr17_mc_mva_{}.root'.format(eleType)
  #nonresonant_mc = 'RootTree_2020Jan16_BuToKee_all_BToKEEAnalyzer_2020Apr17_mc_mva_{}.root'.format(eleType)
  jpsi_mc = 'BParkingNANO_2020Jan16_BuToKJpsi_Toee_BToKEEAnalyzer_2020May03_newVar_mc.root'
  nonresonant_mc = 'BParkingNANO_2020Jan16_BuToKee_all_BToKEEAnalyzer_2020May03_newVar_mc.root'

  data_branches = ['BToKEE_mll_fullfit', 'BToKEE_fit_mass', 'BToKEE_mva', 'BToKEE_event']

  features = ['BToKEE_fit_l1_normpt', 'BToKEE_l1_dxy_sig',
              'BToKEE_fit_l2_normpt', 'BToKEE_l2_dxy_sig',
              'BToKEE_fit_k_normpt', 'BToKEE_k_DCASig',
              'BToKEE_fit_normpt', 'BToKEE_svprob', 'BToKEE_fit_cos2D', 'BToKEE_l_xy_sig', 'BToKEE_dz'
              ]
  #features += ['BToKEE_fit_l1_eta', 'BToKEE_fit_l2_eta', 'BToKEE_fit_k_eta', 'BToKEE_fit_eta']
  features += ['BToKEE_eleDR', 'BToKEE_llkDR']
  features += ['BToKEE_l1_iso04_rel', 'BToKEE_l2_iso04_rel', 'BToKEE_k_iso04_rel', 'BToKEE_b_iso04_rel']
  features += ['BToKEE_ptImbalance']
  features += ['BToKEE_l1_pfmvaId_lowPt', 'BToKEE_l2_pfmvaId_lowPt', 'BToKEE_l1_pfmvaId_highPt', 'BToKEE_l2_pfmvaId_highPt']
  #features += ['BToKEE_Dmass', 'BToKEE_Dmass_flip']
  #features += ['BToKEE_svprob_rank', 'BToKEE_fit_pt_rank', 'BToKEE_fit_cos2D_rank', 'BToKEE_l_xy_rank']

  training_branches = sorted(features)
  ntree_limit = 1196
  modelfile = '../models/xgb_fulldata_06May2020_allq2_isoPFMVANewDRptImb_weighted_11lowq2_pauc02_pf.model'
  model = xgb.Booster({'nthread': 6})
  model.load_model(modelfile)


  mc_branches = ['BToKEE_mll_fullfit', 'BToKEE_l1_isPF', 'BToKEE_l2_isPF', 'BToKEE_l1_pfmvaId', 'BToKEE_l2_pfmvaId'] + training_branches

  drawSNR = True

  #events = uproot.open(inputfile)['tree']
  #params = events.arrays(branches=data_branches)
  #branches = pd.DataFrame(params).sort_index(axis=1)
  branches = get_df(inputfile, branches=data_branches)

  mll_mean_jpsi = triCut_jpsi_mll_mean_pf
  fit_mass_mean_jpsi = triCut_jpsi_mKee_mean_pf
  triCut_jpsi_lower_bound = triCut_jpsi_lower_bound_pf
  triCut_jpsi_upper_bound = triCut_jpsi_upper_bound_pf
  branches['BToKEE_mll_fullfit_centered_jpsi'] = branches['BToKEE_mll_fullfit'] - mll_mean_jpsi
  branches['BToKEE_fit_mass_centered_jpsi'] = branches['BToKEE_fit_mass'] - fit_mass_mean_jpsi
  data_centered_jpsi = np.array([branches['BToKEE_fit_mass_centered_jpsi'],branches['BToKEE_mll_fullfit_centered_jpsi']]).T
  eigVecs_jpsi = triCut_jpsi_rotMatrix_pf 
  data_decorr_jpsi = data_centered_jpsi.dot(eigVecs_jpsi)
  branches['BToKEE_fit_mass_decorr_jpsi'] = data_decorr_jpsi[:,0]
  branches['BToKEE_mll_fullfit_decorr_jpsi'] = data_decorr_jpsi[:,1]

  mll_mean_psi2s = triCut_psi2s_mll_mean_pf
  fit_mass_mean_psi2s = triCut_psi2s_mKee_mean_pf
  triCut_psi2s_lower_bound = triCut_psi2s_lower_bound_pf
  triCut_psi2s_upper_bound = triCut_psi2s_upper_bound_pf
  branches['BToKEE_mll_fullfit_centered_psi2s'] = branches['BToKEE_mll_fullfit'] - mll_mean_psi2s
  branches['BToKEE_fit_mass_centered_psi2s'] = branches['BToKEE_fit_mass'] - fit_mass_mean_psi2s
  data_centered_psi2s = np.array([branches['BToKEE_fit_mass_centered_psi2s'],branches['BToKEE_mll_fullfit_centered_psi2s']]).T
  eigVecs_psi2s = triCut_psi2s_rotMatrix_pf 
  data_decorr_psi2s = data_centered_psi2s.dot(eigVecs_psi2s)
  branches['BToKEE_fit_mass_decorr_psi2s'] = data_decorr_psi2s[:,0]
  branches['BToKEE_mll_fullfit_decorr_psi2s'] = data_decorr_psi2s[:,1]

  jpsi_mc_branches = pd.DataFrame(uproot.open(jpsi_mc)['tree'].arrays(branches=mc_branches)).sort_index(axis=1).query('(BToKEE_l1_isPF == True) and (BToKEE_l2_isPF == True)')
  #jpsi_mc_branches = pd.DataFrame(uproot.open(jpsi_mc)['tree'].arrays(branches=mc_branches)).sort_index(axis=1).query('(BToKEE_l1_isPF == True) and (BToKEE_l2_isPF == True) and (BToKEE_l1_pfmvaId > -2.0) and (BToKEE_l2_pfmvaId > -2.0)')
  jpsi_mc_branches['BToKEE_mva'] = model.predict(xgb.DMatrix(jpsi_mc_branches[training_branches].replace([np.inf, -np.inf], 0.0).sort_index(axis=1)), ntree_limit=ntree_limit)
  jpsi_mc_branches = jpsi_mc_branches[(jpsi_mc_branches['BToKEE_mll_fullfit'] > JPSI_LOW) & (jpsi_mc_branches['BToKEE_mll_fullfit'] < JPSI_UP)]
  nTot_jpsi = float(jpsi_mc_branches.shape[0])

  nonresonant_mc_branches = pd.DataFrame(uproot.open(nonresonant_mc)['tree'].arrays(branches=mc_branches)).sort_index(axis=1).query('(BToKEE_l1_isPF == True) and (BToKEE_l2_isPF == True)')
  #nonresonant_mc_branches = pd.DataFrame(uproot.open(nonresonant_mc)['tree'].arrays(branches=mc_branches)).sort_index(axis=1).query('(BToKEE_l1_isPF == True) and (BToKEE_l2_isPF == True) and (BToKEE_l1_pfmvaId > -2.0) and (BToKEE_l2_pfmvaId > -2.0)')
  nonresonant_mc_branches['BToKEE_mva'] = model.predict(xgb.DMatrix(nonresonant_mc_branches[training_branches].replace([np.inf, -np.inf], 0.0).sort_index(axis=1)), ntree_limit=ntree_limit)
  nonresonant_lowq2_mc_branches = nonresonant_mc_branches[(nonresonant_mc_branches['BToKEE_mll_fullfit'] > NR_LOW) & (nonresonant_mc_branches['BToKEE_mll_fullfit'] < JPSI_LOW)]
  nonresonant_highq2_mc_branches = nonresonant_mc_branches[(nonresonant_mc_branches['BToKEE_mll_fullfit'] > PSI2S_UP) & (nonresonant_mc_branches['BToKEE_mll_fullfit'] < NR_UP)]
  nTot_nonresonant_lowq2 = float(nonresonant_lowq2_mc_branches.shape[0])
  nTot_nonresonant_highq2 = float(nonresonant_highq2_mc_branches.shape[0])
  nTot_nonresonant = float(nonresonant_mc_branches.shape[0])

  print(nTot_jpsi, nTot_nonresonant)

  print(nTot_nonresonant_lowq2, nTot_nonresonant_highq2, nTot_nonresonant)
  output_branches = {}

  results = {'{}_{}'.format(quantity, region): [] for quantity, region in itertools.product(['S', 'SErr', 'B', 'SNR', 'eff'], ['nonresonant_lowq2', 'nonresonant_highq2', 'jpsi', 'jpsitri', 'psi2s', 'psi2stri'])}
  outputplots = {'jpsi': [],
                 'jpsitri': [],
                 'psi2s': [],
                 'psi2stri': [],
                 'nonresonant_lowq2': [],
                 'nonresonant_highq2': [],
                 }

  mvaCutList = np.linspace(10.0, 14.0, 30)
  for mvaCut in mvaCutList:
    # mva selection
    mva_selection = (branches['BToKEE_mva'] > mvaCut)
    selected_branches = branches[mva_selection].sort_values('BToKEE_mva', ascending=False).drop_duplicates(['BToKEE_event'], keep='first')

    # j/psi selection with triangular cut on q2
    fit_params_jpsitri = {'doPartial': False,
                          'drawSNR': drawSNR,
                          'mvaCut': mvaCut,
                          'blinded': False,
                          'params': params_jpsitri,
                          }

    jpsitri_selection = (selected_branches['BToKEE_mll_fullfit_decorr_jpsi'] < triCut_jpsi_upper_bound) & (selected_branches['BToKEE_mll_fullfit'] > JPSI_LOW) & (selected_branches['BToKEE_mll_fullfit'] < JPSI_UP)
    jpsitri_branches = np.array(selected_branches[jpsitri_selection]['BToKEE_fit_mass'], dtype=[('BToKEE_fit_mass', 'f4')])
    jpsitri_tree = array2tree(jpsitri_branches)

    outputname_jpsitri = outputfile + '_jpsi_triCut_mva_{0:.3f}'.format(mvaCut).replace('.','-') + '.pdf'
    outputplots['jpsitri'].append(outputname_jpsitri)
    S_jpsitri, SErr_jpsitri, B_jpsitri = fit_unbinned.fit(jpsitri_tree, outputname_jpsitri, **fit_params_jpsitri)
    results['S_jpsitri'].append(S_jpsitri)
    results['SErr_jpsitri'].append(SErr_jpsitri)
    results['B_jpsitri'].append(B_jpsitri)
    results['SNR_jpsitri'].append(S_jpsitri/np.sqrt(S_jpsitri + B_jpsitri))

    # j/psi selection
    fit_params_jpsi = {'doPartial': True,
                       'partialinputfile': partial_jpsi,
                       'drawSNR': drawSNR,
                       'mvaCut': mvaCut,
                       'blinded': False,
                       'params': params_jpsi,
                       }

    jpsi_selection = (selected_branches['BToKEE_mll_fullfit'] > JPSI_LOW) & (selected_branches['BToKEE_mll_fullfit'] < JPSI_UP)
    jpsi_branches = np.array(selected_branches[jpsi_selection]['BToKEE_fit_mass'], dtype=[('BToKEE_fit_mass', 'f4')])
    jpsi_tree = array2tree(jpsi_branches)

    outputname_jpsi = outputfile + '_jpsi_mva_{0:.3f}'.format(mvaCut).replace('.','-') + '.pdf'
    outputplots['jpsi'].append(outputname_jpsi)
    S_jpsi, SErr_jpsi, B_jpsi = fit_unbinned.fit(jpsi_tree, outputname_jpsi, **fit_params_jpsi)
    results['S_jpsi'].append(S_jpsi)
    results['SErr_jpsi'].append(SErr_jpsi)
    results['B_jpsi'].append(B_jpsi)
    results['SNR_jpsi'].append(S_jpsi/np.sqrt(S_jpsi + B_jpsi))

    eff_jpsi = float(jpsi_mc_branches[(jpsi_mc_branches['BToKEE_mva'] > mvaCut)].shape[0]) / nTot_jpsi
    eff_nonresonant_lowq2 = float(nonresonant_lowq2_mc_branches[(nonresonant_lowq2_mc_branches['BToKEE_mva'] > mvaCut)].shape[0]) / nTot_nonresonant_lowq2
    eff_nonresonant_highq2 = float(nonresonant_highq2_mc_branches[(nonresonant_highq2_mc_branches['BToKEE_mva'] > mvaCut)].shape[0]) / nTot_nonresonant_highq2
    #nTot_nonresonant = float(nonresonant_mc_branches[(nonresonant_mc_branches['BToKEE_mva'] > mvaCut)].shape[0])
    #eff_nonresonant = float(nonresonant_mc_branches[(nonresonant_mc_branches['BToKEE_mva'] > mvaCut) & (nonresonant_mc_branches['BToKEE_mll_fullfit'] > NR_LOW) & (nonresonant_mc_branches['BToKEE_mll_fullfit'] < JPSI_UP)].shape[0]) / nTot_nonresonant
    results['eff_jpsi'].append(eff_jpsi)
    results['eff_nonresonant_lowq2'].append(eff_nonresonant_lowq2)
    results['eff_nonresonant_highq2'].append(eff_nonresonant_highq2)

    expS_lowq2 = S_jpsi * BR_BToKLL / (BR_BToKJpsi * BR_JpsiToLL) * (nTot_nonresonant_lowq2 / nTot_nonresonant) * (eff_nonresonant_lowq2 / eff_jpsi)#(12091.0/44024)
    expS_highq2 = S_jpsi * BR_BToKLL / (BR_BToKJpsi * BR_JpsiToLL) * (nTot_nonresonant_highq2 / nTot_nonresonant) * (eff_nonresonant_highq2 / eff_jpsi)#(12091.0/44024)

    # psi(2s) selection with triangular cut on q2
    fit_params_psi2stri = {'doPartial': False,
                           'drawSNR': drawSNR,
                           'mvaCut': mvaCut,
                           'blinded': False,
                           'params': params_psi2stri,
                           'sigName': "B^{+}#rightarrow K^{+} #psi (2S)(#rightarrow e^{+}e^{-})",
                           'doJpsi': True,
                           'jpsiinputfile': jpsi_psi2s,
                           }
   
    psi2stri_selection = (selected_branches['BToKEE_mll_fullfit_decorr_psi2s'] < triCut_psi2s_upper_bound) & (selected_branches['BToKEE_mll_fullfit'] > JPSI_UP) & (selected_branches['BToKEE_mll_fullfit'] < PSI2S_UP)
    psi2stri_branches = np.array(selected_branches[psi2stri_selection]['BToKEE_fit_mass'], dtype=[('BToKEE_fit_mass', 'f4')])
    psi2stri_tree = array2tree(psi2stri_branches)

    outputname_psi2stri = outputfile + '_psi2s_triCut_mva_{0:.3f}'.format(mvaCut).replace('.','-') + '.pdf'
    outputplots['psi2stri'].append(outputname_psi2stri)
    S_psi2stri, SErr_psi2stri, B_psi2stri = fit_unbinned.fit(psi2stri_tree, outputname_psi2stri, **fit_params_psi2stri)
    results['S_psi2stri'].append(S_psi2stri)
    results['SErr_psi2stri'].append(SErr_psi2stri)
    results['B_psi2stri'].append(B_psi2stri)
    results['SNR_psi2stri'].append(S_psi2stri/np.sqrt(S_psi2stri + B_psi2stri))

    # psi(2s) selection
    fit_params_psi2s = {'doPartial': True,
                        'partialinputfile': partial_psi2s,
                        'drawSNR': drawSNR,
                        'mvaCut': mvaCut,
                        'blinded': False,
                        'params': params_psi2s,
                        'sigName': "B^{+}#rightarrow K^{+} #psi (2S)(#rightarrow e^{+}e^{-})",
                        'doJpsi': True,
                        'jpsiinputfile': jpsi_psi2s,
                        }
   
    psi2s_selection = (selected_branches['BToKEE_mll_fullfit'] > JPSI_UP) & (selected_branches['BToKEE_mll_fullfit'] < PSI2S_UP)
    psi2s_branches = np.array(selected_branches[psi2s_selection]['BToKEE_fit_mass'], dtype=[('BToKEE_fit_mass', 'f4')])
    psi2s_tree = array2tree(psi2s_branches)

    outputname_psi2s = outputfile + '_psi2s_mva_{0:.3f}'.format(mvaCut).replace('.','-') + '.pdf'
    outputplots['psi2s'].append(outputname_psi2s)
    S_psi2s, SErr_psi2s, B_psi2s = fit_unbinned.fit(psi2s_tree, outputname_psi2s, **fit_params_psi2s)
    results['S_psi2s'].append(S_psi2s)
    results['SErr_psi2s'].append(SErr_psi2s)
    results['B_psi2s'].append(B_psi2s)
    results['SNR_psi2s'].append(S_psi2s/np.sqrt(S_psi2s + B_psi2s))
    
    # low q2 non-resonant selection
    fit_params_nonresonant_lowq2 = {'doPartial': True,
                                    'partialinputfile': partial_nonresonant_lowq2,
                                    'drawSNR': drawSNR,
                                    'mvaCut': mvaCut,
                                    'blinded': True,
                                    'expS': expS_lowq2,
                                    'params': params_nonresonant_lowq2,
                                    'sigName': "B^{+}#rightarrow K^{+} e^{+}e^{-}",
                                    }

    nonresonant_lowq2_selection = (selected_branches['BToKEE_mll_fullfit'] > NR_LOW) & (selected_branches['BToKEE_mll_fullfit'] < JPSI_LOW)
    nonresonant_lowq2_branches = np.array(selected_branches[nonresonant_lowq2_selection]['BToKEE_fit_mass'], dtype=[('BToKEE_fit_mass', 'f4')])
    nonresonant_lowq2_tree = array2tree(nonresonant_lowq2_branches)

    outputname_nonresonant_lowq2 = outputfile + '_nonresonant_lowq2_mva_{0:.3f}'.format(mvaCut).replace('.','-') + '.pdf'
    outputplots['nonresonant_lowq2'].append(outputname_nonresonant_lowq2)
    S_NR_lowq2, SErr_NR_lowq2, B_NR_lowq2 = fit_unbinned.fit(nonresonant_lowq2_tree, outputname_nonresonant_lowq2, **fit_params_nonresonant_lowq2)
    results['S_nonresonant_lowq2'].append(S_NR_lowq2)
    results['SErr_nonresonant_lowq2'].append(SErr_NR_lowq2)
    results['B_nonresonant_lowq2'].append(B_NR_lowq2)
    results['SNR_nonresonant_lowq2'].append(S_NR_lowq2/np.sqrt(S_NR_lowq2 + B_NR_lowq2))

    # high q2 non-resonant selection
    fit_params_nonresonant_highq2 = {'doPartial': False,
                                    'drawSNR': drawSNR,
                                    'mvaCut': mvaCut,
                                    'blinded': True,
                                    'expS': expS_highq2,
                                    'params': params_nonresonant_highq2,
                                    'sigName': "B^{+}#rightarrow K^{+} e^{+}e^{-}",
                                    }

    nonresonant_highq2_selection = (selected_branches['BToKEE_mll_fullfit'] > PSI2S_UP) & (selected_branches['BToKEE_mll_fullfit'] < NR_UP)
    nonresonant_highq2_branches = np.array(selected_branches[nonresonant_highq2_selection]['BToKEE_fit_mass'], dtype=[('BToKEE_fit_mass', 'f4')])
    nonresonant_highq2_tree = array2tree(nonresonant_highq2_branches)

    outputname_nonresonant_highq2 = outputfile + '_nonresonant_highq2_mva_{0:.3f}'.format(mvaCut).replace('.','-') + '.pdf'
    outputplots['nonresonant_highq2'].append(outputname_nonresonant_highq2)
    S_NR_highq2, SErr_NR_highq2, B_NR_highq2 = fit_unbinned.fit(nonresonant_highq2_tree, outputname_nonresonant_highq2, **fit_params_nonresonant_highq2)
    results['S_nonresonant_highq2'].append(S_NR_highq2)
    results['SErr_nonresonant_highq2'].append(SErr_NR_highq2)
    results['B_nonresonant_highq2'].append(B_NR_highq2)
    results['SNR_nonresonant_highq2'].append(S_NR_highq2/np.sqrt(S_NR_highq2 + B_NR_highq2))


    print("="*80)
    print('MVA: {}\n\tJ/psi - S: {}, B: {}, S/sqrt(S+B): {}\n\tNon-resonant - S: {}, B: {}, S/sqrt(S+B): {}'.format(mvaCut, S_jpsi, B_jpsi, S_jpsi/np.sqrt(S_jpsi+B_jpsi), S_NR_lowq2, B_NR_lowq2, S_NR_lowq2/np.sqrt(S_NR_lowq2+B_NR_lowq2)))
    print('Jpsi eff: {}, non-resonant eff: {}'.format(eff_jpsi, eff_nonresonant_lowq2))
    print("="*80)
  
  outputname_jpsi = '{}_jpsi_{}.pdf'.format(outputfile, eleType)
  outputname_jpsitri = '{}_jpsi_triCut_{}.pdf'.format(outputfile, eleType)
  outputname_psi2stri = '{}_psi2s_triCut_{}.pdf'.format(outputfile, eleType)
  outputname_psi2s = '{}_psi2s_{}.pdf'.format(outputfile, eleType)
  outputname_nonresonant_lowq2 = '{}_nonresonant_lowq2_{}.pdf'.format(outputfile, eleType)
  outputname_nonresonant_highq2 = '{}_nonresonant_highq2_{}.pdf'.format(outputfile, eleType)
  pdf_combine(outputplots['jpsi'], outputname_jpsi)
  pdf_combine(outputplots['jpsitri'], outputname_jpsitri)
  pdf_combine(outputplots['psi2s'], outputname_psi2s)
  pdf_combine(outputplots['psi2stri'], outputname_psi2stri)
  pdf_combine(outputplots['nonresonant_lowq2'], outputname_nonresonant_lowq2)
  pdf_combine(outputplots['nonresonant_highq2'], outputname_nonresonant_highq2)
  pdf_sidebyside('{}_combined_{}.pdf'.format(outputfile, eleType), [outputname_jpsitri, outputname_jpsi, outputname_psi2stri, outputname_psi2s, outputname_nonresonant_lowq2, outputname_nonresonant_highq2])
  pdf_sidebyside('{}_rectangularCut_combined_{}.pdf'.format(outputfile, eleType), [outputname_jpsi, outputname_psi2s, outputname_nonresonant_lowq2])
  pdf_sidebyside('{}_jpsi_combined_{}.pdf'.format(outputfile, eleType), [outputname_jpsitri, outputname_jpsi])
  pdf_sidebyside('{}_psi2s_combined_{}.pdf'.format(outputfile, eleType), [outputname_psi2stri, outputname_psi2s])
  pdf_sidebyside('{}_nonresonant_combined_{}.pdf'.format(outputfile, eleType), [outputname_nonresonant_lowq2, outputname_nonresonant_highq2])

  map(lambda x: os.system('rm {}'.format(x)), outputplots['jpsi'])
  map(lambda x: os.system('rm {}'.format(x)), outputplots['jpsitri'])
  map(lambda x: os.system('rm {}'.format(x)), outputplots['psi2s'])
  map(lambda x: os.system('rm {}'.format(x)), outputplots['psi2stri'])
  map(lambda x: os.system('rm {}'.format(x)), outputplots['nonresonant_lowq2'])
  map(lambda x: os.system('rm {}'.format(x)), outputplots['nonresonant_highq2'])

  results = {key: np.array(value) for key, value in results.items()}

  '''
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
  '''

  #argmax_SNR_R = np.argmax(SNR_R)
  #print('Best SNR: {}, Best cut: {}'.format(np.max(SNR_R), mvaCutList[argmax_SNR_R]))
  plotSNR(mvaCutList, results)


