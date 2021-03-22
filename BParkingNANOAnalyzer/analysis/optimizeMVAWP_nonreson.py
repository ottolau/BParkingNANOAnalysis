import uproot
import pandas as pd
import numpy as np
from collections import OrderedDict
from scipy import interp
from rootpy.io import root_open
from rootpy.plotting import Hist
from root_numpy import fill_hist, array2root, array2tree
from root_pandas import to_root
import ROOT
from ROOT import RooFit
import itertools
import PyPDF2
#import makePlot_fitPeak_unbinned as fit_unbinned
from fitter import fitter
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

def plotSNR(cut, results, outputfile):
    fig, ax1 = plt.subplots()
    snr_jpsi, = ax1.plot(cut, results['SNR_jpsi'], 'bo', label=r'$J/\psi', markeredgewidth=0)
    ax1.set_xlabel(r'MVA Cut')
    ax1.set_ylabel(r'$J/\psi: S/\sqrt{S+B}$')
    ax1.yaxis.label.set_color('b')
    for t1 in ax1.get_yticklabels():
        t1.set_color('b')
    ax2 = ax1.twinx()
    snr_nonresonant_lowq2, = ax2.plot(cut, results['SNR_nonresonant_lowq2'], 'ro', label=r'Low $q^{2}$ non-resonant', markeredgewidth=0)
    snr_nonresonant_highq2, = ax2.plot(cut, results['SNR_nonresonant_highq2'], 'gv', label=r'High $q^{2}$ non-resonant', markeredgewidth=0)
    #ax2.fill_between(cut, lower_bound, upper_bound, facecolor='yellow', alpha=0.3)
    ax2.set_ylabel(r'Non-resonant: $S/\sqrt{S+B}$')
    ax2.yaxis.label.set_color('r')
    for t1 in ax2.get_yticklabels():
        t1.set_color('r')
    handles = [snr_jpsi, snr_nonresonant_lowq2, snr_nonresonant_highq2]
    labels = [r'$J/\psi$', r'Low $q^{2}$ non-resonant', r'High $q^{2}$ non-resonant']
    #fig.legend(handles=handles, labels=labels, loc=1, bbox_to_anchor=(1,1), bbox_transform=ax2.transAxes)
    #fig.legend(handles=handles, labels=labels, loc=2, bbox_to_anchor=(0,1), bbox_transform=ax2.transAxes)
    fig.legend(handles=handles, labels=labels, loc=3, bbox_to_anchor=(0,0), bbox_transform=ax2.transAxes)
    fig.suptitle(r'$S/\sqrt{S+B}$')
    fig.savefig('{}_SNRPlot.pdf'.format(outputfile), bbox_inches='tight')
    return '{}_SNRPlot.pdf'.format(outputfile)
    
def plotPunzi(cut, results, outputfile):
    fig, ax1 = plt.subplots()
    snr_jpsi, = ax1.plot(cut, map(Punzi_simplify, results['eff_jpsi'], results['B_jpsi'], results['BErr_jpsi'], itertools.repeat(2.0, results['eff_jpsi'].shape[0])), 'bo', label=r'$J/\psi', markeredgewidth=0)
    ax1.set_xlabel(r'MVA Cut')
    ax1.set_ylabel(r'$J/\psi$: Punzi')
    ax1.yaxis.label.set_color('b')
    for t1 in ax1.get_yticklabels():
        t1.set_color('b')
    ax2 = ax1.twinx()
    snr_nonresonant_lowq2, = ax2.plot(cut, map(Punzi_simplify, results['eff_nonresonant_lowq2'], results['B_nonresonant_lowq2'], results['BErr_nonresonant_lowq2'], itertools.repeat(2.0, results['eff_nonresonant_lowq2'].shape[0])), 'ro', label=r'Low $q^{2}$ non-resonant', markeredgewidth=0)
    snr_nonresonant_highq2, = ax2.plot(cut, map(Punzi_simplify, results['eff_nonresonant_highq2'], results['B_nonresonant_highq2'], results['BErr_nonresonant_highq2'], itertools.repeat(2.0, results['eff_nonresonant_highq2'].shape[0])), 'gv', label=r'High $q^{2}$ non-resonant', markeredgewidth=0)
    #ax2.fill_between(cut, lower_bound, upper_bound, facecolor='yellow', alpha=0.3)
    ax2.set_ylabel(r'Non-resonant: Punzi')
    ax2.yaxis.label.set_color('r')
    for t1 in ax2.get_yticklabels():
        t1.set_color('r')
    handles = [snr_jpsi, snr_nonresonant_lowq2, snr_nonresonant_highq2]
    labels = [r'$J/\psi$', r'Low $q^{2}$ non-resonant', r'High $q^{2}$ non-resonant']
    #fig.legend(handles=handles, labels=labels, loc=1, bbox_to_anchor=(1,1), bbox_transform=ax2.transAxes)
    #fig.legend(handles=handles, labels=labels, loc=2, bbox_to_anchor=(0,1), bbox_transform=ax2.transAxes)
    fig.legend(handles=handles, labels=labels, loc=3, bbox_to_anchor=(0,0), bbox_transform=ax2.transAxes)
    fig.suptitle(r'Punzi Signifiance')
    fig.savefig('{}_FOM_Punzi.pdf'.format(outputfile), bbox_inches='tight')
    return '{}_FOM_Punzi.pdf'.format(outputfile)

def plotSignificance(cut, results, outputfile):
    fig, ax1 = plt.subplots()
    snr_jpsi, = ax1.plot(cut, map(Significance, results['S_jpsi'], results['B_jpsi'], results['BErr_jpsi']), 'bo', label=r'$J/\psi', markeredgewidth=0)
    ax1.set_xlabel(r'MVA Cut')
    ax1.set_ylabel(r'$J/\psi: S/\sqrt{S+B+\Delta B^{2}}$')
    ax1.yaxis.label.set_color('b')
    for t1 in ax1.get_yticklabels():
        t1.set_color('b')
    ax2 = ax1.twinx()
    snr_nonresonant_lowq2, = ax2.plot(cut, map(Significance, results['S_nonresonant_lowq2'], results['B_nonresonant_lowq2'], results['BErr_nonresonant_lowq2']), 'ro', label=r'Low $q^{2}$ non-resonant', markeredgewidth=0)
    snr_nonresonant_highq2, = ax2.plot(cut, map(Significance, results['S_nonresonant_highq2'], results['B_nonresonant_highq2'], results['BErr_nonresonant_highq2']), 'gv', label=r'High $q^{2}$ non-resonant', markeredgewidth=0)
    #ax2.fill_between(cut, lower_bound, upper_bound, facecolor='yellow', alpha=0.3)
    ax2.set_ylabel(r'Non-resonant: $S/\sqrt{S+B+\Delta B^{2}}$')
    ax2.yaxis.label.set_color('r')
    for t1 in ax2.get_yticklabels():
        t1.set_color('r')
    handles = [snr_jpsi, snr_nonresonant_lowq2, snr_nonresonant_highq2]
    labels = [r'$J/\psi$', r'Low $q^{2}$ non-resonant', r'High $q^{2}$ non-resonant']
    #fig.legend(handles=handles, labels=labels, loc=1, bbox_to_anchor=(1,1), bbox_transform=ax2.transAxes)
    #fig.legend(handles=handles, labels=labels, loc=2, bbox_to_anchor=(0,1), bbox_transform=ax2.transAxes)
    fig.legend(handles=handles, labels=labels, loc=3, bbox_to_anchor=(0,0), bbox_transform=ax2.transAxes)
    fig.suptitle(r'$S/\sqrt{S+B+\Delta B^{2}}$')
    fig.savefig('{}_FOM_Significance.pdf'.format(outputfile), bbox_inches='tight')
    return '{}_FOM_Significance.pdf'.format(outputfile)

def plotS(cut, results, outputfile):
    fig, ax1 = plt.subplots()
    lower_bound_jpsi = [s-serror for (s, serror) in zip(results['S_jpsi'], results['SErr_jpsi'])]
    upper_bound_jpsi = [s+serror for (s, serror) in zip(results['S_jpsi'], results['SErr_jpsi'])]
    s_jpsi, = ax1.plot(cut, results['S_jpsi'], 'b-', label=r'$J/\psi$')
    ax1.fill_between(cut, lower_bound_jpsi, upper_bound_jpsi, facecolor='blue', alpha=0.3, linewidth=0)
    ax1.set_xlabel(r'MVA Cut')
    ax1.set_ylabel(r'$J/\psi$: Number of signals')
    ax1.set_ylim(bottom=0)
    ax1.yaxis.label.set_color('b')
    for t1 in ax1.get_yticklabels():
        t1.set_color('b')
    ax2 = ax1.twinx()
    s_nonresonant_lowq2, = ax2.plot(cut, results['S_nonresonant_lowq2'], 'r--', label=r'Low $q^{2}$ non-resonant')
    s_nonresonant_highq2, = ax2.plot(cut, results['S_nonresonant_highq2'], 'g-.', label=r'High $q^{2}$ non-resonant')
    ax2.set_ylabel(r'Non-resonant: Number of signals')
    ax2.set_ylim(bottom=0)
    ax2.yaxis.label.set_color('r')
    for t1 in ax2.get_yticklabels():
        t1.set_color('r')
    handles = [s_jpsi, s_nonresonant_lowq2, s_nonresonant_highq2]
    labels = [r'$J/\psi$', r'Low $q^{2}$ non-resonant', r'High $q^{2}$ non-resonant']
    fig.legend(handles=handles, labels=labels, loc=3, bbox_to_anchor=(0,0), bbox_transform=ax2.transAxes)
    fig.suptitle('Number of signals')
    fig.savefig('{}_SPlot.pdf'.format(outputfile), bbox_inches='tight')
    return '{}_SPlot.pdf'.format(outputfile)

def plotB(cut, results, outputfile):
    fig, ax1 = plt.subplots()
    lower_bound_jpsi = [b-berror for (b, berror) in zip(results['B_jpsi'], results['BErr_jpsi'])]
    upper_bound_jpsi = [b+berror for (b, berror) in zip(results['B_jpsi'], results['BErr_jpsi'])]
    b_jpsi, = ax1.plot(cut, results['B_jpsi'], 'b-', label=r'$J/\psi$')
    ax1.fill_between(cut, lower_bound_jpsi, upper_bound_jpsi, facecolor='blue', alpha=0.3, linewidth=0)
    ax1.set_xlabel(r'MVA Cut')
    ax1.set_ylabel(r'$J/\psi$: Number of background')
    #ax1.set_ylim(bottom=0)
    ax1.set_yscale('log')
    ax1.yaxis.label.set_color('b')
    for t1 in ax1.get_yticklabels():
        t1.set_color('b')
    ax2 = ax1.twinx()
    lower_bound_nonresonant_lowq2 = [b-berror for (b, berror) in zip(results['B_nonresonant_lowq2'], results['BErr_nonresonant_lowq2'])]
    upper_bound_nonresonant_lowq2 = [b+berror for (b, berror) in zip(results['B_nonresonant_lowq2'], results['BErr_nonresonant_lowq2'])]
    lower_bound_nonresonant_highq2 = [b-berror for (b, berror) in zip(results['B_nonresonant_highq2'], results['BErr_nonresonant_highq2'])]
    upper_bound_nonresonant_highq2 = [b+berror for (b, berror) in zip(results['B_nonresonant_highq2'], results['BErr_nonresonant_highq2'])]
    b_nonresonant_lowq2, = ax2.plot(cut, results['B_nonresonant_lowq2'], 'r--', label=r'Low $q^{2}$ non-resonant')
    b_nonresonant_highq2, = ax2.plot(cut, results['B_nonresonant_highq2'], 'g-.', label=r'High $q^{2}$ non-resonant')
    ax2.fill_between(cut, lower_bound_nonresonant_lowq2, upper_bound_nonresonant_lowq2, facecolor='red', alpha=0.3, linewidth=0)
    ax2.fill_between(cut, lower_bound_nonresonant_highq2, upper_bound_nonresonant_highq2, facecolor='green', alpha=0.3, linewidth=0)
    ax2.set_ylabel(r'Non-resonant: Number of background')
    #ax2.set_ylim(bottom=0)
    ax2.set_yscale('log')
    ax2.yaxis.label.set_color('r')
    for t1 in ax2.get_yticklabels():
        t1.set_color('r')
    handles = [b_jpsi, b_nonresonant_lowq2, b_nonresonant_highq2]
    labels = [r'$J/\psi$', r'Low $q^{2}$ non-resonant', r'High $q^{2}$ non-resonant']
    fig.legend(handles=handles, labels=labels, loc=1, bbox_to_anchor=(1,1), bbox_transform=ax2.transAxes)
    fig.suptitle('Number of backgrounds')
    fig.savefig('{}_BPlot.pdf'.format(outputfile), bbox_inches='tight')
    return '{}_BPlot.pdf'.format(outputfile)

def plotJpsi2Psi2SRatio(cut, results, outputfile):
    fig, ax1 = plt.subplots()
    ax1.plot(cut, results['S_jpsi']/results['S_psi2s'])
    ax1.set_xlabel(r'MVA Cut')
    ax1.set_ylabel(r'$N(J/\psi) / N(\psi (2S))$')
    ax1.set_ylim(bottom=0)
    fig.suptitle(r'$N(J/\psi) / N(\psi (2S))$')
    fig.savefig('{}_Jpsi2Psi2SRatio.pdf'.format(outputfile), bbox_inches='tight')
    return '{}_Jpsi2Psi2SRatio.pdf'.format(outputfile)

def plotJpsi2NRRatio(cut, results, outputfile):
    fig, ax1 = plt.subplots()
    ax1.plot(cut, results['S_jpsi']/results['S_nonresonant_lowq2'], 'b', label=r'Low $q^{2}$ non-resonant')
    ax1.plot(cut, results['S_jpsi']/results['S_nonresonant_highq2'], 'r', label=r'High $q^{2}$ non-resonant')
    ax1.set_xlabel(r'MVA Cut')
    ax1.set_ylabel(r'$N(J/\psi) / N({\rm Non-resonant})$')
    ax1.set_ylim(bottom=0)
    ax1.legend(loc='upper left')
    fig.suptitle(r'$N(J/\psi) / N({\rm Non-resonant})$')
    fig.savefig('{}_Jpsi2NRRatio.pdf'.format(outputfile), bbox_inches='tight')
    return '{}_Jpsi2NRRatio.pdf'.format(outputfile)

def plotMvaEfficiency(cut, results, outputfile):
    fig, ax1 = plt.subplots()
    ax1.plot(cut, results['eff_jpsi'], 'b-', label=r'$J/\psi$')
    ax1.plot(cut, results['eff_nonresonant_lowq2'], 'r-', label=r'Low $q^{2}$ non-resonant')
    ax1.plot(cut, results['eff_nonresonant_highq2'], 'g-', label=r'High $q^{2}$ non-resonant')
    ax1.set_xlabel(r'MVA Cut')
    ax1.set_ylabel(r'MVA efficiency')
    ax1.legend(loc='upper right')
    fig.suptitle('Selection efficiency')
    fig.savefig('{}_mvaEfficiency.pdf'.format(outputfile), bbox_inches='tight')
    return '{}_mvaEfficiency.pdf'.format(outputfile)

def plotScaledS(cut, results, outputfile):
    fig, ax1 = plt.subplots()
    scl_jpsi, = ax1.plot(cut, results['S_jpsi']/results['eff_jpsi'], 'b-', label=r'$J/\psi$')
    ax1.set_xlabel(r'MVA Cut')
    ax1.set_ylabel(r'$J/\psi: S/\epsilon_{\rm MVA}$')
    ax1.yaxis.label.set_color('b')
    ax1.set_ylim(bottom=0)
    for t1 in ax1.get_yticklabels():
        t1.set_color('b')
    ax2 = ax1.twinx()
    scl_nonresonant_lowq2, = ax2.plot(cut, results['S_nonresonant_lowq2']/results['eff_nonresonant_lowq2'], 'r--', label=r'Low $q^{2}$ mon-resonant')
    scl_nonresonant_highq2, = ax2.plot(cut, results['S_nonresonant_highq2']/results['eff_nonresonant_highq2'], 'g-.', label=r'High $q^{2}$ non-resonant')
    ax2.set_ylabel(r'Non-resonant: $S/\epsilon_{\rm MVA}$')
    ax2.yaxis.label.set_color('r')
    ax2.set_ylim(bottom=0)
    for t1 in ax2.get_yticklabels():
        t1.set_color('r')
    handles = [scl_jpsi, scl_nonresonant_lowq2, scl_nonresonant_highq2]
    labels = [r'$J/\psi$', r'Low $q^{2}$ mon-resonant', r'High $q^{2}$ non-resonant']
    #fig.legend(handles=handles, labels=labels, loc=1, bbox_to_anchor=(1,1), bbox_transform=ax2.transAxes)
    fig.legend(handles=handles, labels=labels, loc=3, bbox_to_anchor=(0,0), bbox_transform=ax2.transAxes)
    fig.suptitle(r'$S/\epsilon_{\rm MVA}$')
    fig.savefig('{}_scaledS.pdf'.format(outputfile), bbox_inches='tight')
    return '{}_scaledS.pdf'.format(outputfile)

def plotSummary(cut, results, outputfile):
    fig = []
    fig.append(plotSNR(cut, results, outputfile))
    fig.append(plotPunzi(cut, results, outputfile))
    fig.append(plotSignificance(cut, results, outputfile))
    fig.append(plotS(cut, results, outputfile))
    fig.append(plotB(cut, results, outputfile))
    fig.append(plotJpsi2Psi2SRatio(cut, results, outputfile))
    fig.append(plotJpsi2NRRatio(cut, results, outputfile))
    fig.append(plotMvaEfficiency(cut, results, outputfile))
    fig.append(plotScaledS(cut, results, outputfile))
    pdf_combine(fig, outputfile+'_summary.pdf')
    map(lambda x: os.system('rm {}'.format(x)), fig)

def get_diagonalCut_var(branches, mll_mean, fit_mass_mean, diagCut_lower_bound, diagCut_jpsi_upper_bound, eigVecs):
  branches['BToKEE_mll_fullfit_centered'] = branches['BToKEE_mll_fullfit'] - mll_mean
  branches['BToKEE_fit_mass_centered'] = branches['BToKEE_fit_mass'] - fit_mass_mean
  data_centered = np.array([branches['BToKEE_fit_mass_centered'],branches['BToKEE_mll_fullfit_centered']]).T
  eigVecs_jpsi = triCut_jpsi_rotMatrix_pf 
  data_decorr = data_centered.dot(eigVecs)
  return data_decorr[:,0], data_decorr[:,1]

def fit(name, selected_branches, fit_params, results, outputfile):
  branches = np.array(selected_branches, dtype=[('BToKEE_fit_mass', 'f4')])
  tree = array2tree(branches)
  outputname = outputfile + '_{0}_mva_{1:.3f}'.format(name, fit_params['mvaCut']).replace('.','-') + '.pdf'
  #Stot, StotErr, S, SErr, B, BErr= fit_unbinned.fit(tree, outputname, **fit_params)
  #output = fit_unbinned.fit(tree, outputname, **fit_params)
  #output = fit_unbinned.fit(tree, outputname, **fit_params)
  b_fitter = fitter()
  b_fitter.init_fit_data(**fit_params)
  output = b_fitter.fit(tree, outputname)

  results['Stot_{}'.format(name)].append(output['Stot'])
  results['StotErr_{}'.format(name)].append(output['StotErr'])
  results['S_{}'.format(name)].append(output['S'])
  results['SErr_{}'.format(name)].append(output['SErr'])
  results['B_{}'.format(name)].append(output['B'])
  results['BErr_{}'.format(name)].append(output['BErr'])
  results['SNR_{}'.format(name)].append(output['S']/np.sqrt(output['S'] + output['B']))
  results['exp_alpha_{}'.format(name)].append(output['exp_alpha'])
  return results, outputname

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
  partial_nonresonant_lowq2 = 'part_workspace_nonresonant_lowq2_{}.root'.format(eleType)
  #partial_nonresonant_lowq2 = 'part_workspace_nonresonant_lowq2_diagCut_{}.root'.format(eleType)
  partial_nonresonant_highq2 = 'part_workspace_nonresonant_highq2_{}.root'.format(eleType)

  #psi2s_jpsi = 'psi2s_workspace_jpsi_{}.root'.format(eleType)
  jpsi_psi2s = 'jpsi_workspace_psi2s_{}.root'.format(eleType)
  #jpsi_nonresonant = 'jpsi_workspace_nonresonant_{}.root'.format(eleType)
  jpsi_nonresonant_lowq2 = 'jpsi_workspace_lowq2_{}.root'.format(eleType)

  params_jpsi = eval('params_jpsi_{}'.format(eleType))
  #params_jpsitri = eval('params_jpsitri_{}'.format(eleType))
  params_jpsitri = params_jpsitri_pf
  params_psi2s = eval('params_psi2s_{}'.format(eleType))
  #params_psi2stri = eval('params_psi2stri_{}'.format(eleType))
  params_psi2stri = params_psi2stri_pf
  params_nonresonant_lowq2 = eval('params_jpsi_{}'.format(eleType))
  params_nonresonant_highq2 = eval('params_jpsi_{}'.format(eleType))

  #jpsi_mc = 'RootTree_2020Jan16_BuToKJpsi_Toee_BToKEEAnalyzer_2020Apr17_mc_mva_{}.root'.format(eleType)
  #nonresonant_mc = 'RootTree_2020Jan16_BuToKee_all_BToKEEAnalyzer_2020Apr17_mc_mva_{}.root'.format(eleType)
  jpsi_mc = 'BParkingNANO_2020Jan16_BuToKJpsi_Toee_BToKEEAnalyzer_2020May16_newVar_mc.root'
  nonresonant_mc = 'BParkingNANO_2020Jan16_BuToKee_all_BToKEEAnalyzer_2020May03_newVar_mc.root'

  data_branches = ['BToKEE_mll_fullfit', 'BToKEE_fit_mass', 'BToKEE_mva', 'BToKEE_event', 'BToKEE_Dmass', 'BToKEE_Dmass_flip']
  #data_branches = ['BToKEE_mll_fullfit', 'BToKEE_fit_mass', 'BToKEE_mva']

  features = ['BToKEE_fit_l1_normpt', 'BToKEE_l1_dxy_sig',
              'BToKEE_fit_l2_normpt', 'BToKEE_l2_dxy_sig',
              'BToKEE_fit_k_normpt', 'BToKEE_k_DCASig',
              'BToKEE_fit_normpt', 'BToKEE_svprob', 'BToKEE_fit_cos2D', 'BToKEE_l_xy_sig', 'BToKEE_dz'
              ]
  #features += ['BToKEE_fit_l1_eta', 'BToKEE_fit_l2_eta', 'BToKEE_fit_k_eta', 'BToKEE_fit_eta']
  features += ['BToKEE_eleDR', 'BToKEE_llkDR']
  features += ['BToKEE_l1_iso04_rel', 'BToKEE_l2_iso04_rel', 'BToKEE_k_iso04_rel', 'BToKEE_b_iso04_rel']
  features += ['BToKEE_ptAsym']
  #features += ['BToKEE_ptImbalance']
  #features += ['BToKEE_l1_pfmvaId_lowPt', 'BToKEE_l2_pfmvaId_lowPt', 'BToKEE_l1_pfmvaId_highPt', 'BToKEE_l2_pfmvaId_highPt']
  features += ['BToKEE_l1_mvaId', 'BToKEE_l2_mvaId']
  features += ['BToKEE_Dmass', 'BToKEE_Dmass_flip']
  #features += ['BToKEE_svprob_rank', 'BToKEE_fit_pt_rank', 'BToKEE_fit_cos2D_rank', 'BToKEE_l_xy_rank']

  training_branches = sorted(features)
  #ntree_limit = 798
  ntree_limit = 878
  #ntree_limit = 769
  #modelfile = '../models/xgb_fulldata_05Jul2020_allq2_isoPFMVANewDRptAsymDmass_weighted_pauc2_mix.model'
  modelfile = '../models/xgb_fulldata_11May2020_allq2_isoPFMVANewDRptAsymDmass_weighted_pauc02_pf.model'
  #modelfile = '../models/xgb_fulldata_30Jul2020_allq2_isoPFMVANewDRptAsymDmass_weighted_pauc2_low_exc.model'
  model = xgb.Booster({'nthread': 6})
  model.load_model(modelfile)

  mc_eleType_selection = {}
  mc_eleType_selection['pf'] = '(BToKEE_l1_isPF == True) and (BToKEE_l2_isPF == True)'
  mc_eleType_selection['mix'] = '(not (BToKEE_l1_isPFoverlap)) and (not (BToKEE_l2_isPFoverlap)) and (not (((BToKEE_l1_isPF == True) and (BToKEE_l2_isPF == True)) or ((BToKEE_l1_isPF == False) and (BToKEE_l2_isPF == False))))'
  mc_eleType_selection['low'] = '(BToKEE_l1_isPF == False) and (BToKEE_l2_isPF == False)'
  mc_eleType_selection['low_exc'] = '(BToKEE_l1_isPF == False) and (BToKEE_l2_isPF == False) and (not (BToKEE_l1_isPFoverlap)) and (not (BToKEE_l2_isPFoverlap))'

  #preselection = '(BToKEE_Dmass > 2.0) and (BToKEE_Dmass_flip > 2.0)'
  #preselection = '(BToKEE_l1_pfmvaId_lowPt > -0.555556) and (BToKEE_l2_pfmvaId_lowPt > -1.666667) and (BToKEE_l1_pfmvaId_highPt > -2.777778) and (BToKEE_l2_pfmvaId_highPt > -4.444444)'
  preselection = ''
  #preselection = '(BToKEE_mll_fullfit > 0.6)'

  mc_branches = list(set(['BToKEE_fit_mass', 'BToKEE_mll_fullfit', 'BToKEE_l1_isPF', 'BToKEE_l2_isPF', 'BToKEE_l1_isPFoverlap', 'BToKEE_l2_isPFoverlap', 'BToKEE_l1_pfmvaId_lowPt', 'BToKEE_l2_pfmvaId_lowPt', 'BToKEE_l1_pfmvaId_highPt', 'BToKEE_l2_pfmvaId_highPt', 'BToKEE_Dmass', 'BToKEE_Dmass_flip'] + training_branches))

  drawSNR = True

  #branches = get_df(inputfile, branches=data_branches)
  branches = get_df(inputfile, branches=data_branches+['BToKEE_ll_charge',])
  branches.query('BToKEE_ll_charge == 0', inplace=True)

  mll_mean_jpsi = triCut_jpsi_mll_mean_pf
  fit_mass_mean_jpsi = triCut_jpsi_mKee_mean_pf
  triCut_jpsi_lower_bound = triCut_jpsi_lower_bound_pf
  triCut_jpsi_upper_bound = triCut_jpsi_upper_bound_pf
  eigVecs_jpsi = triCut_jpsi_rotMatrix_pf 
  branches['BToKEE_fit_mass_decorr_jpsi'], branches['BToKEE_mll_fullfit_decorr_jpsi'] = get_diagonalCut_var(branches, mll_mean_jpsi, fit_mass_mean_jpsi, triCut_jpsi_lower_bound, triCut_jpsi_upper_bound, eigVecs_jpsi)

  mll_mean_psi2s = triCut_psi2s_mll_mean_pf
  fit_mass_mean_psi2s = triCut_psi2s_mKee_mean_pf
  triCut_psi2s_lower_bound = triCut_psi2s_lower_bound_pf
  triCut_psi2s_upper_bound = triCut_psi2s_upper_bound_pf
  eigVecs_psi2s = triCut_psi2s_rotMatrix_pf 
  branches['BToKEE_fit_mass_decorr_psi2s'], branches['BToKEE_mll_fullfit_decorr_psi2s'] = get_diagonalCut_var(branches, mll_mean_psi2s, fit_mass_mean_psi2s, triCut_psi2s_lower_bound, triCut_psi2s_upper_bound, eigVecs_psi2s)


  jpsi_mc_branches = get_df(jpsi_mc, branches=mc_branches).sort_index(axis=1)
  jpsi_mc_branches.query(mc_eleType_selection[eleType], inplace=True)
  #jpsi_mc_branches.query(preselection, inplace=True)
  jpsi_mc_branches['BToKEE_mva'] = model.predict(xgb.DMatrix(jpsi_mc_branches[training_branches].replace([np.inf, -np.inf], 0.0).sort_index(axis=1)), ntree_limit=ntree_limit)
  jpsi_mc_branches.query('(BToKEE_mll_fullfit > @JPSI_LOW) and (BToKEE_mll_fullfit < @JPSI_UP)', inplace=True)
  nTot_jpsi = float(jpsi_mc_branches.shape[0])

  nonresonant_mc_branches = get_df(nonresonant_mc, branches=mc_branches).sort_index(axis=1)
  nonresonant_mc_branches.query(mc_eleType_selection[eleType], inplace=True)
  nonresonant_mc_branches['BToKEE_fit_mass_decorr_jpsi'], nonresonant_mc_branches['BToKEE_mll_fullfit_decorr_jpsi'] = get_diagonalCut_var(nonresonant_mc_branches, mll_mean_jpsi, fit_mass_mean_jpsi, triCut_jpsi_lower_bound, triCut_jpsi_upper_bound, eigVecs_jpsi)
  #nonresonant_mc_branches.query(preselection, inplace=True)
  nonresonant_mc_branches['BToKEE_mva'] = model.predict(xgb.DMatrix(nonresonant_mc_branches[training_branches].replace([np.inf, -np.inf], 0.0).sort_index(axis=1)), ntree_limit=ntree_limit)
  nonresonant_lowq2_mc_branches = nonresonant_mc_branches.query('(BToKEE_mll_fullfit > @NR_LOW) and (BToKEE_mll_fullfit < @JPSI_LOW)')
  #nonresonant_lowq2_mc_branches = nonresonant_mc_branches.query('(BToKEE_mll_fullfit > @NR_LOW) and (BToKEE_mll_fullfit < @JPSI_LOW) and (BToKEE_mll_fullfit_decorr_jpsi < @triCut_jpsi_lower_bound)')
  nonresonant_highq2_mc_branches = nonresonant_mc_branches.query('(BToKEE_mll_fullfit > @PSI2S_UP) and (BToKEE_mll_fullfit < @NR_UP)')
  nTot_nonresonant_lowq2 = float(nonresonant_lowq2_mc_branches.shape[0])
  nTot_nonresonant_highq2 = float(nonresonant_highq2_mc_branches.shape[0])
  nTot_nonresonant = float(nonresonant_mc_branches.shape[0])

  print(nTot_jpsi, nTot_nonresonant)

  print(nTot_nonresonant_lowq2, nTot_nonresonant_highq2, nTot_nonresonant)
  output_branches = {}

  results = {'{}_{}'.format(quantity, region): [] for quantity, region in itertools.product(['Stot', 'StotErr', 'S', 'SErr', 'B', 'BErr', 'SNR', 'eff', 'exp_alpha'], ['nonresonant_lowq2', 'nonresonant_highq2', 'jpsi', 'jpsitri', 'psi2s', 'psi2stri'])}
  outputplots = {'jpsi': [],
                 'jpsitri': [],
                 'psi2s': [],
                 'psi2stri': [],
                 'nonresonant_lowq2': [],
                 'nonresonant_highq2': [],
                 }

  #mvaCutList = np.linspace(8.0, 14.0, 40)
  #mvaCutList = np.linspace(6.0, 12.0, 40)
  #mvaCutList = np.linspace(10.0, 16.0, 40)
  #mvaCutList = np.linspace(12.0, 16.0, 30)
  mvaCutList = np.linspace(7.0, 12.0, 30)
  #mvaCutList = np.array([10.0,])
  for mvaCut in mvaCutList:
    # mva selection
    #mva_selection = (branches['BToKEE_mva'] > mvaCut) #& (branches['BToKEE_Dmass'] > 2.0) & (branches['BToKEE_Dmass_flip'] > 2.0)
    #selected_branches = branches[mva_selection].sort_values('BToKEE_mva', ascending=False).drop_duplicates(['BToKEE_event'], keep='first')
    mva_selection = '(BToKEE_mva > @mvaCut)' #+ ' and ' + preselection
    mc_mva_selection = mva_selection
    if preselection.strip():
      mc_mva_selection += ' and ' + preselection
    selected_branches = branches.query(mva_selection).sort_values('BToKEE_mva', ascending=False).drop_duplicates(['BToKEE_event'], keep='first')
    #selected_branches = branches.query(mva_selection)

    # j/psi selection with triangular cut on q2
    fit_params_jpsitri = {'drawSNR': drawSNR,
                          'mvaCut': mvaCut,
                          'blinded': False,
                          'params': params_jpsitri,
                          }


    jpsitri_selection = '(BToKEE_mll_fullfit_decorr_jpsi < @triCut_jpsi_upper_bound) and (BToKEE_mll_fullfit > @JPSI_LOW) and (BToKEE_mll_fullfit < @JPSI_UP)'
    results, outputname_jpsitri = fit('jpsitri', selected_branches.query(jpsitri_selection)['BToKEE_fit_mass'], fit_params_jpsitri, results, outputfile)
    outputplots['jpsitri'].append(outputname_jpsitri)

    # j/psi selection
    fit_params_jpsi = {'drawSNR': drawSNR,
                       'mvaCut': mvaCut,
                       'blinded': False,
                       'params': params_jpsi,
                       'partialfit': OrderedDict([('partial', {'filename': partial_jpsi, 'label': 'Partially Reco.', 'color': 40})]), 
                       }

    jpsi_selection = '(BToKEE_mll_fullfit > @JPSI_LOW) and (BToKEE_mll_fullfit < @JPSI_UP)'
    results, outputname_jpsi = fit('jpsi', selected_branches.query(jpsi_selection)['BToKEE_fit_mass'], fit_params_jpsi, results, outputfile)
    outputplots['jpsi'].append(outputname_jpsi)

    eff_jpsi = float(jpsi_mc_branches.query(mc_mva_selection).shape[0]) / nTot_jpsi
    eff_nonresonant_lowq2 = float(nonresonant_lowq2_mc_branches.query(mc_mva_selection).shape[0]) / nTot_nonresonant_lowq2
    eff_nonresonant_highq2 = float(nonresonant_highq2_mc_branches.query(mc_mva_selection).shape[0]) / nTot_nonresonant_highq2

    results['eff_jpsi'].append(eff_jpsi)
    results['eff_nonresonant_lowq2'].append(eff_nonresonant_lowq2)
    results['eff_nonresonant_highq2'].append(eff_nonresonant_highq2)

    expS_lowq2 = results['Stot_jpsi'][-1] * BR_BToKLL / (BR_BToKJpsi * BR_JpsiToLL) * (nTot_nonresonant_lowq2 / nTot_nonresonant) * (eff_nonresonant_lowq2 / eff_jpsi)#(12091.0/44024)
    expS_highq2 = results['Stot_jpsi'][-1] * BR_BToKLL / (BR_BToKJpsi * BR_JpsiToLL) * (nTot_nonresonant_highq2 / nTot_nonresonant) * (eff_nonresonant_highq2 / eff_jpsi)#(12091.0/44024)

    expLeakage = results['Stot_jpsi'][-1] * 0.0089478360198 

    # psi(2s) selection with triangular cut on q2
    fit_params_psi2stri = {'drawSNR': drawSNR,
                           'mvaCut': mvaCut,
                           'blinded': False,
                           'params': params_psi2stri,
                           'sigName': "B^{+}#rightarrow K^{+} #psi (2S)(#rightarrow e^{+}e^{-})",
                           'partialfit': OrderedDict([('jpsi', {'filename': jpsi_psi2s, 'label': 'B^{+}#rightarrow K^{+} J/#psi(#rightarrow e^{+}e^{-})', 'color': 46})]), 
                           }
   
    psi2stri_selection = '(BToKEE_mll_fullfit_decorr_psi2s < @triCut_psi2s_upper_bound) and (BToKEE_mll_fullfit > @JPSI_UP) and (BToKEE_mll_fullfit < @PSI2S_UP)'
    results, outputname_psi2stri = fit('psi2stri', selected_branches.query(psi2stri_selection)['BToKEE_fit_mass'], fit_params_psi2stri, results, outputfile)
    outputplots['psi2stri'].append(outputname_psi2stri)

    # psi(2s) selection
    fit_params_psi2s = {'drawSNR': drawSNR,
                        'mvaCut': mvaCut,
                        'blinded': False,
                        'params': params_psi2s,
                        'sigName': "B^{+}#rightarrow K^{+} #psi (2S)(#rightarrow e^{+}e^{-})",
                        'partialfit': OrderedDict([('partial', {'filename': partial_psi2s, 'label': 'Partially Reco.', 'color': 40}), ('jpsi', {'filename': jpsi_psi2s, 'label': 'B^{+}#rightarrow K^{+} J/#psi(#rightarrow e^{+}e^{-})', 'color': 46})]), 
                        }
   
    psi2s_selection = '(BToKEE_mll_fullfit > @JPSI_UP) and (BToKEE_mll_fullfit < @PSI2S_UP)'
    results, outputname_psi2s = fit('psi2s', selected_branches.query(psi2s_selection)['BToKEE_fit_mass'], fit_params_psi2s, results, outputfile)
    outputplots['psi2s'].append(outputname_psi2s)
    
    # low q2 non-resonant selection
    fit_params_nonresonant_lowq2 = {'drawSNR': drawSNR,
                                    'mvaCut': mvaCut,
                                    'blinded': True,
                                    'expS': expS_lowq2,
                                    'params': params_nonresonant_lowq2,
                                    'sigName': "B^{+}#rightarrow K^{+} e^{+}e^{-}",
                                    'partialfit': OrderedDict([('partial', {'filename': partial_nonresonant_lowq2, 'label': 'Partially Reco.', 'color': 40}), ('jpsi', {'filename': jpsi_nonresonant_lowq2, 'label': 'B^{+}#rightarrow K^{+} J/#psi(#rightarrow e^{+}e^{-})', 'color': 46, 'expected_yield': expLeakage})]), 
                                    #'partialfit': OrderedDict([('partial', {'filename': partial_nonresonant_lowq2, 'label': 'Partially Reco.', 'color': 40})]), 
                                    }

    nonresonant_lowq2_selection = '(BToKEE_mll_fullfit > @NR_LOW) and (BToKEE_mll_fullfit < @JPSI_LOW)'
    #nonresonant_lowq2_selection = '(BToKEE_mll_fullfit > @NR_LOW) and (BToKEE_mll_fullfit < @JPSI_LOW) and (BToKEE_mll_fullfit_decorr_jpsi < @triCut_jpsi_lower_bound)'
    results, outputname_nonresonant_lowq2 = fit('nonresonant_lowq2', selected_branches.query(nonresonant_lowq2_selection)['BToKEE_fit_mass'], fit_params_nonresonant_lowq2, results, outputfile)
    outputplots['nonresonant_lowq2'].append(outputname_nonresonant_lowq2)

    # high q2 non-resonant selection
    fit_params_nonresonant_highq2 = {'drawSNR': drawSNR,
                                    'mvaCut': mvaCut,
                                    'blinded': True,
                                    'expS': expS_highq2,
                                    'params': params_nonresonant_highq2,
                                    'sigName': "B^{+}#rightarrow K^{+} e^{+}e^{-}",
                                    'partialfit': OrderedDict([('partial', {'filename': partial_nonresonant_highq2, 'label': 'Partially Reco.', 'color': 40})]), 
                                    }

    nonresonant_highq2_selection = '(BToKEE_mll_fullfit > @PSI2S_UP) and (BToKEE_mll_fullfit < @NR_UP)'
    results, outputname_nonresonant_highq2 = fit('nonresonant_highq2', selected_branches.query(nonresonant_highq2_selection)['BToKEE_fit_mass'], fit_params_nonresonant_highq2, results, outputfile)
    outputplots['nonresonant_highq2'].append(outputname_nonresonant_highq2)

    print("="*80)
    #print('MVA: {}\n\tJ/psi - S: {}, B: {}, S/sqrt(S+B): {}\n\tNon-resonant - S: {}, B: {}, S/sqrt(S+B): {}'.format(mvaCut, S_jpsi, B_jpsi, S_jpsi/np.sqrt(S_jpsi+B_jpsi), S_NR_lowq2, B_NR_lowq2, S_NR_lowq2/np.sqrt(S_NR_lowq2+B_NR_lowq2)))
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
  pdf_sidebyside('{}_rectangularCut_combined_{}.pdf'.format(outputfile, eleType), [outputname_nonresonant_lowq2, outputname_jpsi, outputname_psi2s, outputname_nonresonant_highq2])
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
  results['cut'] = mvaCutList
  results_df = {key: pd.Series(value) for key, value in results.items()}
  print(results_df)

  pd.DataFrame.from_dict(results_df).to_csv('{}_results.csv'.format(outputfile))

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
  plotSummary(mvaCutList, results, outputfile)


