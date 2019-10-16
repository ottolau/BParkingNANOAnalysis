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

import pandas as pd
import numpy as np
from rootpy.io import root_open
from rootpy.plotting import Hist
from root_numpy import fill_hist, array2root, array2tree
from root_pandas import to_root
from keras.models import load_model
from scipy.optimize import curve_fit
from scipy.integrate import quad
import iminuit, probfit
import scipy.stats

import ROOT
from ROOT import RooFit
ROOT.gErrorIgnoreLevel=ROOT.kError
ROOT.RooMsgService.instance().setGlobalKillBelow(RooFit.FATAL)


import argparse
parser = argparse.ArgumentParser(description="A simple ttree plotter")
parser.add_argument("-s", "--signal", dest="signal", default="RootTree_BParkingNANO_2019Sep12_BuToKJpsi_Toee_mvaTraining_sig_pf.root", help="Signal file")
parser.add_argument("-b", "--background", dest="background", default="RootTree_BParkingNANO_2019Sep12_Run2018A2A3B2B3C2C3D2_mvaTraining_bkg_pf.root", help="Background file")
parser.add_argument("-o", "--outputfile", dest="outputfile", default="test", help="Output file containing plots")
parser.add_argument("-m", "--model", dest="model", default="dense_model_pf.h5", help="Trainned model")
args = parser.parse_args()


ELECTRON_MASS = 0.000511
K_MASS = 0.493677
JPSI_LOW = 2.9
JPSI_UP = 3.3
B_LOWSB_LOW = 4.75
B_LOWSB_UP = 5.0
B_UPSB_LOW = 5.5
B_UPSB_UP = 5.75
B_MIN = 4.7
B_MAX = 6.0

def unbinned_exp_LLH(data, loc_init, scale_init, limit_loc, limit_scale):
    # Define function to fit
    def exp_func(x, loc):
        return scipy.stats.expon.pdf(x, loc)

    # Define initial parameters
    init_params = dict(loc = loc_init, scale = scale_init)

    print(iminuit.describe(exp_func))
    # Create an unbinned likelihood object with function and data.
    unbin = probfit.UnbinnedLH(exp_func, data)

    # Minimizes the unbinned likelihood for the given function
    m = iminuit.Minuit(unbin, loc=1.0, scale=1.0, print_level=0)#,
                       #*init_params,
                       #limit_scale = limit_loc,
                       #limit_loc = limit_scale,
                       #print_level=0)
    m.migrad()
    unbin.show(m)
    params = m.values.values() # Get out fit values
    errs   = m.errors.values()
    return params, errs



def plotSNR(cut, sig, bkg, CutBasedWP):
    fig, ax1 = plt.subplots()
    #plt.grid(linestyle='--')
    snr_line, = ax1.plot(cut, sig/np.sqrt(sig+bkg), 'b-', label=r'$S/\sqrt{S+B}$')
    snr_cb_line = ax1.axhline(y=CutBasedWP['SNR'], color='g', linestyle='-.')
    ax1.set_xlabel(r'MVA Cut')
    ax1.set_ylabel(r'$S/\sqrt{S+B}$')
    #ax1.set_ylim(ymin=0)
    for t1 in ax1.get_yticklabels():
        t1.set_color('b')

    ax2 = ax1.twinx()
    s_line, = ax2.plot(cut, sig, 'r-', label=r'Number of signals')
    s_cb_line = ax2.axhline(y=CutBasedWP['S'], color='c', linestyle='-.')
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

    
    fig3, ax3 = plt.subplots()
    plt.grid(linestyle='--')
    ax3.plot(sig, sig/np.sqrt(sig+bkg), 'bo', label='Keras')
    ax3.plot(CutBasedWP['S'], CutBasedWP['SNR'], 'r*', label='Cut-based')
    ax3.set_xlabel(r'S')
    ax3.set_ylabel(r'$S/\sqrt{S+B}$')
    ax3.legend(loc=2)
    fig3.savefig('{}_S_SNRPlot.pdf'.format(args.outputfile), bbox_inches='tight')

def expo(x, a, b):
  return a * np.exp(-b * x)

if __name__ == "__main__":
  # constants
  NMC_TOT = 2122456.0
  CROSS_SECTION = 543100000.0 * 1.0e+3#fb
  FILTER_EFF = 0.0035
  BR_BTOKJPSI = 1.01e-3
  BR_JPSITOEE = 0.06
  FU = 0.4
  BR_BTOKJPSIEE = BR_BTOKJPSI * BR_JPSITOEE
  #BR_BTOKJPSIEE = 1.0
  LUMI_MC = NMC_TOT / (CROSS_SECTION * FILTER_EFF * FU * BR_BTOKJPSIEE) # fb-1
  LUMI_DATA = 9.543240524999998 # fb-1

  print("MC lumiosity: {}".format(LUMI_MC))

  inputfile_sig = args.signal.replace('.h5','')+'.h5'
  inputfile_bkg = args.background.replace('.h5','')+'.h5'

  outputfile = args.outputfile.replace('.root','').replace('.h5','')

  df = {}
  df['sig'] = pd.read_hdf(inputfile_sig, 'branches')
  df['bkg'] = pd.read_hdf(inputfile_bkg, 'branches')

  output_branches = {}
  training_branches = sorted(['BToKEE_l1_normpt', 'BToKEE_l1_eta', 'BToKEE_l1_phi', 'BToKEE_l1_dxy_sig', 'BToKEE_l1_dz', 'BToKEE_l2_normpt', 'BToKEE_l2_eta', 'BToKEE_l2_phi', 'BToKEE_l2_dxy_sig', 'BToKEE_l2_dz', 'BToKEE_k_normpt', 'BToKEE_k_eta', 'BToKEE_k_phi', 'BToKEE_k_DCASig', 'BToKEE_normpt', 'BToKEE_svprob', 'BToKEE_cos2D', 'BToKEE_l_xy_sig'])
  #training_branches = sorted(['BToKEE_l1_normpt', 'BToKEE_l1_eta', 'BToKEE_l1_phi', 'BToKEE_l1_dxy_sig', 'BToKEE_l1_dz', 'BToKEE_l1_mvaId', 'BToKEE_l2_normpt', 'BToKEE_l2_eta', 'BToKEE_l2_phi', 'BToKEE_l2_dxy_sig', 'BToKEE_l2_dz', 'BToKEE_l2_mvaId', 'BToKEE_k_normpt', 'BToKEE_k_eta', 'BToKEE_k_phi', 'BToKEE_k_DCASig', 'BToKEE_normpt', 'BToKEE_svprob', 'BToKEE_cos2D', 'BToKEE_l_xy_sig'])


  for k, branches in df.items():
    branches['BToKEE_normpt'] = branches['BToKEE_pt'] / branches['BToKEE_mass']

    jpsi_selection = (branches['BToKEE_mll_raw'] > JPSI_LOW) & (branches['BToKEE_mll_raw'] < JPSI_UP)
    b_selection = jpsi_selection & (branches['BToKEE_mass'] > B_LOWSB_UP) & (branches['BToKEE_mass'] < B_UPSB_LOW)
    b_lowsb_selection = jpsi_selection & (branches['BToKEE_mass'] > B_LOWSB_LOW) & (branches['BToKEE_mass'] < B_LOWSB_UP)
    b_upsb_selection = jpsi_selection & (branches['BToKEE_mass'] > B_UPSB_LOW) & (branches['BToKEE_mass'] < B_UPSB_UP)
    b_sb_selection = b_lowsb_selection | b_upsb_selection

    general_selection = jpsi_selection & (branches['BToKEE_l1_mvaId'] > 3.94) & (branches['BToKEE_l2_mvaId'] > 3.94) & (branches['BToKEE_k_pt'] > 1.5) 

    # additional cuts, allows various lengths
    l1_pf_selection = (branches['BToKEE_l1_isPF'])
    l2_pf_selection = (branches['BToKEE_l2_isPF'])
    l1_low_selection = (branches['BToKEE_l1_isLowPt']) #& (branches['BToKEE_l1_pt'] < 5.0)
    l2_low_selection = (branches['BToKEE_l2_isLowPt']) #& (branches['BToKEE_l2_pt'] < 5.0)

    pf_selection = l1_pf_selection & l2_pf_selection
    low_selection = l1_low_selection & l2_low_selection
    overlap_veto_selection = np.logical_not(branches['BToKEE_l1_isPFoverlap']) & np.logical_not(branches['BToKEE_l2_isPFoverlap'])
    mix_selection = ((l1_pf_selection & l2_low_selection) | (l2_pf_selection & l1_low_selection))
    low_pfveto_selection = low_selection & overlap_veto_selection
    mix_net_selection = overlap_veto_selection & np.logical_not(pf_selection | low_selection)


    df[k] = branches[general_selection & pf_selection].copy()
    df[k].replace([np.inf, -np.inf], 10.0**+10, inplace=True)


    # count the number of b candidates passes the selection
    #count_selection = jpsi_selection 
    #nBToKEE_selected = self._branches['BToKEE_event'][count_selection].values
    #_, nBToKEE_selected = np.unique(nBToKEE_selected[np.isfinite(nBToKEE_selected)], return_counts=True)


  # add mva id to pandas dataframe
  #print(df['sig'], df['bkg'])

  model = load_model(args.model)
  mva_sig = model.predict(df['sig'][training_branches].sort_index(axis=1).values)
  mva_bkg = model.predict(df['bkg'][training_branches].sort_index(axis=1).values)

  df['sig']['BToKEE_keras'] = mva_sig
  df['bkg']['BToKEE_keras'] = mva_bkg
  #print(df['sig']['BToKEE_keras'])

  SList = []
  SErrList = []
  BList = []
  
  mvaCutList = np.linspace(0.80, 0.99, 5)
  for mvaCut in mvaCutList:
    mvaCutReplace = '{0:.3f}'.format(mvaCut).replace('.','_')
    # mva selection
    selected_branches_sig = df['sig'][(df['sig']['BToKEE_keras'] > mvaCut)]['BToKEE_mass']
    selected_branches_bkg = df['bkg'][(df['bkg']['BToKEE_keras'] > mvaCut)]['BToKEE_mass'].values
    NMC_SELECTED = selected_branches_sig.count()
    h_BToKEE_mass_bkg = Hist(50, 4.7, 6.0, name='h_BToKEE_mass_bkg', title='', type='F') 
    fill_hist(h_BToKEE_mass_bkg, selected_branches_bkg[np.isfinite(selected_branches_bkg)])
    #selected_branches_bkg = selected_branches_bkg[np.isfinite(selected_branches_bkg)]
    #popt, pocv = unbinned_exp_LLH(selected_branches_bkg, loc_init = 0, scale_init = 0.5, limit_loc = (-1, 1), limit_scale = (-1, 1))
    #print(popt)

    h_bins, h_steps = np.linspace(4.7, 6.0, 30, retstep=True)
    h_bkg_y, h_bkg_x = np.histogram(selected_branches_bkg, bins=h_bins)
    h_bkg_x = (h_bkg_x[:-1] + h_bkg_x[1:]) / 2
    remove_zero = np.where(np.greater(h_bkg_y, 1.0))
    h_bkg_x, h_bkg_y = h_bkg_x[remove_zero], h_bkg_y[remove_zero]

    #p0, p1 = np.polyfit(h_bkg_x, np.log(h_bkg_y), 1, w=np.sqrt(h_bkg_y))
    #print(p1, p0)
    popt, pcov = curve_fit(expo, h_bkg_x, h_bkg_y, p0=(100.0, 0.1))    

    plt.figure()
    plt.errorbar(h_bkg_x, h_bkg_y, yerr=np.sqrt(h_bkg_y), fmt='o', label='Data')
    x = np.linspace(4.7, 5.8, 100)
    plt.plot(x, expo(x, *popt), 'r-', label='Background fit')
    #plt.plot(x, scipy.stats.expon.pdf(x, *popt), 'r-', label='Background fit')
    plt.xlabel(r'$m(K^{+}e^{+}e^{-}) [GeV/c^{2}]$')
    plt.ylabel(r'Events')
    plt.legend(loc='upper right')
    plt.savefig('{}_bkgfit_{}.pdf'.format(outputfile, mvaCutReplace), bbox_inches='tight')

    N_BKG = quad(expo, 5.0, 5.4, args=(popt[0], popt[1]))[0] / h_steps
    N_SIG = (LUMI_DATA / LUMI_MC) * (NMC_SELECTED)

    #print(NMC_SELECTED)
    print('MVA cut: {}, Number of selected MC: {}, Expected signal: {}, Background: {}'.format(mvaCut, NMC_SELECTED, N_SIG, N_BKG))
    #N_BKG = fit(h_BToKEE_mass_bkg, mvaCut)
    SList.append(N_SIG)
    BList.append(N_BKG)

  SList = np.array(SList)
  BList = np.array(BList)
  CutBasedWP = {'S': 1561, 'B': 1097, 'SNR': 30.2} # PF
  #CutBasedWP = {'S': 759, 'B': 1394, 'SNR': 16.3} # Mix
  #CutBasedWP = {'S': 140, 'B': 285, 'SNR': 6.8} # Low

  plotSNR(mvaCutList, SList, BList, CutBasedWP)



