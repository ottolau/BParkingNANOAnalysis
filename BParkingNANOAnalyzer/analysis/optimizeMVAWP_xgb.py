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

import uproot
import pandas as pd
import numpy as np
from rootpy.io import root_open
from rootpy.plotting import Hist
from root_numpy import fill_hist, array2root, array2tree
from root_pandas import to_root
from scipy.optimize import curve_fit
from scipy.integrate import quad
import iminuit, probfit
import scipy.stats
import xgboost as xgb
from sklearn.externals import joblib

import ROOT
from ROOT import RooFit
ROOT.gErrorIgnoreLevel=ROOT.kError
ROOT.RooMsgService.instance().setGlobalKillBelow(RooFit.FATAL)


import argparse
parser = argparse.ArgumentParser(description="A simple ttree plotter")
parser.add_argument("-s", "--signal", dest="signal", default="RootTree_BParkingNANO_2019Oct25_BuToKJpsi_Toee_mvaTraining_sig_testing_pf.root", help="Signal file")
parser.add_argument("-b", "--background", dest="background", default="RootTree_BParkingNANO_2019Oct21_Run2018A2A3B2B3C2D2_2020Jan10_mvaTraining_bkg_testing_pf.root", help="Background file")
parser.add_argument("-o", "--outputfile", dest="outputfile", default="test", help="Output file containing plots")
parser.add_argument("-m", "--model", dest="model", default="xgb_fulldata_pf.model", help="Trainned model")
args = parser.parse_args()


ELECTRON_MASS = 0.000511
K_MASS = 0.493677
JPSI_MC = 3.08812
JPSI_SIGMA_MC = 0.04757
JPSI_LOW = JPSI_MC - 3.0*JPSI_SIGMA_MC
JPSI_UP = JPSI_MC + 3.0*JPSI_SIGMA_MC
B_MC = 5.25538
B_SIGMA_MC = 0.07581
B_UP = B_MC + 3.0*B_SIGMA_MC
B_LOW = B_MC - 3.0*B_SIGMA_MC
B_MIN = 4.8
B_MAX = 6.0

def fit(tree, outputfile, mvaCutReplace):
    wspace = ROOT.RooWorkspace('myWorkSpace')
    ROOT.gStyle.SetOptFit(0000);
    ROOT.gROOT.SetBatch(True);
    ROOT.gROOT.SetStyle("Plain");
    ROOT.gStyle.SetGridStyle(3);
    ROOT.gStyle.SetOptStat(000000);
    ROOT.gStyle.SetOptTitle(0)

    thevars = ROOT.RooArgSet()
    xmin, xmax = B_UP, 6.0
    bMass = ROOT.RooRealVar("BToKEE_fit_mass", "m(K^{+}e^{+}e^{-})", 4.0, 6.0, "GeV")
    #dieleMass = ROOT.RooRealVar("BToKEE_mll_fullfit", "m(e^{+}e^{-})", 2.0, 4.0, "GeV")
    wspace.factory('mean[5.27929e+00, 5.2e+00, 5.3e+00]')
    #thevars.add(dieleMass)
    thevars.add(bMass)

    fulldata = ROOT.RooDataSet('fulldata', 'fulldata', tree, ROOT.RooArgSet(thevars))
    theBMassfunc = ROOT.RooFormulaVar("x", "x", "@0", ROOT.RooArgList(bMass) )
    theBMass     = fulldata.addColumn(theBMassfunc) ;
    theBMass.setRange(xmin,xmax);
    thevars.add(theBMass)

    m0 = 3.08812
    si = 0.04757
    #cut = '(BToKEE_mll_fullfit > {}) & (BToKEE_mll_fullfit < {})'.format(m0 - 3.0*si, m0 + 3.0*si)
    cut=''
    print cut    
    data = fulldata.reduce(thevars, cut)
    getattr(wspace,'import')(data, RooFit.Rename("data"))

    # Exponential
    wspace.factory('exp_alpha[-1.0, -100.0, -1.0e-5]')
    alpha = wspace.var('alpha')
    wspace.factory('Exponential::bkg(x,exp_alpha)')

    model = wspace.pdf('bkg')
    bkg = wspace.pdf('bkg')

    # define the set obs = (x)
    wspace.defineSet('obs', 'x')

    # make the set obs known to Python
    obs  = wspace.set('obs')

    ## fit the model to the data.
    results = model.fitTo(data, RooFit.Extended(False), RooFit.Save(), RooFit.Range(xmin,xmax), RooFit.PrintLevel(-1))
    results.Print()

    theBMass.setRange("window",B_LOW,B_UP)
    theBMass.setRange("sideband",B_UP,xmax)
    fracSigRange = bkg.createIntegral(obs,obs,"window") ;
    fracSBRange = bkg.createIntegral(obs,obs,"sideband") ;
    print("Number of background: %f"%(data.sumEntries()*(fracSigRange.getVal()/fracSBRange.getVal())))

    # Plot results of fit on a different frame
    c2 = ROOT.TCanvas('fig_binnedFit', 'fit', 800, 600)
    c2.SetGrid()
    c2.cd()
    ROOT.gPad.SetLeftMargin(0.10)
    ROOT.gPad.SetRightMargin(0.05)

    #xframe = wspace.var('x').frame(RooFit.Title("PF electron"))
    xframe = theBMass.frame()
    data.plotOn(xframe, RooFit.Binning(10), RooFit.Name("data"))
    model.plotOn(xframe,RooFit.Name("global"),RooFit.LineColor(2),RooFit.MoveToBack()) # this will show fit overlay on canvas

    xframe.GetYaxis().SetTitleOffset(0.9)
    xframe.GetYaxis().SetTitleFont(42)
    xframe.GetYaxis().SetTitleSize(0.05)
    xframe.GetYaxis().SetLabelSize(0.04)
    xframe.GetYaxis().SetLabelFont(42)
    xframe.GetXaxis().SetTitleOffset(0.9)
    xframe.GetXaxis().SetTitleFont(42)
    xframe.GetXaxis().SetTitleSize(0.05)
    xframe.GetXaxis().SetLabelSize(0.04)
    xframe.GetXaxis().SetLabelFont(42)

    xframe.GetYaxis().SetTitle("Events")
    xframe.GetXaxis().SetTitle("m(K^{+}e^{+}e^{-}) [GeV]")
    xframe.SetStats(0)
    xframe.SetMinimum(0)
    xframe.Draw()

    legend = ROOT.TLegend(0.65,0.75,0.92,0.85);
    legend.SetTextFont(72);
    legend.SetTextSize(0.04);
    legend.AddEntry(xframe.findObject("data"),"Data","lpe");
    legend.AddEntry(xframe.findObject("global"),"Fit","l");
    legend.Draw();

    c2.cd()
    c2.Update()

    c2.SaveAs('{}_bkgfit_{}.pdf'.format(outputfile, mvaCutReplace))
    return data.sumEntries()*(fracSigRange.getVal()/fracSBRange.getVal())


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
    ax3.plot(sig, sig/np.sqrt(sig+bkg), 'bo', label='XGB')
    #ax3.plot(CutBasedWP['S'], CutBasedWP['SNR'], 'r*', label='Cut-based')
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
  LUMI_DATA = 13.998514728133333 - 1.101911423 # fb-1
  CutBasedWP = {'S': 1451.0, 'B': 818.0, 'SNR': 30.2} # PF
  #CutBasedWP = {'S': 759, 'B': 1394, 'SNR': 16.3} # Mix
  #CutBasedWP = {'S': 140, 'B': 285, 'SNR': 6.8} # Low

  print("MC lumiosity: {}".format(LUMI_MC))
  outputfile = args.outputfile.replace('.root','').replace('.h5','')

  #inputfile_sig = args.signal.replace('.h5','').replace('.root','')+'.h5'
  #inputfile_bkg = args.background.replace('.h5','').replace('.root','')+'.h5'
  #df = {}
  #df['sig'] = pd.read_hdf(inputfile_sig, 'branches')
  #df['bkg'] = pd.read_hdf(inputfile_bkg, 'branches')

  filename = {}
  upfile = {}
  params = {}
  df = {}

  filename['sig'] = args.signal.replace('.h5','').replace('.root','')+'.root'
  filename['bkg'] = args.background.replace('.h5','').replace('.root','')+'.root'

  upfile['bkg'] = uproot.open(filename['bkg'])
  upfile['sig'] = uproot.open(filename['sig'])

  params['bkg'] = upfile['bkg']['tree'].arrays()
  params['sig'] = upfile['sig']['tree'].arrays()

  df['sig'] = pd.DataFrame(params['sig']).sort_index(axis=1)
  df['bkg'] = pd.DataFrame(params['bkg']).sort_index(axis=1)

  output_branches = {}
  #training_branches = sorted(['BToKEE_fit_l1_normpt', 'BToKEE_fit_l1_eta', 'BToKEE_fit_l1_phi', 'BToKEE_l1_dxy_sig', 'BToKEE_l1_dz', 'BToKEE_l1_mvaId', 'BToKEE_fit_l2_normpt', 'BToKEE_fit_l2_eta', 'BToKEE_fit_l2_phi', 'BToKEE_l2_dxy_sig', 'BToKEE_l2_dz', 'BToKEE_l2_mvaId', 'BToKEE_fit_k_normpt', 'BToKEE_fit_k_eta', 'BToKEE_fit_k_phi', 'BToKEE_k_DCASig', 'BToKEE_fit_normpt', 'BToKEE_svprob', 'BToKEE_fit_cos2D', 'BToKEE_l_xy_sig'])

  training_branches = sorted(['BToKEE_fit_l1_normpt', 'BToKEE_fit_l1_eta', 'BToKEE_fit_l1_phi', 'BToKEE_l1_dxy_sig', 'BToKEE_l1_dz', 'BToKEE_fit_l2_normpt', 'BToKEE_fit_l2_eta', 'BToKEE_fit_l2_phi', 'BToKEE_l2_dxy_sig', 'BToKEE_l2_dz', 'BToKEE_fit_k_normpt', 'BToKEE_fit_k_eta', 'BToKEE_fit_k_phi', 'BToKEE_k_DCASig', 'BToKEE_fit_normpt', 'BToKEE_svprob', 'BToKEE_fit_cos2D', 'BToKEE_l_xy_sig'])


  for k, branches in df.items():

    jpsi_selection = (branches['BToKEE_mll_fullfit'] > JPSI_LOW) & (branches['BToKEE_mll_fullfit'] < JPSI_UP)
    #b_selection = jpsi_selection & (branches['BToKEE_fit_mass'] > B_LOWSB_UP) & (branches['BToKEE_fit_mass'] < B_UPSB_LOW)
    b_upsb_selection = jpsi_selection & (branches['BToKEE_fit_mass'] > B_UP)
    b_sb_selection = b_upsb_selection

    general_selection = (branches['BToKEE_l1_mvaId'] > 3.94) & (branches['BToKEE_l2_mvaId'] > 3.94)

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
    #df[k] = branches[general_selection & mix_net_selection].copy()
    #df[k] = branches[general_selection & low_pfveto_selection].copy()
    df[k].replace([np.inf, -np.inf], 10.0**+10, inplace=True)


    # count the number of b candidates passes the selection
    #count_selection = jpsi_selection 
    #nBToKEE_selected = self._branches['BToKEE_event'][count_selection].values
    #_, nBToKEE_selected = np.unique(nBToKEE_selected[np.isfinite(nBToKEE_selected)], return_counts=True)


  # add mva id to pandas dataframe
  #print(df['sig'], df['bkg'])

  #cutBased_selection = (df['sig']['BToKEE_pt'] > 10.0) & (df['sig']['BToKEE_l_xy_sig'] > 6.0 ) & (df['sig']['BToKEE_svprob'] > 0.1) & (df['sig']['BToKEE_cos2D'] > 0.999)
  #NMC_CUTBASED = float(df['sig'][cutBased_selection]['BToKEE_mass'].count())
  #print(NMC_CUTBASED)

  model = xgb.Booster({'nthread': 4})
  #model = xgb.Booster()
  model.load_model(args.model)
  #model = joblib.load('xgb_cv_pf.joblib.dat')
  mva_sig = model.predict(xgb.DMatrix(df['sig'][training_branches].sort_index(axis=1).values))
  mva_bkg = model.predict(xgb.DMatrix(df['bkg'][training_branches].sort_index(axis=1).values))

  print('min: {}, max: {}'.format(min(mva_sig),max(mva_bkg)))
  df['sig']['BToKEE_xgb'] = mva_sig
  df['bkg']['BToKEE_xgb'] = mva_bkg

  SList = []
  SErrList = []
  BList = []
  
  mvaCutList = np.linspace(4.0, 8.0, 20)
  #mvaCutList = np.linspace(3.0, 4.0, 1)

  for mvaCut in mvaCutList:
    mvaCutReplace = '{0:.3f}'.format(mvaCut).replace('.','_')
    # mva selection
    selected_branches_sig = df['sig'][(df['sig']['BToKEE_xgb'] > mvaCut)]['BToKEE_fit_mass']
    selected_branches_bkg = df['bkg'][(df['bkg']['BToKEE_xgb'] > mvaCut)].sort_values('BToKEE_xgb', ascending=False).drop_duplicates(['BToKEE_event'], keep='first')['BToKEE_fit_mass'].values
    #selected_branches_bkg = df['bkg'][(df['bkg']['BToKEE_xgb'] > mvaCut)].sort_values('BToKEE_xgb', ascending=False).drop_duplicates(['BToKEE_event'], keep='first')

    #selected_branches_sig = df['sig'][(df['sig']['BToKEE_xgb'] > mvaCut)].sort_values('BToKEE_xgb', ascending=False).drop_duplicates(['BToKEE_event'], keep='first')['BToKEE_fit_mass']
    #selected_branches_bkg = df['bkg'][(df['bkg']['BToKEE_xgb'] > mvaCut)].sort_values('BToKEE_xgb', ascending=False).drop_duplicates(['BToKEE_event'], keep='first')['BToKEE_fit_mass'].values

    NMC_SELECTED = float(selected_branches_sig.count())

    
    h_BToKEE_mass_bkg = Hist(50, 4.5, 6.0, name='h_BToKEE_mass_bkg', title='', type='F') 
    fill_hist(h_BToKEE_mass_bkg, selected_branches_bkg[np.isfinite(selected_branches_bkg)])
    #selected_branches_bkg = selected_branches_bkg[np.isfinite(selected_branches_bkg)]
    #popt, pocv = unbinned_exp_LLH(selected_branches_bkg, loc_init = 0, scale_init = 0.5, limit_loc = (-1, 1), limit_scale = (-1, 1))
    #print(popt)

    h_bins, h_steps = np.linspace(4.5, 6.0, 30, retstep=True)
    h_bkg_y, h_bkg_x = np.histogram(selected_branches_bkg, bins=h_bins)
    h_bkg_x = (h_bkg_x[:-1] + h_bkg_x[1:]) / 2
    remove_zero = np.where(np.greater(h_bkg_y, 1.0))
    h_bkg_x, h_bkg_y = h_bkg_x[remove_zero], h_bkg_y[remove_zero]

    #p0, p1 = np.polyfit(h_bkg_x, np.log(h_bkg_y), 1, w=np.sqrt(h_bkg_y))
    #print(p1, p0)
    popt, pcov = curve_fit(expo, h_bkg_x, h_bkg_y, p0=(100.0, 0.1))    

    plt.figure()
    plt.errorbar(h_bkg_x, h_bkg_y, yerr=np.sqrt(h_bkg_y), fmt='o', label='Data')
    x = np.linspace(4.5, 6.0, 100)
    plt.plot(x, expo(x, *popt), 'r-', label='Background fit')
    #plt.plot(x, scipy.stats.expon.pdf(x, *popt), 'r-', label='Background fit')
    plt.xlabel(r'$m(K^{+}e^{+}e^{-}) [GeV/c^{2}]$')
    plt.ylabel(r'Events')
    plt.legend(loc='upper right')
    plt.savefig('{}_bkgfit_{}.pdf'.format(outputfile, mvaCutReplace), bbox_inches='tight')

    N_BKG = quad(expo, B_LOW, B_UP, args=(popt[0], popt[1]))[0] / h_steps / 0.25#0.25
    
    #tree = array2tree(np.array(branches[['BToKEE_fit_mass','BToKEE_mll_fullfit']], dtype=[('BToKEE_fit_mass', 'f4'), ('BToKEE_mll_fullfit', 'f4')]))
    #tree = array2tree(np.array(selected_branches_bkg['BToKEE_fit_mass'], dtype=[('BToKEE_fit_mass', 'f4')]))


    #N_BKG = fit(tree, outputfile, mvaCutReplace) / 0.25
    N_SIG = (LUMI_DATA / LUMI_MC) * (NMC_SELECTED) / 0.25#0.25
    #N_SIG = CutBasedWP['S'] * (NMC_SELECTED / NMC_CUTBASED)

    #print(NMC_SELECTED)
    print('MVA cut: {}, Number of selected MC: {}, Expected signal: {}, Background: {}, SNR: {}'.format(mvaCut, NMC_SELECTED, N_SIG, N_BKG, N_SIG/np.sqrt(N_SIG + N_BKG)))
    #N_BKG = fit(h_BToKEE_mass_bkg, mvaCut)
    SList.append(N_SIG)
    BList.append(N_BKG)

  SList = np.array(SList)
  BList = np.array(BList)

  plotSNR(mvaCutList, SList, BList, CutBasedWP)



