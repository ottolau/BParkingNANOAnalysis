import pandas as pd
import numpy as np
from rootpy.io import root_open
from rootpy.plotting import Hist
from root_numpy import fill_hist, array2root, array2tree
from root_pandas import to_root
from keras.models import load_model
import ROOT
from ROOT import RooFit
ROOT.gErrorIgnoreLevel=ROOT.kError
ROOT.RooMsgService.instance().setGlobalKillBelow(RooFit.FATAL)


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

def fit(tree, mvaCut):
  wspace = ROOT.RooWorkspace('myWorkSpace')
  ROOT.gStyle.SetOptFit(0000);
  ROOT.gROOT.SetBatch(True);

  mvaCutReplace = '{0:.3f}'.format(mvaCut).replace('.','_')

  xmin = 4.7 
  xmax = 6.0 

  bMass = ROOT.RooRealVar("BToKEE_mass", "m(K^{+}e^{+}e^{-})", 4.7, 6.0, "GeV")
  thevars = ROOT.RooArgSet()
  thevars.add(bMass)

  fulldata = ROOT.RooDataSet('fulldata', 'fulldata', tree, ROOT.RooArgSet(thevars))
  theBMassfunc = ROOT.RooFormulaVar("x", "x", "@0", ROOT.RooArgList(bMass) )
  theBMass     = fulldata.addColumn(theBMassfunc) ;
  theBMass.setRange(xmin,xmax);
  thevars.add(theBMass)
  #x = wspace.set('x')

  cut = ''

  print cut    
  data = fulldata.reduce(thevars, cut)

  getattr(wspace,'import')(data, RooFit.Rename("data"))

  wspace.factory('nsig[100.0, 0.0, 100000.0]')
  wspace.factory('nbkg[500.0, 0.0, 1000000.0]')
  sigPDF = 2
  bkgPDF = 2

  if sigPDF == 0:
      # Voigtian
      wspace.factory('mean[5.27929e+00, 5.2e+00, 5.3e+00]')
      wspace.factory('width[1.000e-02, 1.000e-04, 1.000e-01]')
      wspace.factory('sigma[7.1858e-02, 1.e-3, 1.e-1]')
      width = wspace.var('width')
      wspace.factory('Voigtian::sig(x,mean,width,sigma)')

  if sigPDF == 1:
      # Gaussian
      wspace.factory('mean[5.2418e+00, 5.20e+00, 5.35e+00]')
      #wspace.factory('mean[3.0969+00, 3.06e+00, 3.10e+00]')
      wspace.factory('sigma[7.1858e-02, 1.e-3, 5.e-1]')
      wspace.factory('Gaussian::sig(x,mean,sigma)')

  if sigPDF == 2:
      # Crystal-ball
      wspace.factory('mean[5.2418e+00, 5.20e+00, 5.35e+00]')
      wspace.factory('sigma[7.1858e-02, 1.e-3, 5.e-1]')
      wspace.factory('alpha[1.0e-1, 0.0, 1.0]')
      wspace.factory('n[5, 1, 10]')
      wspace.factory('CBShape::sig(x,mean,sigma,alpha,n)')

  if bkgPDF == 0:
      # Polynomial
      wspace.factory('c0[1.0, -1.0, 1.0]')
      wspace.factory('c1[-0.1, -1.0, 1.0]')
      wspace.factory('c2[-0.1, -1.0, 1.0]')
      c0 = wspace.var('c0')
      c1 = wspace.var('c1')
      c2 = wspace.var('c2')
      wspace.factory('Chebychev::bkg(x,{c0,c1,c2})')

  if bkgPDF == 1:
      wspace.factory('c1[0.0, -100.0, 100.0]')
      c1 = wspace.var('c1')
      wspace.factory('Polynomial::bkg(x,{c1})')

  if bkgPDF == 2:
      # Exponential
      wspace.factory('exp_alpha[-1.0, -100.0, -1.0e-5]')
      alpha = wspace.var('alpha')
      wspace.factory('Exponential::bkg(x,exp_alpha)')

  #x = wspace.var('x')
  mean = wspace.var('mean')
  sigma = wspace.var('sigma')
  nsig = wspace.var('nsig')
  nbkg = wspace.var('nbkg')

  #parameters = ['c0', 'c1', 'c2', 'mean', 'width', 'sigma', 'nsig', 'nbkg']
 
  # NUMBER OF PARAMETERS
  #P = len(parameters)
  
  wspace.factory('SUM::model(nsig*sig,nbkg*bkg)')
          
  model = wspace.pdf('model')
  bkg = wspace.pdf('bkg')
  sig = wspace.pdf('sig')

  # define the set obs = (x)
  wspace.defineSet('obs', 'x')

  # make the set obs known to Python
  obs  = wspace.set('obs')

  ## fit the model to the data.
  results = model.fitTo(data, RooFit.Extended(True), RooFit.Save(), RooFit.Range(xmin,xmax), RooFit.PrintLevel(-1))
  results.Print()

  #B_SIGNAL_LOW = mean.getVal() - 3.0*sigma.getVal()
  #B_SIGNAL_UP = mean.getVal() + 3.0*sigma.getVal()
  B_SIGNAL_LOW = 5.0
  B_SIGNAL_UP = 5.4
  theBMass.setRange("window",B_SIGNAL_LOW,B_SIGNAL_UP) ;
  fracBkgRange = bkg.createIntegral(obs,obs,"window") ;

  #fracBkgRange = bkg.createIntegral(bkgRangeArgSet,"window") ;
  nbkgWindow = nbkg.getVal() * fracBkgRange.getVal()
  #print(nbkg.getVal(), fracBkgRange.getVal())
  print("MAV cut: %f, Number of signals: %f, Number of background: %f, S/sqrt(S+B): %f"%(mvaCut, nsig.getVal(), nbkgWindow, nsig.getVal()/np.sqrt(nsig.getVal() + nbkgWindow)))

  # Plot results of fit on a different frame
  c2 = ROOT.TCanvas('fig_binnedFit', 'fit', 800, 600)
  c2.SetGrid()
  c2.cd()
  ROOT.gPad.SetLeftMargin(0.10)
  ROOT.gPad.SetRightMargin(0.05)

  #xframe = wspace.var('x').frame(RooFit.Title("PF electron"))
  xframe = theBMass.frame(RooFit.Title("MVA cut: {}".format(mvaCut)))
  data.plotOn(xframe, RooFit.Binning(50), RooFit.Name("data"))
  model.plotOn(xframe,RooFit.Name("global"),RooFit.LineColor(2),RooFit.MoveToBack()) # this will show fit overlay on canvas
  model.plotOn(xframe,RooFit.Name("bkg"),RooFit.Components("bkg"),RooFit.LineStyle(ROOT.kDashed),RooFit.LineColor(ROOT.kMagenta),RooFit.MoveToBack()) ;
  model.plotOn(xframe,RooFit.Name("sig"),RooFit.Components("sig"),RooFit.DrawOption("FL"),RooFit.FillColor(9),RooFit.FillStyle(3004),RooFit.LineStyle(6),RooFit.LineColor(9)) ;
  model.plotOn(xframe,RooFit.VisualizeError(results), RooFit.FillColor(ROOT.kOrange), RooFit.MoveToBack()) # this will show fit overlay on canvas

  xframe.GetYaxis().SetTitleOffset(0.9)
  xframe.GetYaxis().SetTitleFont(42)
  xframe.GetYaxis().SetTitleSize(0.05)
  xframe.GetYaxis().SetLabelSize(0.065)
  xframe.GetYaxis().SetLabelSize(0.04)
  xframe.GetYaxis().SetLabelFont(42)
  xframe.GetXaxis().SetTitleOffset(0.9)
  xframe.GetXaxis().SetTitleFont(42)
  xframe.GetXaxis().SetTitleSize(0.05)
  xframe.GetXaxis().SetLabelSize(0.065)
  xframe.GetXaxis().SetLabelSize(0.04)
  xframe.GetXaxis().SetLabelFont(42)

  xframe.GetYaxis().SetTitle("Events")
  xframe.GetXaxis().SetTitle("m(K^{+}e^{+}e^{-}) [GeV/c^{2}]")
  #xframe.GetXaxis().SetTitle("m(e^{+}e^{-}) [GeV/c^{2}]")
  xframe.SetStats(0)
  xframe.SetMinimum(0)
  xframe.Draw()

  legend = ROOT.TLegend(0.65,0.65,0.92,0.85);
  #legend = ROOT.TLegend(0.65,0.15,0.92,0.35);
  legend.SetTextFont(72);
  legend.SetTextSize(0.04);
  legend.AddEntry(xframe.findObject("data"),"Data","lpe");
  legend.AddEntry(xframe.findObject("bkg"),"Background fit","l");
  legend.AddEntry(xframe.findObject("sig"),"Signal fit","l");
  legend.AddEntry(xframe.findObject("global"),"Global Fit","l");
  legend.Draw();

  c2.cd()
  c2.Update()

  c2.SaveAs('{}_mvaCut{}.pdf'.format(args.outputfile, mvaCutReplace))
  return nsig.getVal(), nsig.getError(), nbkgWindow

def plotSNR(cut, sig, sigError, bkg):
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

    CutBasedWP = {'S': 1561, 'B': 1097, 'SNR': 30.2}

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

    
    fig3, ax3 = plt.subplots()
    plt.grid(linestyle='--')
    ax3.plot(sig, sig/np.sqrt(sig+bkg), 'bo', label='Keras')
    ax3.plot(CutBasedWP['S'], CutBasedWP['SNR'], 'r*', label='Cut-based')
    ax3.set_xlabel(r'S')
    ax3.set_ylabel(r'$S/\sqrt{S+B}$')
    ax3.legend(loc=2)
    fig3.savefig('{}_S_SNRPlot.pdf'.format(args.outputfile), bbox_inches='tight')


if __name__ == "__main__":
  inputfile = args.inputfile.replace('.h5','')+'.h5'
  outputfile = args.outputfile.replace('.root','').replace('.h5','')

  branches = pd.read_hdf(inputfile, 'branches')
  output_branches = {}
  training_branches = sorted(['BToKEE_l1_normpt', 'BToKEE_l1_eta', 'BToKEE_l1_phi', 'BToKEE_l1_dxy_sig', 'BToKEE_l1_dz', 'BToKEE_l2_normpt', 'BToKEE_l2_eta', 'BToKEE_l2_phi', 'BToKEE_l2_dxy_sig', 'BToKEE_l2_dz', 'BToKEE_k_normpt', 'BToKEE_k_eta', 'BToKEE_k_phi', 'BToKEE_k_DCASig', 'BToKEE_normpt', 'BToKEE_svprob', 'BToKEE_cos2D', 'BToKEE_l_xy_sig'])
  #training_branches = sorted(['BToKEE_svprob', 'BToKEE_cos2D', 'BToKEE_l_xy_sig'])

  jpsi_selection = (branches['BToKEE_mll_raw'] > JPSI_LOW) & (branches['BToKEE_mll_raw'] < JPSI_UP)
  b_selection = jpsi_selection & (branches['BToKEE_mass'] > B_LOWSB_UP) & (branches['BToKEE_mass'] < B_UPSB_LOW)
  b_lowsb_selection = jpsi_selection & (branches['BToKEE_mass'] > B_LOWSB_LOW) & (branches['BToKEE_mass'] < B_LOWSB_UP)
  b_upsb_selection = jpsi_selection & (branches['BToKEE_mass'] > B_UPSB_LOW) & (branches['BToKEE_mass'] < B_UPSB_UP)
  b_sb_selection = b_lowsb_selection | b_upsb_selection

  general_selection = jpsi_selection & (branches['BToKEE_l1_mvaId'] > 3.94) & (branches['BToKEE_l2_mvaId'] > 3.94) & (branches['BToKEE_k_pt'] > 1.5) 

  branches = branches[general_selection]
  branches['BToKEE_normpt'] = branches['BToKEE_pt'] / branches['BToKEE_mass']
  branches.replace([np.inf, -np.inf], 10.0**+10, inplace=True)

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

  # count the number of b candidates passes the selection
  #count_selection = jpsi_selection 
  #nBToKEE_selected = self._branches['BToKEE_event'][count_selection].values
  #_, nBToKEE_selected = np.unique(nBToKEE_selected[np.isfinite(nBToKEE_selected)], return_counts=True)

  # add mva id to pandas dataframe

  model = load_model(args.model)
  branches['BToKEE_keras_pf'] = model.predict(branches[training_branches].sort_index(axis=1).values)

  SList = []
  SErrList = []
  BList = []
  
  mvaCutList = np.linspace(0.8, 0.99, 20)
  for mvaCut in mvaCutList:
    # mva selection
    mva_selection = (branches['BToKEE_keras_pf'] > mvaCut) #& (branches['BToKEE_keras_pf'] < 0.999)
    selected_branches = np.array(branches[pf_selection & mva_selection]['BToKEE_mass'], dtype=[('BToKEE_mass', 'f4')])
    tree = array2tree(selected_branches)
    S, SErr, B = fit(tree, mvaCut) 
    SList.append(S)
    SErrList.append(SErr)
    BList.append(B)

  SList = np.array(SList)
  SErrList = np.array(SErrList)
  BList = np.array(BList)
  plotSNR(mvaCutList, SList, SErrList, BList)

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


