import pandas as pd
import numpy as np
from rootpy.io import root_open
from rootpy.plotting import Hist
from root_numpy import fill_hist, array2root, array2tree
from root_pandas import to_root
from keras.models import load_model
import xgboost as xgb
import ROOT
from ROOT import RooFit
ROOT.gErrorIgnoreLevel=ROOT.kError
ROOT.RooMsgService.instance().setGlobalKillBelow(RooFit.FATAL)


import argparse
parser = argparse.ArgumentParser(description="A simple ttree plotter")
parser.add_argument("-i", "--inputfile", dest="inputfile", default="DoubleMuonNtu_Run2016B.list", help="List of input ggNtuplizer files")
parser.add_argument("-o", "--outputfile", dest="outputfile", default="test", help="Output file containing plots")
parser.add_argument("-p", "--pfmodel", dest="pfmodel", default="xgb_fulldata_pf.model", help="Trainned PF model")
parser.add_argument("-m", "--mixmodel", dest="mixmodel", default="xgb_fulldata_mix_net.model", help="Trainned Mix model")
parser.add_argument("-l", "--lowmodel", dest="lowmodel", default="xgb_fulldata_low.model", help="Trainned Low model")
parser.add_argument("-s", "--hist", dest="hist", action='store_true', help="Store histograms or tree")
args = parser.parse_args()

outputbranches = {'BToKEE_mll_raw': {'nbins': 50, 'xmin': 0.0, 'xmax': 5.0},
                  'BToKEE_mll_fullfit': {'nbins': 30, 'xmin': 2.6, 'xmax': 3.6},
                  'BToKEE_mll_llfit': {'nbins': 30, 'xmin': 2.6, 'xmax': 3.6},
                  'BToKEE_mass': {'nbins': 50, 'xmin': 4.7, 'xmax': 6.0},
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
                  #'BToKEE_l1_unBiased': {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0},
                  #'BToKEE_l2_unBiased': {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0},
                  #'BToKEE_l1_ptBiased': {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0},
                  #'BToKEE_l2_ptBiased': {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0},
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
                  'BToKEE_keras_low': {'nbins': 50, 'xmin': 0.0, 'xmax': 1.0},
                  'BToKEE_keras_mix': {'nbins': 50, 'xmin': 0.0, 'xmax': 1.0},
                  }
'''
outputbranches = {'BToKEE_mass_all': {'nbins': 50, 'xmin': 4.7, 'xmax': 6.0},
                  'BToKEE_mass_pf': {'nbins': 50, 'xmin': 4.7, 'xmax': 6.0},
                  'BToKEE_mass_low_pfveto': {'nbins': 50, 'xmin': 4.7, 'xmax': 6.0},
                  'BToKEE_mass_mix_net': {'nbins': 50, 'xmin': 4.7, 'xmax': 6.0},
                  }
'''

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

def CMS_lumi():
    mark = ROOT.TLatex()
    mark.SetNDC()
    lumistamp = '2018 (13 TeV)'
    fontScale = 1.0
    cmsTextSize = 0.042 * fontScale * 1.25
    extraOverCmsTextSize  = 0.76
    extraTextSize = extraOverCmsTextSize*cmsTextSize

    mark.SetTextAlign(11)
    mark.SetTextSize(cmsTextSize)
    mark.SetTextFont(61)
    mark.DrawLatex(ROOT.gPad.GetLeftMargin(), 1 - (ROOT.gPad.GetTopMargin() - 0.017), "CMS")
    mark.SetTextSize(0.042 * fontScale)
    mark.SetTextFont(52)
    mark.DrawLatex(ROOT.gPad.GetLeftMargin() + 0.09, 1 - (ROOT.gPad.GetTopMargin() - 0.017), "Preliminary")
    mark.SetTextSize(extraTextSize)
    mark.SetTextFont(42)
    mark.SetTextAlign(31)
    mark.DrawLatex(1 - ROOT.gPad.GetRightMargin(), 1 - (ROOT.gPad.GetTopMargin() - 0.017), lumistamp)

def fit(tree, eType):
  wspace = ROOT.RooWorkspace('myWorkSpace')
  ROOT.gStyle.SetOptFit(0000);
  ROOT.gROOT.SetBatch(True);
  ROOT.gROOT.SetStyle("Plain");
  ROOT.gStyle.SetGridStyle(3);
  ROOT.gStyle.SetOptStat(000000);
  ROOT.gStyle.SetOptTitle(0)

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
  sigPDF = 4
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
      wspace.factory('sigma[7.1858e-02, 1.e-4, 5.e-1]')
      wspace.factory('alpha[1.0e-1, 0.0, 100.0]')
      wspace.factory('n[10, 10, 10]')
      wspace.factory('CBShape::sig(x,mean,sigma,alpha,n)')

  if sigPDF == 4:
      # Double Crystal-ball
      '''
      wspace.factory('mean[5.2418e+00, 5.20e+00, 5.35e+00]')
      wspace.factory('sigma[7.1858e-02, 1.e-6, 5.e-1]')
      wspace.factory('alpha[1.0e-1, 0.0, 100.0]')
      wspace.factory('n[1, 5, 100]')
      wspace.factory('CBShape::cb1(x,mean,sigma,alpha,n)')
      wspace.factory('sigma2[7.1858e-02, 1.e-6, 5.e-1]')
      wspace.factory('alpha2[1.0e-1, 0.0, 100.0]')
      wspace.factory('n2[1, 5, 100]')
      wspace.factory('CBShape::cb2(x,mean,sigma2,alpha2,n2)')
      wspace.factory('f1[0.5, 0.0, 1.0]')
      '''
      # MVA
      if eType == 'pf':
        wspace.factory('mean[5.2701e+00, 5.2701e+00, 5.2701e+00]')
        wspace.factory('sigma[1.0869e-01, 1.0869e-01, 1.0869e-01]')
        wspace.factory('alpha[2.6632e+00, 2.6632e+00, 2.6632e+00]')
        wspace.factory('n[5.2500e+01 , 5.2500e+01, 5.2500e+01]')
        wspace.factory('CBShape::cb1(x,mean,sigma,alpha,n)')
        wspace.factory('sigma2[3.2178e-02, 3.2178e-02, 3.2178e-02]')
        wspace.factory('alpha2[4.1014e-01, 4.1014e-01, 4.1014e-01]')
        wspace.factory('n2[5.2500e+01, 5.2500e+01, 5.2500e+01]')
        wspace.factory('CBShape::cb2(x,mean,sigma2,alpha2,n2)')
        wspace.factory('f1[6.1236e-01, 6.1236e-01, 6.1236e-01]')
      else:
        wspace.factory('mean[5.2695e+00, 5.2695e+00, 5.2695e+00]')
        wspace.factory('sigma[1.1716e-01, 1.1716e-01, 1.1716e-01]')
        wspace.factory('alpha[3.0430e+00, 3.0430e+00, 3.0430e+00]')
        wspace.factory('n[5.2500e+01 , 5.2500e+01, 5.2500e+01]')
        wspace.factory('CBShape::cb1(x,mean,sigma,alpha,n)')
        wspace.factory('sigma2[5.2612e-02, 5.2612e-02, 5.2612e-02]')
        wspace.factory('alpha2[8.3173e-01, 8.3173e-01, 8.3173e-01]')
        wspace.factory('n2[5.2500e+01, 5.2500e+01, 5.2500e+01]')
        wspace.factory('CBShape::cb2(x,mean,sigma2,alpha2,n2)')
        wspace.factory('f1[5.0383e-01, 5.0383e-01, 5.0383e-01]')

      wspace.factory('SUM::sig(f1*cb1, cb2)')

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
  print("Number of signals: %f, Number of background: %f, S/sqrt(S+B): %f"%(nsig.getVal(), nbkgWindow, nsig.getVal()/np.sqrt(nsig.getVal() + nbkgWindow)))

  # Plot results of fit on a different frame
  c2 = ROOT.TCanvas('fig_binnedFit', 'fit', 800, 600)
  c2.SetGrid()
  c2.cd()
  ROOT.gPad.SetLeftMargin(0.10)
  ROOT.gPad.SetRightMargin(0.05)

  #xframe = wspace.var('x').frame(RooFit.Title("PF electron"))
  xframe = theBMass.frame()
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
  xframe.GetXaxis().SetTitle("m(K^{+}e^{+}e^{-}) [GeV]")
  #xframe.GetXaxis().SetTitle("m(e^{+}e^{-}) [GeV/c^{2}]")
  xframe.SetStats(0)
  xframe.SetMinimum(0)
  xframe.Draw()

  CMS_lumi()

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

  c2.SaveAs('{}_mvafit_{}.pdf'.format(args.outputfile, eType))
  return nsig.getVal(), nsig.getError(), nbkgWindow


if __name__ == "__main__":
  inputfile = args.inputfile.replace('.h5','')+'.h5'
  outputfile = args.outputfile.replace('.root','').replace('.h5','')

  ele_type = {'all': False, 'pf': False, 'low_pfveto': False, 'mix_net': True}
  ele_selection = {'all': 'all_mva_selection', 'pf': 'pf_mva_selection', 'low_pfveto': 'low_mva_selection', 'mix_net': 'mix_mva_selection'}

  branches = pd.read_hdf(inputfile, 'branches')
  output_branches = {}
  pf_training_branches = sorted(['BToKEE_l1_normpt', 'BToKEE_l1_eta', 'BToKEE_l1_phi', 'BToKEE_l1_dxy_sig', 'BToKEE_l1_dz', 'BToKEE_l2_normpt', 'BToKEE_l2_eta', 'BToKEE_l2_phi', 'BToKEE_l2_dxy_sig', 'BToKEE_l2_dz', 'BToKEE_k_normpt', 'BToKEE_k_eta', 'BToKEE_k_phi', 'BToKEE_k_DCASig', 'BToKEE_normpt', 'BToKEE_svprob', 'BToKEE_cos2D', 'BToKEE_l_xy_sig'])
  mix_training_branches = sorted(['BToKEE_l1_normpt', 'BToKEE_l1_eta', 'BToKEE_l1_phi', 'BToKEE_l1_dxy_sig', 'BToKEE_l1_dz', 'BToKEE_l1_mvaId', 'BToKEE_l2_normpt', 'BToKEE_l2_eta', 'BToKEE_l2_phi', 'BToKEE_l2_dxy_sig', 'BToKEE_l2_dz', 'BToKEE_l2_mvaId', 'BToKEE_k_normpt', 'BToKEE_k_eta', 'BToKEE_k_phi', 'BToKEE_k_DCASig', 'BToKEE_normpt', 'BToKEE_svprob', 'BToKEE_cos2D', 'BToKEE_l_xy_sig'])
  low_training_branches = sorted(['BToKEE_l1_normpt', 'BToKEE_l1_eta', 'BToKEE_l1_phi', 'BToKEE_l1_dxy_sig', 'BToKEE_l1_dz', 'BToKEE_l1_mvaId', 'BToKEE_l2_normpt', 'BToKEE_l2_eta', 'BToKEE_l2_phi', 'BToKEE_l2_dxy_sig', 'BToKEE_l2_dz', 'BToKEE_l2_mvaId', 'BToKEE_k_normpt', 'BToKEE_k_eta', 'BToKEE_k_phi', 'BToKEE_k_DCASig', 'BToKEE_normpt', 'BToKEE_svprob', 'BToKEE_cos2D', 'BToKEE_l_xy_sig'])

  jpsi_selection = (branches['BToKEE_mll_raw'] > JPSI_LOW) & (branches['BToKEE_mll_raw'] < JPSI_UP)
  b_selection = jpsi_selection & (branches['BToKEE_mass'] > B_LOWSB_UP) & (branches['BToKEE_mass'] < B_UPSB_LOW)
  b_lowsb_selection = jpsi_selection & (branches['BToKEE_mass'] > B_LOWSB_LOW) & (branches['BToKEE_mass'] < B_LOWSB_UP)
  b_upsb_selection = jpsi_selection & (branches['BToKEE_mass'] > B_UPSB_LOW) & (branches['BToKEE_mass'] < B_UPSB_UP)
  b_sb_selection = b_lowsb_selection | b_upsb_selection

  general_selection = jpsi_selection & (branches['BToKEE_l1_mvaId'] > 3.94) & (branches['BToKEE_l2_mvaId'] > 3.94) & (branches['BToKEE_k_pt'] > 1.0) 

  branches = branches[general_selection]
  branches['BToKEE_normpt'] = branches['BToKEE_pt'] / branches['BToKEE_mass']
  branches.replace([np.inf, -np.inf], 10.0**+10, inplace=True)

  # additional cuts, allows various lengths

  l1_pf_selection = (branches['BToKEE_l1_isPF'])
  l2_pf_selection = (branches['BToKEE_l2_isPF'])
  l1_low_selection = (branches['BToKEE_l1_isLowPt']) #& (branches['BToKEE_l1_pt'] < 5.0)
  l2_low_selection = (branches['BToKEE_l2_isLowPt']) #& (branches['BToKEE_l2_pt'] < 5.0)

  pf_selection = l1_pf_selection & l2_pf_selection & (branches['BToKEE_k_pt'] > 1.5)
  low_selection = l1_low_selection & l2_low_selection
  overlap_veto_selection = np.logical_not(branches['BToKEE_l1_isPFoverlap']) & np.logical_not(branches['BToKEE_l2_isPFoverlap'])
  mix_selection = ((l1_pf_selection & l2_low_selection) | (l2_pf_selection & l1_low_selection))
  low_pfveto_selection = low_selection & overlap_veto_selection
  mix_net_selection = overlap_veto_selection & np.logical_not(pf_selection | low_selection)
  all_selection = pf_selection | low_pfveto_selection | mix_net_selection 

  # count the number of b candidates passes the selection
  #count_selection = jpsi_selection 
  #nBToKEE_selected = self._branches['BToKEE_event'][count_selection].values
  #_, nBToKEE_selected = np.unique(nBToKEE_selected[np.isfinite(nBToKEE_selected)], return_counts=True)

  #mvaCut_pf = 0.89
  #mvaCut_mix = 0.93
  #mvaCut_low = 0.90
  mvaCut_pf = 2.10526315789
  mvaCut_mix = 3.05263157895
  mvaCut_low = 2.57894736842

  branches['BToKEE_keras_pf'] = -99.0
  branches['BToKEE_keras_mix'] = -99.0
  branches['BToKEE_keras_low'] = -99.0

  # add mva id to pandas dataframe

  if ele_type['pf']:
    #model_pf = load_model(args.pfmodel)
    #branches['BToKEE_keras_pf'] = model_pf.predict(branches[pf_training_branches].sort_index(axis=1).values)
    model_pf = xgb.Booster({'nthread': 4})
    model_pf.load_model(args.pfmodel)
    branches['BToKEE_keras_pf'] = model_pf.predict(xgb.DMatrix(branches[pf_training_branches].sort_index(axis=1).values))

    pf_mva_selection = pf_selection & (branches['BToKEE_keras_pf'] > mvaCut_pf)
    tree = array2tree(np.array(branches[pf_mva_selection]['BToKEE_mass'], dtype=[('BToKEE_mass', 'f4')]))
    S, SErr, B = fit(tree, 'pf') 

  if ele_type['mix_net']:
    #model_mix = load_model(args.mixmodel)
    #branches['BToKEE_keras_mix'] = model_mix.predict(branches[mix_training_branches].sort_index(axis=1).values)
    model_mix = xgb.Booster({'nthread': 4})
    model_mix.load_model(args.mixmodel)
    branches['BToKEE_keras_mix'] = model_mix.predict(xgb.DMatrix(branches[mix_training_branches].sort_index(axis=1).values))

    mix_mva_selection = mix_net_selection & (branches['BToKEE_keras_mix'] > mvaCut_mix)
    tree = array2tree(np.array(branches[mix_mva_selection]['BToKEE_mass'], dtype=[('BToKEE_mass', 'f4')]))
    S, SErr, B = fit(tree, 'mix_net') 

  if ele_type['low_pfveto']:
    #model_low = load_model(args.lowmodel)
    #branches['BToKEE_keras_low'] = model_low.predict(branches[low_training_branches].sort_index(axis=1).values)
    model_low = xgb.Booster({'nthread': 4})
    model_low.load_model(args.lowmodel)
    branches['BToKEE_keras_low'] = model_low.predict(xgb.DMatrix(branches[low_training_branches].sort_index(axis=1).values))

    low_mva_selection = low_pfveto_selection & (branches['BToKEE_keras_low'] > mvaCut_low)
    tree = array2tree(np.array(branches[low_mva_selection]['BToKEE_mass'], dtype=[('BToKEE_mass', 'f4')]))
    S, SErr, B = fit(tree, 'low_pfveto') 

  if ele_type['all']:
    all_mva_selection = pf_mva_selection | mix_mva_selection | low_mva_selection
    tree = array2tree(np.array(branches[all_mva_selection]['BToKEE_mass'], dtype=[('BToKEE_mass', 'f4')]))
    S, SErr, B = fit(tree, 'all') 

  output_branches = {}
  for eType, eBool in ele_type.items():
    if not eBool: continue
    output_branches[eType] = branches[eval(ele_selection[eType])]
    if args.hist:
      file_out = root_open('{}_histograms_{}.root'.format(outputfile, eType), 'recreate')
      hist_list = {hist_name: Hist(hist_bins['nbins'], hist_bins['xmin'], hist_bins['xmax'], name=hist_name, title='', type='F') for hist_name, hist_bins in sorted(outputbranches.items())}
      for hist_name, hist_bins in sorted(outputbranches.items()):
        if hist_name in branches.keys():
          branch_np = output_branches[eType][hist_name].values
          fill_hist(hist_list[hist_name], branch_np[np.isfinite(branch_np)])
          hist_list[hist_name].write()
      file_out.close()

    else:
      output_branches[eType][outputbranches.keys()].to_root('{}_kinematics_{}.root'.format(outputfile, eType), key='tree')
   


