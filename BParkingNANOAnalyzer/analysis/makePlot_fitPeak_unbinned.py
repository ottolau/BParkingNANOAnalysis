#!/usr/bin/env python
# Load the operating system and system modules into memory
import os,sys

# Load the sleep function from the time module
from time import sleep

# Import all functions from the math module if you are sure there
# will be no name collisions
import math

# Load everything that is in PyROOT. This is fine if, again, you are
# sure there are no name collisions between PyROOT and the names
# in other modules, otherwise use the syntax:
# example:
#  from ROOT import TCanvas, TFile, kBlue
#
import ROOT
from ROOT import RooFit
import numpy as np
ROOT.gROOT.ProcessLine(open('models.cc').read())
#ROOT.gSystem.Load('models.cc')
from ROOT import DoubleCBFast

#ROOT.gErrorIgnoreLevel=ROOT.kError
#ROOT.RooMsgService.instance().setGlobalKillBelow(RooFit.FATAL)

ELECTRON_MASS = 0.000511
K_MASS = 0.493677
JPSI_MC = 3.08812
JPSI_SIGMA_MC = 0.04757
JPSI_LOW = JPSI_MC - 3.0*JPSI_SIGMA_MC
JPSI_UP = JPSI_MC + 3.0*JPSI_SIGMA_MC
B_MC = 5.25538
B_SIGMA_MC = 0.07581
B_LOW = B_MC - 3.0*B_SIGMA_MC
B_UP = B_MC + 3.0*B_SIGMA_MC
B_MIN = 4.5
B_MAX = 6.0

def CMS_lumi(isMC):
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
    mark.DrawLatex(ROOT.gPad.GetLeftMargin() + 0.09, 1 - (ROOT.gPad.GetTopMargin() - 0.017), "Simulation Preliminary" if isMC else "Preliminary")
    mark.SetTextSize(extraTextSize)
    mark.SetTextFont(42)
    mark.SetTextAlign(31)
    mark.DrawLatex(1 - ROOT.gPad.GetRightMargin(), 1 - (ROOT.gPad.GetTopMargin() - 0.017), lumistamp)


#-------------------------------------------------------------
# The main function can be called whatever you like. We follow
# C++ and call it main..dull, but clear!
def fit(inputfile, outputfile, sigPDF=0, bkgPDF=0, fitJpsi=False, isMC=False, doPartial=False, partialinputfile=None):

    #msgservice = ROOT.RooMsgService.instance()
    #msgservice.setGlobalKillBelow(RooFit.FATAL)
    wspace = ROOT.RooWorkspace('myWorkSpace')
    ROOT.gStyle.SetOptFit(0000);
    #ROOT.gStyle.SetOptFit(1);
    ROOT.gROOT.SetBatch(True);
    ROOT.gROOT.SetStyle("Plain");
    ROOT.gStyle.SetGridStyle(3);
    ROOT.gStyle.SetOptStat(000000);
    ROOT.gStyle.SetOptTitle(0)

    tree = ROOT.TChain('tree')
    tree.AddFile(inputfile)
    thevars = ROOT.RooArgSet()

    if fitJpsi:
      xmin, xmax = 2.5, 3.5
      bMass = ROOT.RooRealVar("BToKEE_mll_fullfit", "m(e^{+}e^{-})", 2.0, 4.0, "GeV")
      wspace.factory('mean[3.096916, 2.9, 3.3]')

    else:
      xmin, xmax = 4.8, 6.0
      bMass = ROOT.RooRealVar("BToKEE_fit_mass", "m(K^{+}e^{+}e^{-})", 4.0, 6.0, "GeV")
      dieleMass = ROOT.RooRealVar("BToKEE_mll_fullfit", "m(e^{+}e^{-})", 2.0, 4.0, "GeV")
      if isMC:
        wspace.factory('mean[5.27929e+00, 5.2e+00, 5.3e+00]')
      else:
        wspace.factory('mean[5.25538, 5.25538, 5.25538]')
      thevars.add(dieleMass)

    thevars.add(bMass)

    fulldata = ROOT.RooDataSet('fulldata', 'fulldata', tree, ROOT.RooArgSet(thevars))
    theBMassfunc = ROOT.RooFormulaVar("x", "x", "@0", ROOT.RooArgList(bMass) )
    theBMass     = fulldata.addColumn(theBMassfunc) ;
    theBMass.setRange(xmin,xmax);
    thevars.add(theBMass)

    if fitJpsi:
      cut = ''
    else:
      m0 = 3.08812
      si = 0.04757
      cut = '(BToKEE_mll_fullfit > {}) & (BToKEE_mll_fullfit < {})'.format(m0 - 3.0*si, m0 + 3.0*si)

    print cut    
    data = fulldata.reduce(thevars, cut)
    getattr(wspace,'import')(data, RooFit.Rename("data"))

    wspace.factory('nsig[100.0, 0.0, 1000000.0]')
    wspace.factory('nbkg[500.0, 0.0, 1000000.0]')
    wspace.factory('npartial[100.0, 0.0, 100000.0]')

    if sigPDF == 0:
        # Voigtian
        wspace.factory('width[1.000e-02, 1.000e-04, 1.000e-01]')
        wspace.factory('sigma[7.1858e-02, 1.e-3, 1.e-1]')
        wspace.factory('Voigtian::sig(x,mean,width,sigma)')

    if sigPDF == 1:
        # Gaussian
        wspace.factory('sigma[7.1858e-02, 1.0e-3, 5.0e-1]')
        wspace.factory('Gaussian::sig(x,mean,sigma)')

    if sigPDF == 2:
        # Crystal-ball
        wspace.factory('sigma[7.1858e-02, 1.0e-6, 5.0e-1]')
        wspace.factory('alpha[1.0, 0.0, 10.0]')
        wspace.factory('n[2, 1, 10]')
        wspace.factory('CBShape::sig(x,mean,sigma,alpha,n)')

    if sigPDF == 3:
        # Double Gaussian
        wspace.factory('sigma1[7.1858e-02, 1.0e-3, 5.0e-1]')
        wspace.factory('Gaussian::gaus1(x,mean,sigma1)')
        wspace.factory('sigma2[7.1858e-02, 1.0e-6, 5.0e-1]')
        wspace.factory('Gaussian::gaus2(x,mean,sigma2)')
        wspace.factory('f1[0.5, 0.0, 1.0]')
        wspace.factory('SUM::sig(f1*gaus1, gaus2)')

    if sigPDF == 4:
        # Double Crystal-ball
        wspace.factory('sigma1[7.1858e-02, 1.0e-6, 5.0e-1]')
        wspace.factory('alpha1[1.0, 0.0, 10.0]')
        wspace.factory('n1[2.0, 1, 10]')
        wspace.factory('CBShape::cb1(x,mean,sigma1,alpha1,n1)')
        wspace.factory('sigma2[7.1858e-03, 1.0e-6, 5.0e-1]')
        wspace.factory('alpha2[1.0, 0.0, 10.0]')
        wspace.factory('n2[2.0, 1, 10]')
        wspace.factory('CBShape::cb2(x,mean,sigma2,alpha2,n2)')
        wspace.factory('f1[0.5, 0.0, 1.0]')
        wspace.factory('SUM::sig(f1*cb1, cb2)')

    if sigPDF == 5:
        # Double-sided Crystal-ball
        if isMC:
          wspace.factory('width[7.1858e-02, 1.0e-6, 5.0e-1]')
          wspace.factory('alpha1[1.0, 0.0, 10.0]')
          wspace.factory('n1[2.0, 1.0, 10.0]')
          wspace.factory('alpha2[1.0, 0.0, 10.0]')
          wspace.factory('n2[2.0, 1.0, 10.0]')
        else:
          wspace.factory('width[0.07581, 0.07581, 0.07581]')
          wspace.factory('alpha1[2.32, 2.32, 2.32]')
          wspace.factory('n1[2.69, 2.69, 2.69]')
          wspace.factory('alpha2[2.49, 2.49, 2.49]')
          wspace.factory('n2[2.41, 2.41, 2.41]')

        wspace.factory('GenericPdf::sig("DoubleCBFast(x,mean,width,alpha1,n1,alpha2,n2)", {x,mean,width,alpha1,n1,alpha2,n2})')

    if bkgPDF == 0:
        # Polynomial
        wspace.factory('c0[1.0, -1.0, 1.0]')
        wspace.factory('c1[-0.1, -1.0, 1.0]')
        wspace.factory('c2[-0.1, -1.0, 1.0]')
        wspace.factory('Chebychev::bkg(x,{c0,c1,c2})')

    if bkgPDF == 1:
        wspace.factory('c1[0.0, -100.0, 100.0]')
        wspace.factory('Polynomial::bkg(x,{c1})')

    if bkgPDF == 2:
        # Exponential
        wspace.factory('exp_alpha[-1.0, -100.0, -1.0e-5]')
        alpha = wspace.var('alpha')
        wspace.factory('Exponential::bkg(x,exp_alpha)')

    if not isMC:
      if doPartial:
        #ws = ROOT.RooWorkspace('myWS')
        '''
        partialMass = ROOT.RooRealVar("BToKEE_fit_mass", "m(K^{+}e^{+}e^{-})", 4.0, 6.0, "GeV")
        partialtree = ROOT.TChain('tree')
        partialtree.AddFile(partialinputfile)
        partialthevars = ROOT.RooArgSet()
        partialthevars.add(partialMass)

        partialfulldata = ROOT.RooDataSet('partialfulldata', 'partialfulldata', partialtree, ROOT.RooArgSet(partialthevars))
        thePartialfunc = ROOT.RooFormulaVar("y", "y", "@0", ROOT.RooArgList(partialMass) )
        thePartialMass = partialfulldata.addColumn(thePartialfunc) ;
        thePartialMass.setRange(xmin,xmax);
        partialthevars.add(thePartialMass)

        partialdata = partialfulldata.reduce(partialthevars, '')
        getattr(ws,'import')(partialdata, RooFit.Rename("partialdata"))
        #wspace.factory('SUM::model1(nsig*sig,nbkg*bkg)')
        print('Loading KDE...')
        ws.factory('KeysPdf::partial(y,partialdata,MirrorBoth,2)')
        #partial = ws.pdf('partial')
        wf = ROOT.TFile("part_workspace.root", "RECREATE")
        ws.Write()
        wf.Close()
        '''
        wpf = ROOT.TFile( "part_workspace.root","READ")
        wp = wpf.Get("myPartialWorkSpace")
        partial = wp.pdf("partial")

        getattr(wspace, "import")(partial, RooFit.Rename("partial"))
      
        wspace.factory('SUM::model1(f1[0.5,0.0,1.0]*partial,bkg)')
        #wspace.factory('AddPdf::partial(partialtest, {npartial})')
        #wspace.factory('Pdf::partial(partialkeys,npartial)')
        print('Finished loading KDE!')
        #wspace.factory('SUM::model(npartial*partial,model1)')
        wspace.factory('SUM::model(nsig*sig,nbkg*model1)')
        #wspace.factory('SUM::model(nsig*sig,nbkg*bkg)')

        wspace.Print()

      else:
        wspace.factory('SUM::model(nsig*sig,nbkg*bkg)')
            
    model = wspace.pdf('sig' if isMC else 'model')
    bkg = wspace.pdf('bkg')
    sig = wspace.pdf('sig')
    nsig = wspace.var('nsig')
    nbkg = wspace.var('nbkg')

    # define the set obs = (x)
    wspace.defineSet('obs', 'x')

    # make the set obs known to Python
    obs  = wspace.set('obs')

    ## fit the model to the data.
    print('Fitting data...')
    results = model.fitTo(data, RooFit.Extended(True), RooFit.Save(), RooFit.Range(xmin,xmax), RooFit.PrintLevel(-1))
    results.Print()

    theBMass.setRange("window",B_LOW,B_UP) ;
    fracBkgRange = bkg.createIntegral(obs,obs,"window") ;
    nbkgWindow = nbkg.getVal() * fracBkgRange.getVal()
    print("Number of signals: %f, Number of background: %f, S/sqrt(S+B): %f"%(nsig.getVal(), nbkgWindow, nsig.getVal()/np.sqrt(nsig.getVal() + nbkgWindow)))

    # Plot results of fit on a different frame
    c2 = ROOT.TCanvas('fig_binnedFit', 'fit', 800, 600)
    c2.SetGrid()
    c2.cd()
    ROOT.gPad.SetLeftMargin(0.10)
    ROOT.gPad.SetRightMargin(0.05)

    #xframe = wspace.var('x').frame(RooFit.Title("PF electron"))
    xframe = theBMass.frame()
    #xframe = thePartialMass.frame()

    data.plotOn(xframe, RooFit.Binning(50), RooFit.Name("data"))
    model.plotOn(xframe,RooFit.Name("global"),RooFit.LineColor(2),RooFit.MoveToBack()) # this will show fit overlay on canvas
    if not isMC:
      model.plotOn(xframe,RooFit.Name("bkg"),RooFit.Components("bkg"),RooFit.LineStyle(ROOT.kDashed),RooFit.LineColor(ROOT.kMagenta),RooFit.MoveToBack()) ;
      if doPartial:
        model.plotOn(xframe,RooFit.Name("partial"),RooFit.Components("partial"),RooFit.LineStyle(ROOT.kDashed),RooFit.LineColor(8),RooFit.MoveToBack()) ;
        #partial.plotOn(xframe,RooFit.Name("partial"),RooFit.LineStyle(ROOT.kDashed),RooFit.LineColor(ROOT.kGreen),RooFit.MoveToBack()) ;
      model.plotOn(xframe,RooFit.Name("sig"),RooFit.Components("sig"),RooFit.DrawOption("FL"),RooFit.FillColor(9),RooFit.FillStyle(3004),RooFit.LineStyle(6),RooFit.LineColor(9)) ;
      #model.plotOn(xframe,RooFit.VisualizeError(results), RooFit.FillColor(ROOT.kOrange), RooFit.MoveToBack()) # this will show fit overlay on canvas
    else:
      model.paramOn(xframe,RooFit.Layout(0.15,0.45,0.85))
      xframe.getAttText().SetTextSize(0.03)

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
    if fitJpsi:
      xframe.GetXaxis().SetTitle("m(e^{+}e^{-}) [GeV]")
    else:
      xframe.GetXaxis().SetTitle("m(K^{+}e^{+}e^{-}) [GeV]")
    xframe.SetStats(0)
    xframe.SetMinimum(0)
    xframe.Draw()

    CMS_lumi(isMC)

    if isMC:
      legend = ROOT.TLegend(0.65,0.75,0.92,0.85);
    else:
      legend = ROOT.TLegend(0.65,0.65,0.92,0.85);

    #legend = ROOT.TLegend(0.65,0.15,0.92,0.35);
    legend.SetTextFont(42);
    legend.SetTextSize(0.04);
    legend.AddEntry(xframe.findObject("data"),"Data","lpe");
    if isMC:
      legend.AddEntry(xframe.findObject("global"),"Total","l");
    if not isMC:
      legend.AddEntry(xframe.findObject("bkg"),"Combinatorial","l");
      if doPartial:
        legend.AddEntry(xframe.findObject("partial"),"Partially Reco.","l");
      legend.AddEntry(xframe.findObject("sig"),"Signal","l");
    legend.Draw();

    c2.cd()
    c2.Update()

    c2.SaveAs(outputfile)

    print("="*80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Produce histograms')
    parser.add_argument("-i", "--inputfile", dest="inputfile", default="", help="ROOT file contains histograms")
    parser.add_argument("-o", "--outputfile", dest="outputfile", default="", help="ROOT file contains histograms")
    args = parser.parse_args()

    #fit(args.inputfile, args.outputfile, sigPDF=5, bkgPDF=2, fitJpsi=True, isMC=True)
    fit(args.inputfile, args.outputfile, sigPDF=5, bkgPDF=2, doPartial=True, partialinputfile='RootTree_2019Oct28_BdToKstarJpsi_ToKPiee_BToKEEAnalyzer_noCut_pf.root')
    #fit(args.inputfile, args.outputfile, sigPDF=5, bkgPDF=2, doPartial=False)


