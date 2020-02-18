#!/usr/bin/env python
import os,sys
from time import sleep
import math
import ROOT
from ROOT import RooFit
import numpy as np
ROOT.gROOT.ProcessLine(open('models.cc').read())
from ROOT import DoubleCBFast
from helper import *
#ROOT.gErrorIgnoreLevel=ROOT.kError
#ROOT.RooMsgService.instance().setGlobalKillBelow(RooFit.FATAL)

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


def fit(tree, outputfile, sigPDF=3, bkgPDF=2, fitJpsi=False, isMC=False, doPartial=False, partialinputfile='part_workspace.root', drawSNR=False, mvaCut=0.0, blinded=False, expS=100):
    msgservice = ROOT.RooMsgService.instance()
    msgservice.setGlobalKillBelow(RooFit.FATAL)
    wspace = ROOT.RooWorkspace('myWorkSpace')
    ROOT.gStyle.SetOptFit(0000);
    ROOT.gROOT.SetBatch(True);
    ROOT.gROOT.SetStyle("Plain");
    ROOT.gStyle.SetGridStyle(3);
    ROOT.gStyle.SetOptStat(000000);
    ROOT.gStyle.SetOptTitle(0)

    #tree = ROOT.TChain('tree')
    #tree.AddFile(inputfile)
    thevars = ROOT.RooArgSet()

    if fitJpsi:
      xmin, xmax = 2.5, 3.5
      bMass = ROOT.RooRealVar("BToKEE_mll_fullfit", "m(e^{+}e^{-})", 2.0, 4.0, "GeV")
      wspace.factory('mean[3.096916, 2.9, 3.3]')

    else:
      bMass = ROOT.RooRealVar("BToKEE_fit_mass", "m(K^{+}e^{+}e^{-})", 4.0, 6.0, "GeV")
      dieleMass = ROOT.RooRealVar("BToKEE_mll_fullfit", "m(e^{+}e^{-})", 2.0, 4.0, "GeV")
      if isMC:
        xmin, xmax = 4.5, 6.0
        wspace.factory('mean[5.272e+00, 5.22e+00, 5.3e+00]')
      else:
        xmin, xmax = FIT_LOW, FIT_UP
        #wspace.factory('mean[5.2694, 5.2694, 5.2694]')
        wspace.factory('mean[5.2681, 5.2681, 5.2681]')
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
      cut = ''

    #print cut    
    data = fulldata.reduce(thevars, cut)
    getattr(wspace,'import')(data, RooFit.Rename("data"))

    if not blinded:
      wspace.factory('nsig[5000.0, 0.0, 1000000.0]')
    else:
      wspace.factory('nsig[{0}, {0}, {0}]'.format(expS))
    wspace.factory('nbkg[10000.0, 0.0, 1000000.0]')
    wspace.factory('npartial[1000.0, 0.0, 100000.0]')

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
        # Double-sided Crystal-ball
        if isMC:
          wspace.factory('width[4.1858e-02, 1.0e-6, 5.0e-1]')
          wspace.factory('alpha1[1.0, 0.0, 10.0]')
          wspace.factory('n1[1.0, 1.0, 10.0]')
          wspace.factory('alpha2[1.0, 0.0, 10.0]')
          wspace.factory('n2[1.0, 1.0, 10.0]')
        else:
          # PF
          #wspace.factory('width[0.06187, 0.06187, 0.06187]')
          #wspace.factory('alpha1[0.667, 0.667, 0.667]')
          #wspace.factory('n1[2.39, 2.39, 2.39]')
          #wspace.factory('alpha2[2.442, 2.442, 2.442]')
          #wspace.factory('n2[3.09, 3.09, 3.09]')

          # Mix
          wspace.factory('width[0.0561, 0.0561, 0.0561]')
          wspace.factory('alpha1[0.642, 0.642, 0.642]')
          wspace.factory('n1[2.31, 2.31, 2.31]')
          wspace.factory('alpha2[1.700, 1.700, 1.700]')
          wspace.factory('n2[10.0, 10.0, 10.0]')

        wspace.factory('GenericPdf::sig("DoubleCBFast(x,mean,width,alpha1,n1,alpha2,n2)", {x,mean,width,alpha1,n1,alpha2,n2})')

    if sigPDF == 4:
        # Two Double-sided Crystal-ball
        wspace.factory('width[7.1858e-02, 1.0e-6, 5.0e-1]')
        wspace.factory('alpha1[1.0, 0.0, 10.0]')
        wspace.factory('n1[2.0, 1.0, 10.0]')
        wspace.factory('alpha2[1.0, 0.0, 10.0]')
        wspace.factory('n2[2.0, 1.0, 10.0]')
        wspace.factory('GenericPdf::cb("DoubleCBFast(x,mean,width,alpha1,n1,alpha2,n2)", {x,mean,width,alpha1,n1,alpha2,n2})')
        wspace.factory('sigma[7.1858e-03, 1.0e-6, 5.0e-1]')
        wspace.factory('Gaussian::gaus(x,mean,sigma)')
        wspace.factory('f1[0.5, 0.0, 1.0]')
        wspace.factory('SUM::sig(f1*cb, gaus)')

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
        wspace.factory('exp_alpha[-3.0, -100.0, -1.0e-5]')
        alpha = wspace.var('alpha')
        wspace.factory('Exponential::bkg(x,exp_alpha)')

    if not isMC:
      if doPartial:
        wpf = ROOT.TFile(partialinputfile,"READ")
        wp = wpf.Get("myPartialWorkSpace")
        partial = wp.pdf("partial")

        getattr(wspace, "import")(partial, RooFit.Rename("partial"))
      
        wspace.factory('SUM::model1(f1[0.5,0.0,1.0]*partial,bkg)')
        print('Finished loading KDE!')
        wspace.factory('SUM::model(nsig*sig,nbkg*model1)')
        #if not blinded:
          #wspace.factory('SUM::model(nsig*sig,nbkg*model1)')
        #else:
          #wspace.factory('ExtendPdf::model(model1,nbkg)')

      else:
        wspace.factory('SUM::model(nsig*sig,nbkg*bkg)')
    else:
      wspace.factory('ExtendPdf::model(sig,nsig)')
            
    model = wspace.pdf('model')
    bkg = wspace.pdf('model1')
    sig = wspace.pdf('sig')
    nsig = wspace.var('nsig')
    nbkg = wspace.var('nbkg')

    # define the set obs = (x)
    wspace.defineSet('obs', 'x')

    # make the set obs known to Python
    obs  = wspace.set('obs')

    theBMass.setRange("window",B_LOW,B_UP) 
    theBMass.setRange("SB1",FIT_LOW,BLIND_LOW) 
    theBMass.setRange("SB2",BLIND_UP,B_MAX) 

    ## fit the model to the data.
    print('Fitting data...')
    if not blinded:
      results = model.fitTo(data, RooFit.Extended(True), RooFit.Save(), RooFit.Range(xmin,xmax), RooFit.PrintLevel(-1))
    else:
      results = model.fitTo(data, RooFit.Extended(True), RooFit.Save(), RooFit.Range("SB1,SB2"), RooFit.PrintLevel(-1))

    results.Print()

    if not isMC:
      fracBkgRange = bkg.createIntegral(obs,obs,"window") ;
      fracBkgRangeErr = fracBkgRange.getPropagatedError(results, obs)
      nbkgWindow = nbkg.getVal() * fracBkgRange.getVal()
      #print(nbkg.getVal(), fracBkgRange.getVal())
      #print(fracBkgRange.getVal(), fracBkgRange.getPropagatedError(results, obs))
      fb = fracBkgRange.getVal()
      dfb = fracBkgRangeErr
      nb = nbkg.getVal()
      dnb = nbkg.getError()
      #print(nb*fb*np.sqrt(pow(dfb/fb,2)+pow(dnb/nb,2)))
      print("Number of signals: %f, Number of background: %f, S/sqrt(S+B): %f"%(nsig.getVal(), nbkgWindow, nsig.getVal()/np.sqrt(nsig.getVal() + nbkgWindow)))
    else:
      fracSigRange = sig.createIntegral(obs,obs,"window") ;
      print(data.sumEntries(),fracSigRange.getVal())

    # Plot results of fit on a different frame
    c2 = ROOT.TCanvas('fig_binnedFit', 'fit', 800, 600)
    c2.SetGrid()
    c2.cd()
    ROOT.gPad.SetLeftMargin(0.10)
    ROOT.gPad.SetRightMargin(0.05)

    #xframe = wspace.var('x').frame(RooFit.Title("PF electron"))
    xframe = theBMass.frame()
    nbin_data = 30 if blinded else 50
    data.plotOn(xframe, RooFit.Binning(nbin_data), RooFit.Name("data"))

    model.plotOn(xframe,RooFit.Name("global"),RooFit.Range("Full"),RooFit.LineColor(2),RooFit.MoveToBack()) # this will show fit overlay on canvas
    if not isMC:
      '''
      model.plotOn(xframe,RooFit.Name("bkg"),RooFit.Components("bkg"),RooFit.LineStyle(ROOT.kDashed),RooFit.LineColor(ROOT.kMagenta),RooFit.MoveToBack()) ;
      if doPartial:
        model.plotOn(xframe,RooFit.Name("partial"),RooFit.Components("partial"),RooFit.LineStyle(ROOT.kDashed),RooFit.LineColor(8),RooFit.MoveToBack()) ;
        #partial.plotOn(xframe,RooFit.Name("partial"),RooFit.LineStyle(ROOT.kDashed),RooFit.LineColor(ROOT.kGreen),RooFit.MoveToBack()) ;
      model.plotOn(xframe,RooFit.Name("sig"),RooFit.Components("sig"),RooFit.DrawOption("FL"),RooFit.FillColor(9),RooFit.FillStyle(3004),RooFit.LineStyle(6),RooFit.LineColor(9)) ;
      #model.plotOn(xframe,RooFit.VisualizeError(results), RooFit.FillColor(ROOT.kOrange), RooFit.MoveToBack()) # this will show fit overlay on canvas
      '''
      model.plotOn(xframe,RooFit.Name("bkg"),RooFit.Components("bkg"),RooFit.Range("Full"),RooFit.DrawOption("F"),RooFit.VLines(),RooFit.FillColor(42),RooFit.LineColor(42),RooFit.LineWidth(1),RooFit.MoveToBack())
      if doPartial:
        model.plotOn(xframe,RooFit.Name("partial"),RooFit.Components("bkg,partial"),RooFit.Range("Full"),RooFit.DrawOption("F"),RooFit.VLines(),RooFit.FillColor(40),RooFit.LineColor(40),RooFit.LineWidth(1),RooFit.MoveToBack()) ;
      model.plotOn(xframe,RooFit.Name("sig"),RooFit.Components("sig"),RooFit.Range("Full"),RooFit.DrawOption("L"),RooFit.LineStyle(2),RooFit.LineColor(1)) ;

    else:
      #if fitJpsi:
        #model.paramOn(xframe,RooFit.Layout(0.15,0.45,0.85))
      #else:
        #model.paramOn(xframe,RooFit.Layout(0.65,0.92,0.70))
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
      if drawSNR:
        pt = ROOT.TPaveText(0.72,0.38,0.92,0.50,"brNDC")
        pt.SetFillColor(0)
        pt.SetBorderSize(1)
        pt.SetTextFont(42);
        pt.SetTextSize(0.04);
        pt.SetTextAlign(12)
        pt.AddText("MVA cut: {0:.2f}".format(mvaCut))
        pt.AddText("S: {0:.0f}#pm{1:.0f}".format(nsig.getVal(),nsig.getError()))
        pt.Draw()

    else:
      legend = ROOT.TLegend(0.60,0.65,0.92,0.85);
      if drawSNR:
        pt = ROOT.TPaveText(0.72,0.38,0.92,0.63,"brNDC")
        pt.SetFillColor(0)
        pt.SetBorderSize(1)
        pt.SetTextFont(42);
        pt.SetTextSize(0.04);
        pt.SetTextAlign(12)
        pt.AddText("MVA cut: {0:.2f}".format(mvaCut))
        pt.AddText("S: {0:.0f}#pm{1:.0f}".format(nsig.getVal(),nsig.getError()))
        pt.AddText("B: {0:.0f}".format(nbkgWindow))
        pt.AddText("S/#sqrt{{S+B}}: {0:.1f}".format(nsig.getVal()/np.sqrt(nsig.getVal() + nbkgWindow)))
        pt.Draw()

    #legend = ROOT.TLegend(0.65,0.15,0.92,0.35);
    legend.SetTextFont(42);
    legend.SetTextSize(0.04);
    legend.AddEntry(xframe.findObject("data"),"Data","lpe");
    if isMC:
      legend.AddEntry(xframe.findObject("global"),"Total Fit","l");
    if not isMC:
      legend.AddEntry(xframe.findObject("bkg"),"Combinatorial","f");
      if doPartial:
        legend.AddEntry(xframe.findObject("partial"),"Partially Reco.","f");
      legend.AddEntry(xframe.findObject("sig"),"B^{+}#rightarrow K^{+} J/#psi(#rightarrow e^{+}e^{-})","l");
    legend.Draw();

    c2.cd()
    c2.Update()

    c2.SaveAs(outputfile)
    print("="*80)
    if not isMC:
      return nsig.getVal(), nsig.getError(), nbkgWindow 
    else:
      return 0.0, 0.0, 0.0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Produce histograms')
    parser.add_argument("-i", "--inputfile", dest="inputfile", default="", help="ROOT file contains histograms")
    parser.add_argument("-o", "--outputfile", dest="outputfile", default="", help="ROOT file contains histograms")
    args = parser.parse_args()

    tree = ROOT.TChain('tree')
    tree.AddFile(args.inputfile)
    fit(tree, args.outputfile, fitJpsi=False, isMC=True)
    #fit(tree, args.outputfile, doPartial=True)
    #fit(tree, args.outputfile, doPartial=True, drawSNR=True, mvaCut=2.0)


