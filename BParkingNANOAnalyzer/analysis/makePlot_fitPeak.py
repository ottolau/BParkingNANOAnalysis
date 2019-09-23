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

def CMS_lumi():
    mark = ROOT.TLatex()
    mark.SetNDC()
    lumistamp = ''
    fontScale = 1.0
    mark.SetTextAlign(11)
    mark.SetTextSize(0.042 * fontScale * 1.25)
    mark.SetTextFont(61)
    mark.DrawLatex(ROOT.gPad.GetLeftMargin(), 1 - (ROOT.gPad.GetTopMargin() - 0.017), "CMS")
    mark.SetTextSize(0.042 * fontScale)
    mark.SetTextFont(52)
    mark.DrawLatex(ROOT.gPad.GetLeftMargin() + 0.08, 1 - (ROOT.gPad.GetTopMargin() - 0.017), "Preliminary")
    mark.SetTextFont(42)
    mark.SetTextAlign(31)
    mark.DrawLatex(1 - ROOT.gPad.GetRightMargin(), 1 - (ROOT.gPad.GetTopMargin() - 0.017), lumistamp)


#-------------------------------------------------------------
# The main function can be called whatever you like. We follow
# C++ and call it main..dull, but clear!
def fit(inputfile, outputfile, hist_name, sigPDF=0, bkgPDF=0):

    #msgservice = ROOT.RooMsgService.instance()
    #msgservice.setGlobalKillBelow(RooFit.FATAL)
    wspace = ROOT.RooWorkspace('myWorkSpace')
    ROOT.gStyle.SetOptFit(0000);
    ROOT.gROOT.SetBatch(True);
    ROOT.gROOT.SetStyle("Plain");
    ROOT.gStyle.SetGridStyle(3);
    ROOT.gStyle.SetOptStat(000000);
    ROOT.gStyle.SetOptTitle(0)

    xmin = 4.5 
    xmax = 6.0 
    #xmin = 2.6
    #xmax = 3.6

    wspace.factory('x[5.0,%f,%f]' % (xmin, xmax))

    #M = 15 
    #wspace.var('x').setBins(M)

    wspace.factory('nsig[100.0, 0.0, 100000.0]')
    wspace.factory('nbkg[500.0, 0.0, 1000000.0]')

    x = wspace.var('x')
    mean = wspace.var('mean')
    sigma = wspace.var('sigma')
    nsig = wspace.var('nsig')
    nbkg = wspace.var('nbkg')

    if sigPDF == 0:
        # Voigtian
        wspace.factory('mean[5.27929e+00, 5.2e+00, 5.3e+00]')
        wspace.factory('width[1.000e-02, 1.000e-04, 1.000e-01]')
        wspace.factory('sigma[7.1858e-02, 1.e-3, 1.e-1]')
        width = wspace.var('width')
        wspace.factory('Voigtian::sig(x,mean,width,sigma)')

    else:
        # Gaussian
        wspace.factory('mean[5.2418e+00, 5.20e+00, 5.35e+00]')
        #wspace.factory('mean[3.0969+00, 3.06e+00, 3.10e+00]')
        wspace.factory('sigma[7.1858e-02, 1.e-3, 5.e-1]')
        wspace.factory('Gaussian::sig(x,mean,sigma)')

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
        wspace.factory('alpha[-1.0, -100.0, 0.0]')
        alpha = wspace.var('alpha')
        wspace.factory('Exponential::bkg(x,alpha)')


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
    l = ROOT.RooArgList(obs)

    f = ROOT.TFile(inputfile,"READ")

    hist = f.Get(hist_name)
    #hist.Rebin(2)
    hist.SetStats(0)
    hdata = ROOT.RooDataHist("hdata", "hdata", l, RooFit.Import(hist))

    #hdata = RooDataHist('hdata', 'binned data', obs)
    #hdata.add(data)  # add the data to the RooDataHist and bin them
    print("="*40)
    hdata.Print('verbose')
    print("="*40)

    # Do a multinomial fit to the binned data by
    # turning off extended likelihood mode. If you
    # want a multi-Poisson fit, change False to True.
    # (If interested, ask what all this means!)
    results = model.fitTo(hdata, RooFit.Save(), RooFit.Extended(True))
    #results = model.fitTo(hdata, RooFit.Minimizer("Minutit2","Migrad"))
    results.Print()

    #bkgRange = ROOT.RooRealVar("bkgRange","bkgRange",5.25,5.45) ;
    #bkgRangeArgSet = ROOT.RooArgSet(bkgRange)
    x.setRange("window",5.25,5.45) ;
    fracBkgRange = bkg.createIntegral(obs,obs,"window") ;

    #fracBkgRange = bkg.createIntegral(bkgRangeArgSet,"window") ;
    nbkgWindow = nbkg.getVal() * fracBkgRange.getVal()
    print(nbkg.getVal(), fracBkgRange.getVal())
    print("Number of signals: %f, Number of background: %f, S/sqrt(S+B): %f"%(nsig.getVal(), nbkgWindow, nsig.getVal()/math.sqrt(nsig.getVal() + nbkgWindow)))


    # Plot results of fit on a different frame
    c2 = ROOT.TCanvas('fig_binnedFit', 'fit', 800, 600)
    c2.SetGrid()
    c2.cd()
    ROOT.gPad.SetLeftMargin(0.10)
    ROOT.gPad.SetRightMargin(0.05)

    #xframe = wspace.var('x').frame(RooFit.Title("PF electron"))
    xframe = wspace.var('x').frame()
    hdata.plotOn(xframe, RooFit.Name("data"))
    model.plotOn(xframe,RooFit.Name("global"),RooFit.LineColor(2),RooFit.MoveToBack()) # this will show fit overlay on canvas
    #model.plotOn(xframe,RooFit.Name("bkg"),RooFit.Components("bkg"),RooFit.LineStyle(ROOT.kDashed),RooFit.LineColor(ROOT.kMagenta),RooFit.MoveToBack()) ;
    #model.plotOn(xframe,RooFit.Name("sig"),RooFit.Components("sig"),RooFit.DrawOption("FL"),RooFit.FillColor(9),RooFit.FillStyle(3004),RooFit.LineStyle(6),RooFit.LineColor(9)) ;
    #model.plotOn(xframe,RooFit.VisualizeError(results), RooFit.FillColor(ROOT.kOrange), RooFit.MoveToBack()) # this will show fit overlay on canvas

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

    CMS_lumi()

    legend = ROOT.TLegend(0.65,0.65,0.92,0.85);
    #legend = ROOT.TLegend(0.65,0.15,0.92,0.35);
    legend.SetTextFont(72);
    legend.SetTextSize(0.04);
    legend.AddEntry(xframe.findObject("data"),"Data","lpe");
    legend.AddEntry(xframe.findObject("bkg"),"Background fit","l");
    legend.AddEntry(xframe.findObject("sig"),"Signal fit","l");
    legend.AddEntry(xframe.findObject("global"),"Global Fit","l");
    #legend.Draw();

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

    fit(args.inputfile, args.outputfile, 'BToKEE_mass_pf', sigPDF=1, bkgPDF=2)
    #fit(args.inputfile, args.outputfile, 'BToKEE_mll_raw_jpsi_pf', sigPDF=1, bkgPDF=1)

