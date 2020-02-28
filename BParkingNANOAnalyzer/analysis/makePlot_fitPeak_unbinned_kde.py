#!/usr/bin/env python
import os,sys
from time import sleep
import math
import ROOT
from ROOT import RooFit
import numpy as np
from makePlot_fitPeak_unbinned import CMS_lumi

ROOT.gErrorIgnoreLevel=ROOT.kError
ROOT.RooMsgService.instance().setGlobalKillBelow(RooFit.FATAL)

def fit_kde(tree, outputfile, isMC=False):

    #msgservice = ROOT.RooMsgService.instance()
    #msgservice.setGlobalKillBelow(RooFit.FATAL)
    wspace = ROOT.RooWorkspace('myPartialWorkSpace')
    ROOT.gStyle.SetOptFit(0000);
    #ROOT.gStyle.SetOptFit(1);
    ROOT.gROOT.SetBatch(True);
    ROOT.gROOT.SetStyle("Plain");
    ROOT.gStyle.SetGridStyle(3);
    ROOT.gStyle.SetOptStat(000000);
    ROOT.gStyle.SetOptTitle(0)

    xmin, xmax = 4.5, 6.0
    bMass = ROOT.RooRealVar("BToKEE_fit_mass", "m(K^{+}e^{+}e^{-})", 4.0, 6.0, "GeV")

    thevars = ROOT.RooArgSet()
    thevars.add(bMass)

    fulldata = ROOT.RooDataSet('fulldata', 'fulldata', tree, ROOT.RooArgSet(thevars))
    theBMassfunc = ROOT.RooFormulaVar("x", "x", "@0", ROOT.RooArgList(bMass) )
    theBMass     = fulldata.addColumn(theBMassfunc) ;
    theBMass.setRange(xmin,xmax);
    thevars.add(theBMass)

    cut = ''
    print cut    
    data = fulldata.reduce(thevars, cut)
    getattr(wspace,'import')(data, RooFit.Rename("data"))


    # define the set obs = (x)
    wspace.defineSet('obs', 'x')

    # make the set obs known to Python
    obs  = wspace.set('obs')
    #wspace.factory('KeysPdf::partial(x,data,MirrorBoth,2.0)')
    wspace.factory('KeysPdf::partial(x,data,MirrorLeft,2.0)')
    model = wspace.pdf('partial')

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

    CMS_lumi(isMC)

    legend = ROOT.TLegend(0.65,0.75,0.92,0.85);
    #legend = ROOT.TLegend(0.65,0.15,0.92,0.35);
    legend.SetTextFont(72);
    legend.SetTextSize(0.04);
    legend.AddEntry(xframe.findObject("data"),"Data","lpe");
    legend.AddEntry(xframe.findObject("global"),"Global Fit","l");
    legend.Draw();

    c2.cd()
    c2.Update()

    c2.SaveAs(outputfile.replace('.root','')+'.pdf')
    #wf = ROOT.TFile("part_workspace.root", "RECREATE")
    wf = ROOT.TFile(outputfile.replace('.root','')+'.root', "RECREATE")
    wspace.Write()
    wf.Close()

    print("="*80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Fit KDE')
    parser.add_argument("-i", "--inputfile", dest="inputfile", default="", help="Input file")
    parser.add_argument("-o", "--outputfile", dest="outputfile", default="", help="Output file")
    args = parser.parse_args()

    tree = ROOT.TChain('tree')
    tree.AddFile(args.inputfile)
    fit_kde(tree, args.outputfile, isMC=True)

