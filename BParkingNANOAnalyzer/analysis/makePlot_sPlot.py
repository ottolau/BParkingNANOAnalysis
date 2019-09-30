import os
import sys
import ROOT
from itertools import combinations, product, groupby
import numpy as np
from array import array


ROOT.gROOT.SetBatch(ROOT.kTRUE);
ROOT.gStyle.SetOptStat(0)
#ROOT.gStyle.SetOptTitle(0)

varUnitMap = {"k_pt": "kaon p_{T} [GeV]",
              "l1_pt": "leading electron p_{T} [GeV]",
              "l2_pt": "subleading electron p_{T} [GeV]",
              "BToKEE_pt": "B^{+} p_{T} [GeV]",
              "mll_raw": "m(e^{+}e^{-}) [GeV/c^{2}]",
              "mass": "m(K^{+}e^{+}e^{-}) [GeV/c^{2}]",
              "eta": "#eta",
              "phi": "#phi",
              "dR": "#Delta R",
              "D0": "d_{xy} [cm]",
              "Dz": "d_{z} [cm]",
              "SIP": "dB/#sigma_{dB}",
              "D0Error": "#sigma_{d_{xy}} [cm]",
              "DzError": "#sigma_{d_{z}} [cm]",
              "D0Sig": "d_{xy}/#sigma_{d_{xy}}",
              "DzSig": "d_{z}/#sigma_{d_{z}}",
              "normChi2": "#chi^{2}_{track}/d.o.f.",
              "svCtxy": "ct_{xy} [cm]",
              "svLxy": "L_{xy} [cm]",
              "l_xy_sig": "L_{xy} / #sigma_{L_{xy}}",
              "svChi2": "#chi^{2}_{SV}",
              "svProb": "P(#chi^{2}_{SV})",
              "svCosAngle": "cos #alpha_{2D}",
              "electron_dR": "#Delta R(e^{+}, e^{-})",
              "kaon_ee_dR": "#Delta R(K^{+}, K^{-})",
              "jpsiPhiOpen": "#Delta R(e^{+}e^{-}, #phi)",
              "jpsi_InvM": "m(e^{+}e^{-}) [GeV]",
              "phiee_InvM": "m(K^{+}K^{-}) [GeV]",
              "bsee_InvM": "m(K^{+}K^{-}e^{+}e^{-}) [GeV]",
              "q2": "q^{2} [GeV^{2}]",
              "prob": "P(#chi^{2}_{SV})",
              "cos2D": "cos #alpha_{2D}",
              "lxySig": "L_{xy} / #sigma_{L_{xy}}",
              "unBiased": "Low pT electron unbiased BDT",
                }

def setup_pad():
    pad = ROOT.TPad("pad", "pad", 0.0, 0.0, 1.0, 1.0)
    pad.SetTopMargin(0.08)
    pad.SetBottomMargin(0.12)
    pad.SetLeftMargin(0.11)
    pad.SetRightMargin(0.06)
    return pad

def CMS_lumi(pad):
    mark = ROOT.TLatex()
    mark.SetNDC()
    lumistamp = ''
    fontScale = 1.0
    mark.SetTextAlign(11)
    mark.SetTextSize(0.042 * fontScale * 1.25)
    mark.SetTextFont(61)
    mark.DrawLatex(ROOT.gPad.GetLeftMargin(), 1 - (ROOT.gPad.GetTopMargin() - 0.017), "CMS")
    pad.Update()
    mark.SetTextSize(0.042 * fontScale)
    mark.SetTextFont(52)
    mark.DrawLatex(ROOT.gPad.GetLeftMargin() + 0.08, 1 - (ROOT.gPad.GetTopMargin() - 0.017), "Preliminary")
    pad.Update()
    mark.SetTextFont(42)
    mark.SetTextAlign(31)
    mark.DrawLatex(1 - ROOT.gPad.GetRightMargin(), 1 - (ROOT.gPad.GetTopMargin() - 0.017), lumistamp)
    pad.Update()


def draw_hist(histo, histo_name, x_label, y_label, norm=False, same=False, err=False):
    histo.SetTitle(histo_name)
    histo.GetYaxis().SetTitle(y_label)
    histo.GetXaxis().SetTitle(x_label)
    histo.SetTitleFont(42)
    histo.SetTitleSize(0.05)
    histo.GetYaxis().SetTitleOffset(0.9)
    histo.GetYaxis().SetTitleFont(42)
    histo.GetYaxis().SetTitleSize(0.05)
    histo.GetYaxis().SetLabelSize(0.065)
    histo.GetYaxis().SetLabelSize(0.04)
    histo.GetYaxis().SetLabelFont(42)
    histo.GetXaxis().SetTitleOffset(0.9)
    histo.GetXaxis().SetTitleFont(42)
    histo.GetXaxis().SetTitleSize(0.05)
    histo.GetXaxis().SetLabelSize(0.065)
    histo.GetXaxis().SetLabelSize(0.04)
    histo.GetXaxis().SetLabelFont(42)
    if same:
        histo.SetFillColorAlpha(46,1)
        histo.SetLineColor(46)
        histo.SetFillStyle(3335)
        if norm:
            histo.DrawNormalized("HIST SAME")       
        else:
            if err:
                histo.Draw("E SAME")
            else:
                histo.Draw("HIST SAME")

    else:
        histo.SetFillColorAlpha(40,1)
        histo.SetFillStyle(4050)
        if norm:
            histo.DrawNormalized("HIST")       
        else:
            if err:
                histo.SetLineColor(4)
                histo.SetMarkerStyle(22)
                histo.Draw("E")
            else:
                histo.Draw("HIST")

outputbranches = {'BToKEE_l1_pt': {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0},
                  'BToKEE_l2_pt': {'nbins': 50, 'xmin': 0.0, 'xmax': 15.0},
                  'BToKEE_l1_mvaId': {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0},
                  'BToKEE_l2_mvaId': {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0},
                  'BToKEE_k_pt': {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0},
                  'BToKEE_pt': {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0},
                  'BToKEE_svprob': {'nbins': 50, 'xmin': 0.0, 'xmax': 1.0},
                  'BToKEE_cos2D': {'nbins': 50, 'xmin': 0.999, 'xmax': 1.0},
                  'BToKEE_l_xy_sig': {'nbins': 50, 'xmin': 0.0, 'xmax': 50.0},
                  }

def make_sPlots(filename, outputFolder='Figures'):
    f = ROOT.TFile(filename, 'READ')
    t = f.Get('fulldata')
    outputfile = filename.replace('.root','') + "_sPlot"
    if not os.path.exists(outputFolder):
        os.system("mkdir -p %s"%(outputFolder))
    branches = [b.GetName() for b in t.GetListOfBranches() if b.GetName() in outputbranches.keys()]
    nItems = len(branches)
    nPages = 0
    for branch in sorted(branches):
      t.Draw("{}>>{}({},{},{})".format(branch, branch, outputbranches[branch]['nbins'], outputbranches[branch]['xmin'], outputbranches[branch]['xmax']),"nsig_sw","goff")
      histo = ROOT.gDirectory.Get(branch)
      canvas_name = "c_" + branch
      c = ROOT.TCanvas(canvas_name, canvas_name, 800, 600)
      pad = setup_pad()
      pad.Draw()
      pad.cd()

      unit = ""
      for var in varUnitMap.keys():
          if var in branch: unit = varUnitMap[var]
     
      draw_hist(histo, branch, unit, "Events", err=True)

      pad.cd()
      CMS_lumi(pad)

      c.cd()
      c.Update()

      if nPages == 0:
          c.Print("{}/{}.pdf(".format(outputFolder, outputfile),"pdf")
      elif nPages == (nItems - 1):
          c.Print("{}/{}.pdf)".format(outputFolder, outputfile),"pdf")
      else:
          c.Print("{}/{}.pdf".format(outputFolder, outputfile),"pdf") 
      nPages += 1
      c.Clear()




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Produce histograms')
    parser.add_argument("-i", "--inputfile", dest="inputfile", default="", help="ROOT file contains histograms")
    parser.add_argument("-s", "--singalfile", dest="signalfile", default="", help="ROOT file contains histograms")
    parser.add_argument("-b", "--backgroundfile", dest="backgroundfile", default="", help="ROOT file contains histograms")
    args = parser.parse_args()

    make_sPlots(args.inputfile)

