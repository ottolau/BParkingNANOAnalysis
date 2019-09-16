import os
import sys
import ROOT
from itertools import combinations, product
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


def draw_hist(histo, histo_name, x_label, y_label, norm=False, same=False):
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
            histo.Draw("HIST SAME")
    else:
        histo.SetFillColorAlpha(40,1)
        histo.SetFillStyle(4050)
        if norm:
            histo.DrawNormalized("HIST")       
        else:
            histo.Draw("HIST")


def make_plots(filename, outputFolder='Figures'):
    f = ROOT.TFile(filename)
    dir_list = ROOT.gDirectory.GetListOfKeys()
    outputfile = filename.replace('.root','') + "_distributions"
    if not os.path.exists(outputFolder):
        os.system("mkdir -p %s"%(outputFolder))

    saveHist = ['h_elePt_lead', 'h_elePt_sublead', 'h_kaonPt_lead', 'h_kaonPt_sublead']
    #notSaveHist = ['BToKEE_fit_mass_all', 'BToKEE_fit_mass_pf', 'BToKEE_fit_mass_low']
    notSaveHist = ['BToKEE_mass']

    nItems = len([key for key in dir_list if key.GetClassName() == "TH1F" and key.ReadObj().GetName() not in notSaveHist])
    nPages = 0

    for key in dir_list:
        if  key.GetClassName() != "TH1F": continue
        histo = key.ReadObj()
        histo_name = histo.GetName()
        if histo_name in notSaveHist: continue
        #if histo_name not in saveHist: continue

        canvas_name = "c_" + histo_name
        c = ROOT.TCanvas(canvas_name, canvas_name, 800, 600)
        pad = setup_pad()
        pad.Draw()
        pad.cd()

        unit = ""
        for var in varUnitMap.keys():
            if var in histo_name: unit = varUnitMap[var]
       
        draw_hist(histo, histo_name, unit, "Events")

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


def make_comparisons(signalfile, backgroundfile, outputFolder='Figures'):
    f1 = ROOT.TFile(signalfile)
    dir_list_sig = ROOT.gDirectory.GetListOfKeys()
    ROOT.gDirectory.Clear()
    f2 = ROOT.TFile(backgroundfile)
    dir_list_bkg = ROOT.gDirectory.GetListOfKeys()
    outputfile = signalfile.replace('.root','') + "_comparisons"

    nItems = sum(1 for prob in product(dir_list_sig, dir_list_bkg) if prob[0].GetClassName() == "TH1F" and prob[1].GetClassName() == "TH1F" and prob[0].ReadObj().GetName() == prob[1].ReadObj().GetName())
    nPages = 0

    for key1, key2 in product(dir_list_sig, dir_list_bkg):
        if key1.GetClassName() != "TH1F" or key2.GetClassName() != "TH1F": continue
        hist_sig = key1.ReadObj()
        hist_bkg = key2.ReadObj()
        hist_sig_name = hist_sig.GetName()
        hist_bkg_name = hist_bkg.GetName()
        if hist_sig_name != hist_bkg_name: continue

        canvas_name = "c_" + hist_sig_name
        for v in varUnitMap.keys():
            if v in hist_sig_name: var = v
        unit = varUnitMap[var]
       
        c = ROOT.TCanvas(canvas_name, canvas_name, 800, 600)
        c.cd()
        pad = setup_pad()
        pad.Draw()
        pad.cd()

        draw_hist(hist_sig, hist_sig_name, unit, "a.u.", True,  False)
        draw_hist(hist_bkg, hist_sig_name, unit, "a.u.", True,  True)

        l1 = ROOT.TLegend(0.7,0.8,0.92,0.9)
        l1.SetTextFont(72)
        l1.SetTextSize(0.04)
        l1.AddEntry(hist_sig,"Signal")
        l1.AddEntry(hist_bkg,"Background")
        l1.Draw("same")
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

    make_plots(args.inputfile)
    #make_comparisons(args.signalfile, args.backgroundfile)



