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
              "k_normpt": "kaon normalized p_{T}",
              "l1_normpt": "leading electron normalized p_{T}",
              "l2_normpt": "subleading electron normalized p_{T}",
              "BToKEE_normpt": "B^{+} normalized p_{T}",
              "mll": "m(e^{+}e^{-}) [GeV/c^{2}]",
              "mass": "m(K^{+}e^{+}e^{-}) [GeV]",
              "eta": "#eta",
              "phi": "#phi",
              "dxy_sig": "d_{xy}/#sigma_{d_{xy}}",
              "dz": "d_{z} [cm]",
              "l_xy_sig": "L_{xy} / #sigma_{L_{xy}}",
              "svprob": "P(#chi^{2}_{SV})",
              "prob": "P(#chi^{2}_{SV})",
              "cos2D": "cos #alpha_{2D}",
              "unBiased": "Low pT electron unbiased BDT",
              "ptBiased": "Low pT electron pt biased BDT",
              "mvaId": "Low pT electron performance id",
              "k_DCASig": "kaon DCA significance",
              "isLowPt": "isLowPt",
              "isPFoverlap": "isPFoverlap",
              "isPF": "isPF",
                }

def setup_pad():
    pad = ROOT.TPad("pad", "pad", 0.0, 0.0, 1.0, 1.0)
    pad.SetTopMargin(0.08)
    pad.SetBottomMargin(0.12)
    #pad.SetLeftMargin(0.11)
    pad.SetLeftMargin(0.11)
    pad.SetRightMargin(0.06)
    return pad

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
    #mark.DrawLatex(ROOT.gPad.GetLeftMargin() + 0.09, 1 - (ROOT.gPad.GetTopMargin() - 0.017), "Simulation Preliminary")
    mark.SetTextSize(extraTextSize)
    mark.SetTextFont(42)
    mark.SetTextAlign(31)
    mark.DrawLatex(1 - ROOT.gPad.GetRightMargin(), 1 - (ROOT.gPad.GetTopMargin() - 0.017), lumistamp)


def draw_hist(histo, histo_name, x_label, y_label, norm=False, same=False, err=False):
    histo.SetTitle(histo_name)
    histo.GetYaxis().SetTitle(y_label)
    histo.GetXaxis().SetTitle(x_label)
    histo.SetTitleFont(42)
    histo.SetTitleSize(0.05)
    histo.GetYaxis().SetTitleOffset(0.9)
    histo.GetYaxis().SetTitleFont(42)
    histo.GetYaxis().SetTitleSize(0.04)
    #histo.GetYaxis().SetLabelSize(0.065)
    histo.GetYaxis().SetLabelSize(0.05)
    histo.GetYaxis().SetLabelFont(42)
    histo.GetXaxis().SetTitleOffset(0.9)
    histo.GetXaxis().SetTitleFont(42)
    histo.GetXaxis().SetTitleSize(0.04)
    #histo.GetXaxis().SetLabelSize(0.065)
    histo.GetXaxis().SetLabelSize(0.05)
    histo.GetXaxis().SetLabelFont(42)
    if same:
        histo.SetFillColorAlpha(46,1)
        histo.SetLineColor(46)
        histo.SetFillStyle(3335)
        if norm:
            #histo.DrawNormalized("HIST SAME")       
            #histo.Scale(1.0/histo.Integral())
            histo.Draw("HIST SAME")

        else:
            if err:
                histo.Draw("E SAME")
            else:
                histo.Draw("HIST SAME")

    else:
        histo.SetFillColorAlpha(40,1)
        histo.SetFillStyle(4050)
        if norm:
            #histo.DrawNormalized("HIST")
            #histo.Scale(1.0/histo.Integral())
            histo.Draw("HIST")
        else:
            if err:
                histo.SetLineColor(4)
                histo.SetMarkerStyle(22)
                histo.Draw("E")
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
       
        draw_hist(histo, histo_name, unit, "Events", err=True)

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

def make_2plots(inputfile, hist_name_1, hist_name_2, outputfile):
    f1 = ROOT.TFile(inputfile)
    dir_list = ROOT.gDirectory.GetListOfKeys()
    for key in dir_list:
        if key.ReadObj().GetName() == hist_name_1: h1 = key.ReadObj()
        if key.ReadObj().GetName() == hist_name_2: h2 = key.ReadObj()

    canvas_name = "c_" + hist_name_1
    for v in varUnitMap.keys():
        if v in hist_name_1: var = v
    unit = varUnitMap[var]
   
    c = ROOT.TCanvas(canvas_name, canvas_name, 800, 600)
    c.cd()
    pad = setup_pad()
    pad.Draw()
    pad.cd()

    draw_hist(h1, hist_name_1, unit, "Events", False,  False, True)
    draw_hist(h2, hist_name_2, unit, "Events", False,  True, True)

    l1 = ROOT.TLegend(0.6,0.8,0.92,0.9)
    l1.SetTextFont(72)
    l1.SetTextSize(0.04)
    l1.AddEntry(h1,hist_name_1)
    l1.AddEntry(h2,hist_name_2)
    l1.Draw("same")
    pad.cd()
    CMS_lumi(pad)

    c.cd()
    c.Update()
    c.SaveAs(outputfile)


def make_comparisons(signalfile, backgroundfile, outputFolder='Figures'):
    f1 = ROOT.TFile(signalfile)
    dir_list_sig = ROOT.gDirectory.GetListOfKeys()
    ROOT.gDirectory.Clear()
    f2 = ROOT.TFile(backgroundfile)
    dir_list_bkg = ROOT.gDirectory.GetListOfKeys()
    outputfile = signalfile.replace('.root','') + "_comparisons"

    skipHist = ['BToKEE_l1_mvaId', 'BToKEE_l1_unBiased', 'BToKEE_l1_ptBiased', 'BToKEE_l2_mvaId', 'BToKEE_l2_unBiased', 'BToKEE_l2_ptBiased']

    nItems = sum(1 for prob in product(dir_list_sig, dir_list_bkg) if prob[0].GetClassName() == "TH1F" and prob[1].GetClassName() == "TH1F" and prob[0].ReadObj().GetName() == prob[1].ReadObj().GetName() and (prob[0].ReadObj().GetName() not in skipHist))
    nPages = 0

    for key1, key2 in product(dir_list_sig, dir_list_bkg):
        if key1.GetClassName() != "TH1F" or key2.GetClassName() != "TH1F": continue
        hist_sig = key1.ReadObj()
        hist_bkg = key2.ReadObj()
        hist_sig_name = hist_sig.GetName()
        hist_bkg_name = hist_bkg.GetName()
        if hist_sig_name != hist_bkg_name: continue
        if hist_sig_name in skipHist: continue
        print(hist_sig_name)

        canvas_name = "c_" + hist_sig_name
        for v in varUnitMap.keys():
            if v in hist_sig_name: var = v
        unit = varUnitMap[var]
       
        c = ROOT.TCanvas(canvas_name, canvas_name, 800, 600)
        c.cd()
        pad = setup_pad()
        pad.Draw()
        pad.cd()

        hist_sig.Scale(1.0/hist_sig.Integral())
        hist_bkg.Scale(1.0/hist_bkg.Integral())
        hist_sig.SetMaximum(1.25*max(hist_sig.GetMaximum(), hist_bkg.GetMaximum()))

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

def make_eleStack(inputfile, outputfile):
    eleType = ['pf', 'low_pfveto', 'mix_net', 'all']
    
    for eType in eleType:
      f1 = ROOT.TFile('{}_{}.root'.format(inputfile, eType))
      dir_list = ROOT.gDirectory.GetListOfKeys()
      for key in dir_list:
        if key.ReadObj().GetName() == 'BToKEE_mass':
          if eType == 'pf':
            h_pf = key.ReadObj()
          if eType == 'low_pfveto':
            h_low = key.ReadObj()
          if eType == 'mix_net':
            h_mix = key.ReadObj()
          if eType == 'all':
            h_all = key.ReadObj()
      ROOT.gDirectory.Clear()
    '''
    f1 = ROOT.TFile(inputfile)
    dir_list = ROOT.gDirectory.GetListOfKeys()
    for key in dir_list:
      if key.ReadObj().GetName() == 'BToKEE_mass_pf':
        h_pf = key.ReadObj()
      if key.ReadObj().GetName() == 'BToKEE_mass_low_pfveto':
        h_low = key.ReadObj()
      if key.ReadObj().GetName() == 'BToKEE_mass_mix_net':
        h_mix = key.ReadObj()
      if key.ReadObj().GetName() == 'BToKEE_mass_all':
        h_all = key.ReadObj()
    '''
    canvas_name = "c_" + h_pf.GetName()
    for v in varUnitMap.keys():
        if v in h_pf.GetName(): var = v
    unit = varUnitMap[var]
   
    c = ROOT.TCanvas(canvas_name, canvas_name, 800, 600)
    c.cd()
    pad = setup_pad()
    pad.Draw()
    pad.cd()

    
    h_pf.SetFillColorAlpha(42,1)
    h_pf.SetFillStyle(4050)
    h_low.SetFillColorAlpha(46,1)
    h_low.SetFillStyle(4050)
    h_mix.SetFillColorAlpha(40,1)
    h_mix.SetFillStyle(4050)
    '''
    h_pf.SetFillColorAlpha(0,1)
    h_pf.SetFillStyle(4050)
    h_extra = h_low.Clone()
    h_extra.Add(h_mix)
    h_extra.SetFillColorAlpha(1,1)
    h_extra.SetFillStyle(3354)
    '''
    h_all.SetFillColorAlpha(1,1)
    h_all.SetLineColor(1)
    h_all.SetFillStyle(3335)
    h_all.SetMarkerStyle(1)

    h_stack = ROOT.THStack("h_stack", "")
    h_stack.Add(h_pf)
    #h_stack.Add(h_extra)
    h_stack.Add(h_mix)
    h_stack.Add(h_low)
    h_stack.Draw("HIST")
    h_all.Draw('E2 SAME')

    h_stack.GetYaxis().SetTitle('Events')
    h_stack.GetXaxis().SetTitle(unit)
    h_stack.GetYaxis().SetTitleOffset(1.0)
    h_stack.GetYaxis().SetTitleFont(42)
    h_stack.GetYaxis().SetTitleSize(0.05)
    h_stack.GetYaxis().SetLabelSize(0.065)
    h_stack.GetYaxis().SetLabelSize(0.04)
    h_stack.GetYaxis().SetLabelFont(42)
    h_stack.GetXaxis().SetTitleOffset(0.9)
    h_stack.GetXaxis().SetTitleFont(42)
    h_stack.GetXaxis().SetTitleSize(0.05)
    h_stack.GetXaxis().SetLabelSize(0.065)
    h_stack.GetXaxis().SetLabelSize(0.04)
    h_stack.GetXaxis().SetLabelFont(42)
    #h_stack.SetMaximum(400)

    l1 = ROOT.TLegend(0.6,0.75,0.92,0.9)
    l1.SetTextFont(42)
    l1.SetTextSize(0.04)
    l1.SetLineColor(ROOT.kWhite)
    l1.SetFillStyle(0)
    l1.AddEntry(h_pf, 'PF electrons', 'f')
    #l1.AddEntry(h_extra, 'Additional electrons', 'f')
    #l1.AddEntry(h_all, 'Total', 'l')
    l1.AddEntry(h_mix, 'PF + Low pT electron', 'f')
    l1.AddEntry(h_low, 'Low pT electron', 'f')
    l1.AddEntry(h_all, 'Uncertainty', 'f')
    l1.Draw("same")
    pad.cd()
    CMS_lumi()

    c.cd()
    c.Update()
    c.SaveAs(outputfile)

def make_subtraction(inputfile, outputfile, outputFolder='Figures'):
    sub_list = ['BToKEE_l1_pt_pf', 'BToKEE_l2_pt_pf', 'BToKEE_l1_pt_low', 'BToKEE_l2_pt_low']
    f1 = ROOT.TFile(inputfile)
    dir_list = ROOT.gDirectory.GetListOfKeys()
    h_list = [key.ReadObj() for key in dir_list if ('_sb' in key.ReadObj().GetName() or '_sig' in key.ReadObj().GetName()) and key.ReadObj().GetName().replace('_sb','').replace('_sig','') in sub_list]

    nItems = len(h_list)/2
    nPages = 0

    for key, group in groupby(h_list, lambda x: x.GetName().replace('_sb','').replace('_sig','')):
      for h in group:
        if '_sig' in h.GetName():
          h_sig = h
        if '_sb' in h.GetName():
          h_sb = h

      # do the subtraction
      h_sub = h_sig.Clone()
      h_sub.Add(h_sb, -1.0)

      canvas_name = "c_" + key
      for v in varUnitMap.keys():
          if v in key: var = v
      unit = varUnitMap[var]

      c = ROOT.TCanvas(canvas_name, canvas_name, 800, 600)
      c.cd()
      pad = setup_pad()
      pad.Draw()
      pad.cd()

      draw_hist(h_sig, key, unit, "Events", err=True)
      h_sub.SetMarkerStyle(20)
      h_sub.SetLineColor(2)
      h_sb.SetMarkerStyle(21)
      h_sb.SetLineColor(3)
      h_sub.Draw('E SAME')
      h_sb.Draw('E SAME')

      l1 = ROOT.TLegend(0.6,0.7,0.92,0.9)
      l1.SetTextFont(42)
      l1.SetTextSize(0.04)
      l1.AddEntry(h_sub,'Subtracted','lp')
      l1.AddEntry(h_sig,'Signal','lp')
      l1.AddEntry(h_sb,'Sideband','lp')
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

    #make_plots(args.inputfile)
    #make_comparisons(args.signalfile, args.backgroundfile)
    #make_2plots(args.inputfile, 'BToKEE_mass_pf', 'BToKEE_fit_mass_pf', 'BToKEE_mass_comp_MC.pdf')
    make_eleStack(args.inputfile, 'test.pdf')
    #make_subtraction(args.inputfile, 'test.pdf')


