#!/usr/bin/env python
import os
from time import sleep
import math
from collections import OrderedDict
import root_numpy
import ROOT
from ROOT import RooFit
import numpy as np
ROOT.gROOT.ProcessLine(open('models.cc').read())
from ROOT import DoubleCBFast
import sys
sys.path.append('../')
from scripts.helper import *
#ROOT.gErrorIgnoreLevel=ROOT.kError
#ROOT.RooMsgService.instance().setGlobalKillBelow(RooFit.FATAL)
import matplotlib as mpl
mpl.use('agg')
import matplotlib.font_manager
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
        'axes.grid' : True,
        'text.usetex': True,
        'figure.figsize': fig_size}
plt.rcParams.update(params)


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


def fit(tree, outputfile, **kwargs):
    sigPDF = kwargs.get('sigPDF', 0)
    bkgPDF = kwargs.get('bkgPDF', 0)
    isMC = kwargs.get('isMC', False)
    drawSNR = kwargs.get('drawSNR', False)
    mvaCut = kwargs.get('mvaCut', 0.0)
    blinded = kwargs.get('blinded', False)
    sig_low = kwargs.get('sig_low', BLIND_LOW)
    sig_up = kwargs.get('sig_up', BLIND_UP)
    fom_low = kwargs.get('fom_low', B_FOM_LOW)
    fom_up = kwargs.get('fom_up', B_FOM_UP)
    expS = kwargs.get('expS', 0)
    params = kwargs.get('params', {})
    sigName = kwargs.get('sigName', "B^{+}#rightarrow K^{+} J/#psi(#rightarrow e^{+}e^{-})")
    floatSig = kwargs.get('floatSig', False)
    prefix = kwargs.get('prefix', 'BToKEE')
    partialfit = kwargs.get('partialfit', {})
    fitmode = kwargs.get('fitmode', 'b')
    plotSigmaBand = kwargs.get('plotSigmaBand', False)

    msgservice = ROOT.RooMsgService.instance()
    msgservice.setGlobalKillBelow(RooFit.FATAL)
    wspace = ROOT.RooWorkspace('myWorkSpace')
    ROOT.gStyle.SetOptFit(0000);
    ROOT.gROOT.SetBatch(True);
    ROOT.gROOT.SetStyle("Plain");
    ROOT.gStyle.SetGridStyle(3);
    ROOT.gStyle.SetOptStat(000000);
    ROOT.gStyle.SetOptTitle(0)

    thevars = ROOT.RooArgSet()

    if fitmode == 'jpsi':
      xmin, xmax = 2.5, 3.5
      #xmin, xmax = 3.0, 4.0
      bMass = ROOT.RooRealVar("{}_mll_fullfit".format(prefix), "m(e^{+}e^{-})", 2.0, 5.0, "GeV")
      wspace.factory('mean[3.096916, 2.9, 3.8]')

    elif fitmode == 'phi':
      xmin, xmax = 0.98, 1.06
      bMass = ROOT.RooRealVar("{}_fit_phi_mass".format(prefix), "m(K^{+}K^{-})", 0.98, 1.06, "GeV")
      wspace.factory('mean[1.02, 0.98, 1.06]')

    else:
      bMass = ROOT.RooRealVar("{}_fit_mass".format(prefix), "m(K^{+}e^{+}e^{-})", 4.0, 6.0, "GeV")
      xmin, xmax = FIT_LOW, FIT_UP
      #xmin, xmax = FIT_LOW, 5.7
      wspace.factory('mean[5.272e+00, 5.22e+00, 5.5e+00]')

    thevars.add(bMass)

    fulldata = ROOT.RooDataSet('fulldata', 'fulldata', tree, ROOT.RooArgSet(thevars))
    theBMassfunc = ROOT.RooFormulaVar("x", "x", "@0", ROOT.RooArgList(bMass) )
    theBMass     = fulldata.addColumn(theBMassfunc) ;
    theBMass.setRange(xmin,xmax);
    thevars.add(theBMass)


    cut = ''

    #print cut    
    data = fulldata.reduce(thevars, cut)
    getattr(wspace,'import')(data, RooFit.Rename("data"))

    wspace.factory('nsig[5000.0, 0.0, 1000000.0]' )
    wspace.factory('nbkg[10000.0, 0.0, 1000000.0]')

    if sigPDF == 0:
        # Double-sided Crystal-ball
        wspace.factory('width[4.1858e-02, 1.0e-6, 5.0e-1]')
        wspace.factory('alpha1[1.0, 0.0, 10.0]')
        wspace.factory('n1[1.0, 1.0, 20.0]')
        wspace.factory('alpha2[1.0, 0.0, 10.0]')
        wspace.factory('n2[1.0, 1.0, 20.0]')
        wspace.factory('GenericPdf::sig("DoubleCBFast(x,mean,width,alpha1,n1,alpha2,n2)", {x,mean,width,alpha1,n1,alpha2,n2})')
        mean = wspace.var('mean')
        width = wspace.var('width')
        alpha1 = wspace.var('alpha1')
        n1 = wspace.var('n1')
        alpha2 = wspace.var('alpha2')
        n2 = wspace.var('n2')


    if sigPDF == 1:
        # Voigtian
        wspace.factory('width[1.000e-02, 1.000e-04, 1.000e-01]')
        wspace.factory('sigma[7.1858e-02, 1.e-3, 1.e-1]')
        wspace.factory('Voigtian::sig(x,mean,width,sigma)')

    if sigPDF == 2:
        # Gaussian
        wspace.factory('sigma[7.1858e-02, 1.0e-3, 5.0e-1]')
        wspace.factory('Gaussian::sig(x,mean,sigma)')

    if sigPDF == 3:
        # Crystal-ball
        wspace.factory('sigma[7.1858e-02, 1.0e-6, 5.0e-1]')
        wspace.factory('alpha[1.0, 0.0, 10.0]')
        wspace.factory('n[2, 1, 10]')
        wspace.factory('CBShape::sig(x,mean,sigma,alpha,n)')

    if sigPDF == 4:
        # Two Double-sided Crystal-ball
        wspace.factory('width[7.1858e-02, 1.0e-6, 5.0e-1]')
        wspace.factory('alpha1[1.0, 0.0, 10.0]')
        wspace.factory('n1[2.0, 1.0, 20.0]')
        wspace.factory('alpha2[1.0, 0.0, 10.0]')
        wspace.factory('n2[2.0, 1.0, 20.0]')
        wspace.factory('GenericPdf::cb("DoubleCBFast(x,mean,width,alpha1,n1,alpha2,n2)", {x,mean,width,alpha1,n1,alpha2,n2})')
        wspace.factory('sigma[7.1858e-03, 1.0e-6, 5.0e-1]')
        wspace.factory('Gaussian::gaus(x,mean,sigma)')
        wspace.factory('f1[0.5, 0.0, 1.0]')
        wspace.factory('SUM::sig(f1*cb, gaus)')

    if bkgPDF == 0:
        # Exponential
        wspace.factory('exp_alpha[-1.0, -100.0, -1.e-4]')
        exp_alpha = wspace.var('exp_alpha')
        wspace.factory('Exponential::bkg(x,exp_alpha)')

    if bkgPDF == 1:
        # Polynomial
        wspace.factory('c0[1.0, -1.0, 1.0]')
        wspace.factory('c1[-0.1, -1.0, 1.0]')
        wspace.factory('c2[-0.1, -1.0, 1.0]')
        wspace.factory('Chebychev::bkg(x,{c0,c1,c2})')

    if bkgPDF == 2:
        wspace.factory('c1[0.0, -100.0, 100.0]')
        wspace.factory('Polynomial::bkg(x,{c1})')


    if isMC:
      wspace.factory('ExtendPdf::model(sig,nsig)')
    else:
      npartial = {}
      for name, info in partialfit.items():
        wpf = ROOT.TFile(info['filename'], "READ")
        wp = wpf.Get("myPartialWorkSpace")
        partialPDF = wp.pdf(name)
        wspace.factory('n{}[{}, {}, {}]'.format(name, info['expected_yield'], info['expected_yield'] - 4.0*np.sqrt(info['expected_yield']), info['expected_yield'] + 4.0*np.sqrt(info['expected_yield'])) if 'expected_yield' in info else 'n{}[10.0, 0.0, 100000.0]'.format(name))
        getattr(wspace, "import")(partialPDF, RooFit.Rename(name))
        npartial[name] = wspace.var('n'+name)
        #if 'expected_yield' in info:
          #npartial[name].setVal(info['expected_yield'])
          #npartial[name].setConstant(True)
      wspace.factory('SUM::model(nsig*sig,nbkg*bkg{})'.format(','+','.join(['n'+name+'*'+name for name in partialfit.keys()])))

      if not floatSig:
        mean.setVal(params['mean'])
        width.setVal(params['width'])
        alpha1.setVal(params['alpha1'])
        n1.setVal(params['n1'])
        alpha2.setVal(params['alpha2'])
        n2.setVal(params['n2'])
        mean.setConstant(True)
        width.setConstant(True)
        alpha1.setConstant(True)
        n1.setConstant(True)
        alpha2.setConstant(True)
        n2.setConstant(True)
            
    model = wspace.pdf('model')
    bkg = wspace.pdf('bkg')
    sig = wspace.pdf('sig')
    nsig = wspace.var('nsig')
    if blinded:
      nsig.setVal(expS)
      #nsig.setVal(0.0)
      nsig.setConstant(True)
    nbkg = wspace.var('nbkg')
    partial_pdf = {}
    for name in partialfit.keys():
      partial_pdf[name] = wspace.pdf(name)

    # define the set obs = (x)
    wspace.defineSet('obs', 'x')

    # make the set obs known to Python
    obs  = wspace.set('obs')

    theBMass.setRange("window",sig_low,sig_up) 
    theBMass.setRange("fom_window",fom_low,fom_up) 
    theBMass.setRange("SB1",xmin,sig_low) 
    theBMass.setRange("SB2",sig_up,xmax) 

    ## fit the model to the data.
    print('Fitting data...')
    if  blinded:
      results = model.fitTo(data, RooFit.Extended(True), RooFit.Save(), RooFit.Range("SB1,SB2"), RooFit.SplitRange(True), RooFit.PrintLevel(-1))
      #nsig.setVal(expS)
    else:
      results = model.fitTo(data, RooFit.Extended(True), RooFit.Save(), RooFit.Range(xmin,xmax), RooFit.PrintLevel(-1))

    results.Print()
    fitConverged = True if results.status() == 0 else False
    #fitConverged = False

    nbin_data = 50

    if isMC:
      fracSigRange = sig.createIntegral(obs,obs,"fom_window") ;
      print(data.sumEntries(),fracSigRange.getVal())
    else:
      nsig_interested_pdf = sig.createIntegral(obs,obs,"fom_window") ;
      nsig_interested_pdf_err = nsig_interested_pdf.getPropagatedError(results, obs)
      nsig_interested = nsig.getVal() * nsig_interested_pdf.getVal()
      nsig_interested_err = nsig_interested * np.sqrt(pow(nsig.getError()/nsig.getVal(), 2) + pow(nsig_interested_pdf_err/nsig_interested_pdf.getVal(), 2)) if nsig.getVal() != 0.0 else 0.0
      nbkg_comb_pdf = bkg.createIntegral(obs,obs,"fom_window")
      nbkg_comb_pdf_err = nbkg_comb_pdf.getPropagatedError(results, obs)
      nbkg_comb = nbkg.getVal() * nbkg_comb_pdf.getVal()
      nbkg_comb_err = nbkg_comb * np.sqrt(pow(nbkg.getError()/nbkg.getVal(), 2) + pow(nbkg_comb_pdf_err/nbkg_comb_pdf.getVal(), 2)) if nbkg.getVal() != 0.0 else 0.0
      nbkg_total = nbkg_comb
      print("*"*80)
      print("MVA Cut: {}".format(mvaCut))
      if not fitConverged:
        print("*"*20 + "NOT COVERGE" + "*"*20)
      print("Number of signals: {}".format(nsig.getVal()))
      print("Number of signals in 1.4 sigma: {}, uncertainty: {}".format(nsig_interested, nsig_interested_err))
      print("Number of background - combinatorial: {}, uncertainty: {}".format(nbkg_comb, nbkg_comb_err))
      for name, nvar in npartial.items():
        nbkg_pdf_pdf = partial_pdf[name].createIntegral(obs,obs,"fom_window")
        nbkg_partial = nvar.getVal() * nbkg_pdf_pdf.getVal()
        nbkg_total += nbkg_partial
        print("Number of background - {}: {}".format(name, nbkg_partial))

      
      # Calculate 1-sigma error band of the total bkg through linear error propagation
      #xvar = np.linspace(xmin, xmax, nbin_data, endpoint=False) + ( (xmax - xmin) / (nbin_data * 2.0) )
      #xvar = np.array([x for x in xvar if ((x > sig_low) and x < sig_up)])
      #nbinx = xvar.shape[0]
      #bkgframe = ROOT.RooPlot()
      bkgframe = theBMass.frame()
      #bkgframe = xframe
      #bkgframe = xframe.emptyClone('test')
      data.plotOn(bkgframe, RooFit.Binning(nbin_data), RooFit.Name("data"))

      nd = 1.0
      nbinx = 1000
      xvar = np.linspace(fom_low, fom_up, nbinx)
      fit_params = model.getVariables()
      ordered_fit_params = ['exp_alpha', 'nbkg'] + ['n'+name for name in partialfit.keys()]
      if not blinded:
        ordered_fit_params += ['nsig',]
      full_bkg = ['bkg',] + [name for name in partialfit.keys()]
      #full_bkg = ['bkg',]
      fit_params_info = OrderedDict()
      for name in ordered_fit_params:
        fit_params_info[name] = {'mean': fit_params.find(name).getVal(), 'error': fit_params.find(name).getError()}
      model.plotOn(bkgframe,RooFit.Components(",".join(full_bkg)),RooFit.Normalization(nd, ROOT.RooAbsReal.Relative))
      model_curve = bkgframe.getCurve()
      model_cen = np.array([model_curve.interpolate(x) for x in xvar])
      #print(model_cen)
      bkgframe.remove(str(0),False)
      #results.covarianceMatrix().Print()
      #results.correlationMatrix().Print()
      covMatrix = root_numpy.matrix(results.covarianceMatrix())
      exp_event = model.expectedEvents(fit_params)
      fa = []
      for name, info in fit_params_info.items():
        adjust_norm = info['error'] if (name in (['nsig', 'nbkg',] + ['n'+p for p in partialfit.keys()])) else 0.0
        fit_params.setRealValue(name, info['mean']+info['error'])

        model.plotOn(bkgframe,RooFit.Components(",".join(full_bkg)),RooFit.Normalization(nd*(exp_event+adjust_norm), ROOT.RooAbsReal.NumEvent))
        model_curve = bkgframe.getCurve()
        fa_plus = np.array([model_curve.interpolate(x) for x in xvar])
        bkgframe.remove(str(0),False)
        fit_params.setRealValue(name, info['mean']-2.0*info['error'])

        model.plotOn(bkgframe,RooFit.Components(",".join(full_bkg)),RooFit.Normalization(nd*(exp_event-adjust_norm), ROOT.RooAbsReal.NumEvent))
        model_curve = bkgframe.getCurve()
        fa_minus = np.array([model_curve.interpolate(x) for x in xvar])
        bkgframe.remove(str(0),False)
        if name == 'nsig':
          fa.append(np.zeros(nbinx))
        else:
          fa.append((fa_plus - fa_minus) / (2.0*info['error']))
        # reset the params matrix
        fit_params.setRealValue(name, info['mean'])

      fa = np.array(fa).T
      tmp = np.array([np.asarray(np.matmul(FA, covMatrix)).flatten() for FA in fa])
      bkg_unc = np.sqrt(np.array([np.dot(t, FA) for t, FA in zip(tmp, fa)]))
      nbkg_total_err = np.sqrt(np.trapz(bkg_unc*bkg_unc, x=xvar)) / ((xmax-xmin)/nbin_data)
      #print(np.trapz(model_cen, x=xvar) / ((xmax-xmin)/nbin_data))

      fig, ax = plt.subplots()
      ax.plot(xvar, model_cen, 'b-', label=r'$N_{{\rm bkg}}={0:.1f}\pm{1:.1f}$'.format(nbkg_total, nbkg_total_err))
      ax.fill_between(xvar, model_cen-bkg_unc, model_cen+bkg_unc, facecolor='red', alpha=0.5, linewidth=0.0, label=r'$1\sigma$')
      ax.set_xlabel(r'$m(K^{+}e^{+}e^{-}) [{\rm GeV}]$')
      ax.set_ylabel(r'a.u.')
      ax.set_ylim(bottom=0)
      ax.legend(loc='upper right')
      if plotSigmaBand:
        fig.savefig(outputfile.replace('.pdf','')+'_totalbkg_1sigma.pdf', bbox_inches='tight')
      
      
      #nbkg_total_err = 1.0
      #nbkg_total_err = get_nbkg_err(xframe, model, partialfit, blinded, results, nbkg_total, plotSigmaBand, outputfile, sig_low, sig_up, xmin, xmax, nbin_data)

      SNR = nsig_interested/np.sqrt(nsig_interested + nbkg_total)

      print("Total number of background: {}, uncertainty: {}".format(nbkg_total, nbkg_total_err))
      #print("S/sqrt(S+B): {}".format(nsig.getVal()/np.sqrt(nsig.getVal() + nbkg_total)))
      print("S/sqrt(S+B): {}".format(SNR))
      print("*"*80)


    # Plot results of fit on a different frame
    c2 = ROOT.TCanvas('fig_binnedFit', 'fit', 800, 600)
    c2.SetGrid()
    c2.cd()
    ROOT.gPad.SetLeftMargin(0.10)
    ROOT.gPad.SetRightMargin(0.05)

    #xframe = wspace.var('x').frame(RooFit.Title("PF electron"))
    xframe = theBMass.frame()


    if isMC:
      data.plotOn(xframe, RooFit.Binning(nbin_data), RooFit.Name("data"))
      model.plotOn(xframe,RooFit.Name("global"),RooFit.Range("Full"),RooFit.LineColor(2),RooFit.MoveToBack()) # this will show fit overlay on canvas
      if fitmode == 'b':
        model.paramOn(xframe,RooFit.Layout(0.60,0.92,0.73))
      else:
        model.paramOn(xframe,RooFit.Layout(0.15,0.45,0.73))
      xframe.getAttText().SetTextSize(0.03)

    else:
      if blinded:
        #nd = data.reduce('(({0}_fit_mass > {1}) & ({0}_fit_mass < {2})) | (({0}_fit_mass > {3}) & ({0}_fit_mass < {4}))'.format(prefix, xmin, sig_low, sig_up, xmax)).sumEntries() / data.reduce('({0}_fit_mass > {1}) & ({0}_fit_mass < {2})'.format(prefix, xmin, xmax)).sumEntries()
        nd = 1.0
        data.plotOn(xframe, RooFit.Binning(nbin_data), RooFit.CutRange("SB1,SB2"), RooFit.Name("data"))
      else:
        nd = 1.0
        data.plotOn(xframe, RooFit.Binning(nbin_data), RooFit.Name("data"))
      model.plotOn(xframe,RooFit.Name("global"),RooFit.Range("Full"),RooFit.Normalization(nd, ROOT.RooAbsReal.Relative),RooFit.LineColor(2),RooFit.MoveToBack()) # this will show fit overlay on canvas
      model.plotOn(xframe,RooFit.Name("bkg"),RooFit.Components("bkg"),RooFit.Range("Full"),RooFit.Normalization(nd, ROOT.RooAbsReal.Relative),RooFit.DrawOption("F"),RooFit.VLines(),RooFit.FillColor(42),RooFit.LineColor(42),RooFit.LineWidth(1),RooFit.MoveToBack())
      plotted_partial = []
      for name, info in partialfit.items():
        model.plotOn(xframe,RooFit.Name(name),RooFit.Components("bkg,"+",".join(plotted_partial)+",{}".format(name)),RooFit.Range("Full"),RooFit.Normalization(nd, ROOT.RooAbsReal.Relative),RooFit.DrawOption("F"),RooFit.VLines(),RooFit.FillColor(info['color']),RooFit.LineColor(info['color']),RooFit.LineWidth(1),RooFit.MoveToBack())
        plotted_partial.append(name)
      model.plotOn(xframe,RooFit.Name("sig"),RooFit.Components("sig"),RooFit.Range("Full"),RooFit.Normalization(nd, ROOT.RooAbsReal.Relative),RooFit.DrawOption("L"),RooFit.LineStyle(2),RooFit.LineColor(1)) 
      model.paramOn(xframe,RooFit.Layout(0.15,0.45,0.83))
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

    xframe.GetYaxis().SetTitle("Events / {0:.0f} MeV".format((xmax - xmin)/nbin_data*1000.))
    xtitle = "m(K^{+}e^{+}e^{-}) [GeV]" if prefix == 'BToKEE' else "m(K^{+}K^{-}e^{+}e^{-}) [GeV]"
    if fitmode == 'jpsi':
      xtitle = "m(e^{+}e^{-}) [GeV]"
    elif fitmode == 'phi':
      xtitle = "m(K^{+}K^{-}) [GeV]"
    xframe.GetXaxis().SetTitle(xtitle)
    #xframe.GetXaxis().SetTitle("2nd principal component [GeV]" if fitJpsi else "m(K^{+}e^{+}e^{-}) [GeV]")
    xframe.SetStats(0)
    xframe.SetMinimum(0)
    xframe.Draw()

    CMS_lumi(isMC)

    if isMC:
      if fitmode == 'b':
        legend = ROOT.TLegend(0.65,0.75,0.92,0.85);
        pt = ROOT.TPaveText(0.72,0.38,0.92,0.50,"brNDC")
      else:
        legend = ROOT.TLegend(0.15,0.75,0.42,0.85);
        pt = ROOT.TPaveText(0.15,0.38,0.45,0.50,"brNDC")
      legend.AddEntry(xframe.findObject("global"),"Total Fit","l");

    else:
      legend = ROOT.TLegend(0.56,0.65,0.92,0.85) if prefix == 'BToKEE' else ROOT.TLegend(0.46,0.70,0.92,0.85)
      legend.AddEntry(xframe.findObject("bkg"),"Combinatorial","f");
      pt = ROOT.TPaveText(0.7,0.35,0.92,0.63,"brNDC")
      #pt = ROOT.TPaveText(0.72,0.30,0.92,0.63,"brNDC")
      for name, info in partialfit.items():
        legend.AddEntry(xframe.findObject(name),info['label'],"f");
      legend.AddEntry(xframe.findObject("sig"),sigName,"l");

    legend.SetTextFont(42);
    legend.SetTextSize(0.04);
    legend.AddEntry(xframe.findObject("data"),"Data","lpe");
    legend.Draw();



    if drawSNR:
      pt.SetFillColor(0)
      pt.SetBorderSize(1)
      pt.SetTextFont(42);
      pt.SetTextSize(0.04);
      pt.SetTextAlign(12)
      pt.AddText("MVA cut: {0:.2f}".format(mvaCut))
      pt.AddText("S_{{total}}: {0:.0f}#pm{1:.0f}".format(nsig.getVal(),nsig.getError()))
      pt.AddText("S: {0:.0f}#pm{1:.0f}".format(nsig_interested,nsig_interested_err))
      if not isMC:
        pt.AddText("B: {0:.0f}#pm{1:.0f}".format(nbkg_total, nbkg_total_err))
        #pt.AddText("S/#sqrt{{S+B}}: {0:.1f}".format(nsig.getVal()/np.sqrt(nsig.getVal() + nbkg_total)))
        pt.AddText("S/#sqrt{{S+B}}: {0:.1f}".format(SNR))
        #pt.AddText("Punzi: {0:.1f}".format(Punzi(nbkgWindow, 2.0, 5.0)))
      if not fitConverged:
        pt.AddText("Fit is not converged")
      pt.Draw()


    c2.cd()
    c2.Update()

    c2.SaveAs(outputfile.replace('.pdf','')+'.pdf')
    print("="*80)

    if not isMC:
      #return nsig.getVal(), nsig.getError(), nbkg_total, nbkg_total_err
      #return nsig.getVal(), nsig.getError(), nsig_interested, nsig_interested_err, nbkg_total, nbkg_total_err
      output = {}
      output['Stot'] = nsig.getVal()
      output['StotErr'] = nsig.getError()
      output['S'] = nsig_interested
      output['SErr'] = nsig_interested_err
      output['B'] = nbkg_total
      output['BErr'] = nbkg_total_err
      output['exp_alpha'] = fit_params.find('exp_alpha').getVal()
      return output
    else:
      return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


def fit_kde(tree, outputfile, isMC=True, pdfname='partial'):
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
    wspace.factory('KeysPdf::{0}(x,data,MirrorLeft,2.0)'.format(pdfname))
    model = wspace.pdf(pdfname)

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

    c2.SaveAs(outputfile.replace('.pdf','').replace('.root','')+'.pdf')
    #wf = ROOT.TFile("part_workspace.root", "RECREATE")
    wf = ROOT.TFile(outputfile.replace('.pdf','').replace('.root','')+'.root', "RECREATE")
    wspace.Write()
    wf.Close()

    print("="*80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Unbinned likelihood fit')
    parser.add_argument("-i", "--inputfile", dest="inputfile", default="", help="Input file")
    parser.add_argument("-o", "--outputfile", dest="outputfile", default="", help="Output file")
    parser.add_argument("-p", "--partial", dest="partial", action="store_true", help="Fit partially reconstructed background")
    parser.add_argument("-n", "--pdfname", dest="pdfname", default="partial", help="PDF name of the Partially reconstructed background")
    args = parser.parse_args()
    
    params = params_jpsi_pf
    partialfit = OrderedDict()
    #partialfit['partial'] = {'filename': 'part_workspace_jpsi_pf.root', 'label': 'Partially Reco.', 'color': 40}
    partialfit['partial'] = {'filename': 'part_workspace_nonresonant_lowq2_pf.root', 'label': 'Partially Reco.', 'color': 40}
    partialfit['jpsi'] = {'filename': 'jpsi_workspace_lowq2_pf.root', 'label': 'B^{+}#rightarrow K^{+} J/#psi(#rightarrow e^{+}e^{-})', 'color': 46, 'expected_yield': 150}

    tree = ROOT.TChain('tree')
    tree.AddFile(args.inputfile)
    if not args.partial:
      #fit(tree, args.outputfile, fitPhi=True, isMC=True, prefix='BToPhiEE')
      #fit(tree, args.outputfile, fitJpsi=True, isMC=True, prefix='BToPhiEE')
      #fit(tree, args.outputfile, fitJpsi=False, isMC=True, prefix='BToPhiEE')
      #fit(tree, args.outputfile, fitJpsi=False, isMC=False, prefix='BToPhiEE', params=params)
      #fit(tree, args.outputfile, params=params, partialfit=partialfit, blinded=False, drawSNR=True, plotSigmaBand=True)
      #fit(tree, args.outputfile, params=params, partialfit=partialfit, blinded=True, expS=11.53528522, drawSNR=True, plotSigmaBand=True)
      #fit(tree, args.outputfile, params=params, blinded=False, drawSNR='N/A', prefix='BToPhiEE')
      #fit(tree, args.outputfile, params=params, blinded=True, drawSNR='N/A', prefix='BToPhiEE')
      #fit(tree, args.outputfile, doPartial=True, partialinputfile='part_workspace_jpsi_low.root', drawSNR=True, mvaCut=13.58, params=params)
      #fit(tree, args.outputfile, doPartial=True, partialinputfile='part_workspace_psi2s_pf.root', doJpsi=True, jpsiinputfile='jpsi_workspace_psi2s_pf.root', drawSNR=True, mvaCut=7.0, params=params)
      fit(tree, args.outputfile, isMC=True)

    else:
      fit_kde(tree, args.outputfile, pdfname=args.pdfname)
    

