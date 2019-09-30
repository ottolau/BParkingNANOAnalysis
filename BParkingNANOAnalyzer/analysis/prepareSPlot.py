import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument("inputfile" , nargs='*', help = "Path to the input ROOT file")
parser.add_argument("-r", "--range"    , dest = "range"   , help = "Define B mass range"                                  , default = '4.7,6.0')
args = parser.parse_args()


import ROOT
from ROOT import gSystem

gSystem.Load('libRooFit')
from ROOT import RooFit, RooStats, RooRealVar, RooDataSet, RooArgList, RooTreeData, RooArgSet, RooAddPdf, RooFormulaVar, RooWorkspace, RooAbsData
from ROOT import RooGaussian, RooCBShape, RooExponential, RooChebychev, TCanvas
import sys
# sys.path.append('..')
import math
ROOT.gROOT.SetBatch(True);

lowRange  = float( args.range.split(',')[0] )
highRange = float( args.range.split(',')[1] )


B0Mass_   = 5.27958
JPsiMass_ = 3.096916
PsiPMass_ = 3.686109
KStMass_  = 0.896

B0Mass     = RooRealVar("B0Mass"    , "B0Mass"  , B0Mass_  )
#JPsiMass   = RooRealVar("JPsiMass"  , "JPsiMass", JPsiMass_)
#PsiPMass   = RooRealVar("PsiPMass"  , "PsiPMass", PsiPMass_)
#KStMass    = RooRealVar("KStMass"   , "KStMass" , KStMass_ )

nSigma_psiRej = 3.

RooAbsData.setDefaultStorageType(RooAbsData.Tree)

tree = ROOT.TChain('tree')
# tree.AddFile('jun14/small.root')
for filename in args.inputfile:
# for filename in args.inputfile.split(' '):
    print(filename)
    tree.AddFile(filename)

bMass = RooRealVar("BToKEE_mass", "m(K^{+}e^{+}e^{-})", 2, 20, "GeV")
l1_pt = RooRealVar('BToKEE_l1_pt', 'l1_pt', 0, 1000)
l2_pt = RooRealVar('BToKEE_l2_pt', 'l2_pt', 0, 1000)
k_pt = RooRealVar('BToKEE_k_pt', 'k_pt', 0, 1000)
b_pt = RooRealVar('BToKEE_pt', 'b_pt', 0, 1000)
l1_mvaId = RooRealVar('BToKEE_l1_mvaId', 'l1_mvaId', 0, 30)
l2_mvaId = RooRealVar('BToKEE_l2_mvaId', 'l2_mvaId', 0, 30)
cos2D = RooRealVar('BToKEE_cos2D', 'cos2D', 0, 1)
l_xy_sig = RooRealVar('BToKEE_l_xy_sig', 'l_xy_sig', 0, 1000)
svprob = RooRealVar('BToKEE_svprob', 'svprob', 0, 1)


thevars = RooArgSet()
thevars.add(bMass)
thevars.add(l1_pt)
thevars.add(l2_pt)
thevars.add(k_pt)
thevars.add(b_pt)
thevars.add(l1_mvaId)
thevars.add(l2_mvaId)
thevars.add(cos2D)
thevars.add(l_xy_sig)
thevars.add(svprob)

  
fulldata   = RooDataSet('fulldata', 'fulldata', tree,  RooArgSet(thevars))

theBMassfunc = RooFormulaVar("theBMass", "theBMass", "@0", RooArgList(bMass) )
theBMass     = fulldata.addColumn(theBMassfunc) ;
theBMass.setRange(lowRange,highRange);
thevars.add(theBMass)

cut = ''

print cut    
data = fulldata.reduce(thevars, cut)

## mass model 
mean        = RooRealVar ("mass"          , "mean"          ,  B0Mass_,   5,    5.5, "GeV")
sigma       = RooRealVar ("sigma"         , "sigma"         ,  9.0e-2,    1.0e-4,   1.0, "GeV")
alpha       = RooRealVar ("alpha"         , "alpha"         ,  1.0,       0.0, 1.0e+4)
n           = RooRealVar ("n"             , "n"             ,  5,         1, 100)
signalCB    = RooCBShape ("signalCB"      , "signal cb"     ,  theBMass,  mean,sigma,alpha,n)

sigma2      = RooRealVar ("sigma2"        , "sigma2"        ,  9.0e-3,    1.0e-5,   1.0, "GeV")
alpha2      = RooRealVar ("alpha2"        , "alpha2"        ,  1.0,       0.0, 1.0e+4)
n2          = RooRealVar ("n2"            , "n2"            ,  5,         1, 100)
signalCB2   = RooCBShape ("signalCB2"     , "signal cb 2"   ,  theBMass,  mean,sigma2,alpha2,n2)
f1          = RooRealVar ("f1"            , "f1"            ,  0.5  ,     0.,   1.)

CB          = RooAddPdf  ("CB"            , "CB1+CB2"       , RooArgList(signalCB,signalCB2), RooArgList(f1))

## make bkg model
exp_alpha   = RooRealVar ("exp_alpha"     , "exp_alpha"     , -1.0, -100, 0)
bkg_exp     = RooExponential("bkg_exp"    , "bkg_exp"       , theBMass, exp_alpha)

## combined model
nsig        = RooRealVar("nsig"           , "signal frac"   ,    300000,     0,    5000000)
nbkg        = RooRealVar("nbkg"           , "bkg fraction"  ,    100000,     0,    1000000)
#fitFunction = RooAddPdf ("fitFunction"    , "fit function"  ,  RooArgList(CB, bkg_exp), RooArgList(nsig, nbkg))
fitFunction = RooAddPdf ("fitfunction"    , "fit function"  ,  RooArgList(signalCB, bkg_exp), RooArgList(nsig, nbkg))

  
print 'Calculate sWeights'

## fit the model to the data.
r = fitFunction.fitTo(data, RooFit.Extended(True), RooFit.Save(), RooFit.Range(lowRange,highRange))

frame = theBMass.frame()
data.plotOn(frame, RooFit.Binning(50), RooFit.MarkerSize(.5))
fitFunction.plotOn(frame, );
fitFunction.plotOn(frame, RooFit.Components("bkg_exp"),      RooFit.LineStyle(ROOT.kDashed));
fitFunction.plotOn(frame, RooFit.Components("signalCB"),  RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kOrange+1));
fitFunction.paramOn(frame,  RooFit.Layout(0.62,0.86,0.88))
frame.SetTitle('')
frame.GetYaxis().SetTitleOffset(1.35)
frame.getAttText().SetTextSize(0.022) 
frame.getAttText().SetTextFont(42) 
frame.getAttLine().SetLineColor(0) 
c1 = ROOT.TCanvas()
frame.Draw()
c1.SaveAs('fitLMNRKstar_forSplot.pdf')


print 'sigma: ' , sigma .getVal()
print 'nsig: '  , nsig  .getVal()
print 'nbkg: '  , nbkg  .getVal()

# ROOT.RooMsgService.instance().setSilentMode(True)

## Now we use the SPlot class to add SWeights to our data set based on our model and our yield variables
sData = RooStats.SPlot("sData","An SPlot", data, fitFunction, RooArgList(nsig, nbkg) )

## Check that our weights have the desired properties
print 'Check SWeights:'
print 'Yield of B0 is ' , nsig.getVal(), '.  From sWeights it is ', sData.GetYieldFromSWeight('nsig')
print 'Yield of bkg is ', nbkg.getVal(), '.  From sWeights it is ', sData.GetYieldFromSWeight('nbkg')

outfile = ROOT.TFile("out_distribution_LMNR.root", "recreate")
outfile . cd();
#thetree = data.tree()
thetree = data.GetClonedTree()
thetree . Write();
outfile . Close();


'''
ROOT.gStyle.SetOptTitle(0)
c2 = ROOT.TCanvas('fig_binnedFit', 'fit', 800, 600)
c2.SetGrid()
c2.cd()
ROOT.gPad.SetLeftMargin(0.10)
ROOT.gPad.SetRightMargin(0.05)

dataw_sig = ROOT.RooDataSet(data.GetName(), data.GetTitle(), data, data.get(), "", "nsig_sw")
dataw_bkg = ROOT.RooDataSet(data.GetName(), data.GetTitle(), data, data.get(), "", "nbkg_sw")
elePt.setRange(0, 10)
frame = elePt.frame()
data.plotOn(frame, RooFit.Name('data'), RooFit.Binning(30), RooFit.MarkerSize(.5))
dataw_sig.plotOn(frame, RooFit.Name('signal'), ROOT.RooFit.DataError(ROOT.RooAbsData.SumW2), RooFit.Binning(30), RooFit.MarkerSize(.5), RooFit.LineColor(ROOT.kRed))
dataw_bkg.plotOn(frame, RooFit.Name('background'), ROOT.RooFit.DataError(ROOT.RooAbsData.SumW2), RooFit.Binning(30), RooFit.MarkerSize(.5), RooFit.LineColor(ROOT.kBlue))

frame.GetYaxis().SetTitleOffset(0.9)
frame.GetYaxis().SetTitleFont(42)
frame.GetYaxis().SetTitleSize(0.05)
frame.GetYaxis().SetLabelSize(0.065)
frame.GetYaxis().SetLabelSize(0.04)
frame.GetYaxis().SetLabelFont(42)
frame.GetXaxis().SetTitleOffset(0.9)
frame.GetXaxis().SetTitleFont(42)
frame.GetXaxis().SetTitleSize(0.05)
frame.GetXaxis().SetLabelSize(0.065)
frame.GetXaxis().SetLabelSize(0.04)
frame.GetXaxis().SetLabelFont(42)

frame.GetYaxis().SetTitle("Events")
frame.GetXaxis().SetTitle("p_{T}^{e} [GeV/c]")
frame.SetStats(0)
frame.SetMinimum(0)
frame.Draw()

#legend = ROOT.TLegend(0.15,0.65,0.42,0.85);
legend = ROOT.TLegend(0.65,0.65,0.92,0.85);
#legend = ROOT.TLegend(0.65,0.15,0.92,0.35);
legend.SetTextFont(42);
legend.SetTextSize(0.04);
legend.AddEntry(frame.findObject("data"),"Data","lp");
legend.AddEntry(frame.findObject("background"),"Background","lp");
legend.AddEntry(frame.findObject("signal"),"Signal","lp");
legend.Draw();

c2.cd()
c2.Update()

c2.SaveAs('test_{}.pdf'.format(eleType))

'''



