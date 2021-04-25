import numpy as np
import uproot
import pandas as pd
#import ROOT

ELECTRON_MASS = 0.000511
K_MASS = 0.493677
PI_MASS = 0.139570
#NR_LOW = np.sqrt(1.1)
LOWQ2_LOW = np.sqrt(0.045)
LOWQ2_UP = 2.5
JPSI_MC = 3.08991
JPSI_SIGMA_MC = 0.04205
JPSI_LOW = 2.8
#JPSI_UP = JPSI_MC + 3.0*JPSI_SIGMA_MC
JPSI_UP = 3.25
PSI2S_LOW = 3.4
PSI2S_UP = 3.85
HIGHQ2_UP = np.sqrt(25.0)
PHI_SIGMA_MC = 0.0026836
PHI_LOW = 1.01957 - 4.0*PHI_SIGMA_MC
PHI_UP = 1.01957 + 4.0*PHI_SIGMA_MC
B_MC = 5.2676
B_SIGMA_MC = 0.06070
#B_LOW = B_MC - 3.0*B_SIGMA_MC
#B_UP = B_MC + 3.0*B_SIGMA_MC
B_LOW = 5.05
B_UP = 5.45
BS_LOW = 5.15
BS_UP = 5.55
B_SB_LOW = B_MC - 6.0*B_SIGMA_MC
B_SB_UP = B_MC + 6.0*B_SIGMA_MC
BLIND_LOW = B_LOW
BLIND_UP = B_UP
B_FOM_LOW = 5.05
B_FOM_UP = 5.45
#B_FOM_LOW = 5.183
#B_FOM_UP = 5.353
B_MIN = 4.7
B_MAX = 6.0
FIT_LOW = 4.7
FIT_UP = 6.0
D_MASS_CUT = 1.9
BR_BToKJpsi = 1.01e-3
BR_JpsiToLL = 0.0597
BR_BToKLL = 4.51e-7
BR_BToPhiJpsi = 1.08e-3
BR_PhiToKK = 0.492
BR_BToPhiLL = 8.2e-7

params_lowq2_pf = {'mean': 5.27116, 'width': 0.05432, 'alpha1': 0.662, 'n1': 3.05, 'alpha2': 2.71, 'n2': 2.71, 'mean_gaus': 5.249, 'sigma': 0.223, 'f1': 0.928}
params_lowq2_mix = {'mean': 5.268012, 'width': 0.057381, 'alpha1': 0.59669, 'n1': 1.8979, 'alpha2': 1.3315, 'n2': 1.7794, 'mean_gaus': 5.18, 'sigma': 0.26, 'f1': 1.0}
params_jpsi_pf = {'mean': 5.2761, 'width': 0.0613, 'alpha1': 1.255, 'n1': 8.6, 'alpha2': 1.284, 'n2': 27.0, 'mean_gaus': 5.159, 'sigma': 0.1220, 'f1': 0.678}
params_jpsi_mix = {'mean': 5.2733, 'width': 0.0608, 'alpha1': 1.17, 'n1': 9.1, 'alpha2': 1.239, 'n2': 18.2, 'mean_gaus': 5.151, 'sigma': 0.1249, 'f1': 0.626}
params_psi2s_pf = {'mean': 5.2703, 'width': 0.07803, 'alpha1': 0.912, 'n1': 30.0, 'alpha2': 3.13, 'n2': 2.6, 'mean_gaus': 5.013, 'sigma': 0.1103, 'f1': 0.891}
params_psi2s_mix = {'mean': 5.26297, 'width': 0.08076, 'alpha1': 0.785, 'n1': 30.0, 'alpha2': 2.457, 'n2': 8.1, 'mean_gaus': 4.9803, 'sigma': 0.1036, 'f1': 0.885}
params_highq2_pf = {'mean': 5.27876, 'width': 0.0703, 'alpha1': 0.716, 'n1': 30.0, 'alpha2': 1.108, 'n2': 2.05, 'mean_gaus': 5.231, 'sigma': 0.270, 'f1': 0.906}
params_highq2_mix = {'mean': 5.275092, 'width': 0.0716, 'alpha1': 0.729, 'n1': 6.0, 'alpha2': 1.157, 'n2': 1.815, 'mean_gaus': 5.182, 'sigma': 0.217, 'f1': 0.900}

params_rphi_jpsi_pf = {'mean': 5.35897, 'width': 0.05558, 'alpha1': 0.5363, 'n1': 2.76, 'alpha2': 1.269, 'n2': 20}
params_rphi_jpsi_low = {'mean': 5.3546, 'width': 0.0609, 'alpha1': 0.560, 'n1': 2.98, 'alpha2': 1.400, 'n2': 20}

triCut_jpsi_mll_mean_pf = 3.00233244896
triCut_jpsi_mKee_mean_pf = 5.17609024048
triCut_jpsi_rotMatrix_pf = np.array([[0.74743241, -0.66433786], [0.66433786, 0.74743241]])
triCut_jpsi_lower_bound_pf = -0.0977517530843
triCut_jpsi_upper_bound_pf = 0.0824238599635

triCut_psi2s_mll_mean_pf = 3.58506560326
triCut_psi2s_mKee_mean_pf = 5.17640161514
triCut_psi2s_rotMatrix_pf = np.array([[0.72047561, -0.69348028], [0.69348028, 0.72047561]])
triCut_psi2s_lower_bound_pf = -0.0704979038994
triCut_psi2s_upper_bound_pf = 0.059937465045

def Punzi(B, a, b):
  return (b*b)/2.0 + a*np.sqrt(B) + (b/2.0)*np.sqrt(b*b + 4.0*a*np.sqrt(B) + 4.0*B)

def Punzi_simplify(eff, B, dB, a):
  return eff / ((a/2.0) + np.sqrt(B + dB*dB))

def Significance(S, B, dB):
  return S / np.sqrt(S + B + dB*dB)

def get_df(root_file_name, tree='tree', branches=['*']):
    print('Opening file {}...'.format(root_file_name))
    f = uproot.open(root_file_name)
    if len(f.allkeys()) == 0:
        return pd.DataFrame()
    print('Not an null file')
    #df = uproot.open(root_file_name)["tree"].pandas.df()
    #df = pd.DataFrame(uproot.open(root_file_name)["tree"].arrays(namedecode="utf-8"))
    df = pd.DataFrame(f[tree].arrays(branches=branches))
    print('Finished opening file {}...'.format(root_file_name))
    return df

def get_diagonalCut_var(branches, mll_mean, fit_mass_mean, diagCut_lower_bound, diagCut_jpsi_upper_bound, eigVecs):
  branches['BToKEE_mll_fullfit_centered'] = branches['BToKEE_mll_fullfit'] - mll_mean
  branches['BToKEE_fit_mass_centered'] = branches['BToKEE_fit_mass'] - fit_mass_mean
  data_centered = np.array([branches['BToKEE_fit_mass_centered'],branches['BToKEE_mll_fullfit_centered']]).T
  eigVecs_jpsi = triCut_jpsi_rotMatrix_pf 
  data_decorr = data_centered.dot(eigVecs)
  return data_decorr[:,0], data_decorr[:,1]

'''
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
'''
