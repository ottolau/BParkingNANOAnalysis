import numpy as np
import uproot
import pandas as pd

ELECTRON_MASS = 0.000511
K_MASS = 0.493677
PI_MASS = 0.139570
#NR_LOW = np.sqrt(1.1)
NR_LOW = np.sqrt(0.045)
JPSI_MC = 3.08991
JPSI_SIGMA_MC = 0.04205
JPSI_LOW = np.sqrt(6.0)
#JPSI_UP = JPSI_MC + 3.0*JPSI_SIGMA_MC
JPSI_UP = 3.25
PSI2S_UP = 3.85
NR_UP = np.sqrt(20.0)
B_MC = 5.2676
B_SIGMA_MC = 0.06070
#B_LOW = B_MC - 3.0*B_SIGMA_MC
#B_UP = B_MC + 3.0*B_SIGMA_MC
B_LOW = 5.05
B_UP = 5.45
B_SB_LOW = B_MC - 6.0*B_SIGMA_MC
B_SB_UP = B_MC + 6.0*B_SIGMA_MC
BLIND_LOW = B_LOW
BLIND_UP = B_UP
B_MIN = 4.5
B_MAX = 6.0
FIT_LOW = 4.7
FIT_UP = 6.0
D_MASS_CUT = 1.9
BR_BToKJpsi = 1.01e-3
BR_JpsiToLL = 0.0597
BR_BToKLL = 4.51e-7

params_jpsitri_pf = {'mean': 5.2671, 'width': 0.0637, 'alpha1': 0.683, 'n1': 2.02, 'alpha2': 1.692, 'n2': 10.0}
params_jpsi_pf = {'mean': 5.2676, 'width': 0.06070, 'alpha1': 0.677, 'n1': 1.56, 'alpha2': 1.440, 'n2': 8.9}
params_jpsi_low = {'mean': 5.2654, 'width': 0.0638, 'alpha1': 0.655, 'n1': 1.75, 'alpha2': 1.509, 'n2': 9.85}
params_psi2stri_pf = {'mean': 5.2628, 'width': 0.0753, 'alpha1': 0.642, 'n1': 10.0, 'alpha2': 4.6, 'n2': 6.4}
params_psi2s_pf = {'mean': 5.2646, 'width': 0.0726, 'alpha1': 0.591, 'n1': 10.0, 'alpha2': 1.87, 'n2': 10.0}
params_psi2s_low = {'mean': 5.25464, 'width': 0.0822, 'alpha1': 0.642, 'n1': 9.76, 'alpha2': 6.0, 'n2': 2.94}
params_jpsi_cutbased_pf = {'mean': 5.2621, 'width': 0.0658, 'alpha1': 0.945, 'n1': 1.065, 'alpha2': 1.59704, 'n2': 9.962}

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

def get_df(root_file_name, branches=['*']):
    print('Opening file {}...'.format(root_file_name))
    f = uproot.open(root_file_name)
    if len(f.allkeys()) == 0:
        return pd.DataFrame()
    #df = uproot.open(root_file_name)["tree"].pandas.df()
    #df = pd.DataFrame(uproot.open(root_file_name)["tree"].arrays(namedecode="utf-8"))
    df = pd.DataFrame(uproot.open(root_file_name)["tree"].arrays(branches=branches))
    print('Finished opening file {}...'.format(root_file_name))
    return df

