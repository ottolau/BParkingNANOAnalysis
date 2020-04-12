import numpy as np

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

params_jpsi_pf = {'mean': 5.2676, 'width': 0.06070, 'alpha1': 0.677, 'n1': 1.56, 'alpha2': 1.440, 'n2': 8.9}
params_jpsi_low = {'mean': 5.2654, 'width': 0.0638, 'alpha1': 0.655, 'n1': 1.75, 'alpha2': 1.509, 'n2': 9.85}
params_psi2s_low = {'mean': 5.25464, 'width': 0.0822, 'alpha1': 0.642, 'n1': 9.76, 'alpha2': 6.0, 'n2': 2.94}
params_psi2s_pf = {'mean': 5.2646, 'width': 0.0726, 'alpha1': 0.591, 'n1': 10.0, 'alpha2': 1.87, 'n2': 10.0}
params_jpsi_cutbased_pf = {'mean': 5.2621, 'width': 0.0658, 'alpha1': 0.945, 'n1': 1.065, 'alpha2': 1.59704, 'n2': 9.962}

def Punzi(B, a, b):
  return (b*b)/2.0 + a*np.sqrt(B) + (b/2.0)*np.sqrt(b*b + 4.0*a*np.sqrt(B) + 4.0*B)
