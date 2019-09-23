import pandas as pd
import numpy as np
from rootpy.plotting import Hist
from root_numpy import fill_hist

import argparse
parser = argparse.ArgumentParser(description="A simple ttree plotter")
parser.add_argument("-i", "--inputfile", dest="inputfile", default="DoubleMuonNtu_Run2016B.list", help="List of input ggNtuplizer files")
parser.add_argument("-o", "--outputfile", dest="outputfile", default="plots.root", help="Output file containing plots")
parser.add_argument("-s", "--hist", dest="hist", action='store_true', help="Store histograms or tree")
args = parser.parse_args()



outputbranches = {'BToKEE_mll_raw': {'nbins': 50, 'xmin': 0.0, 'xmax': 5.0},
                  'BToKEE_mll_raw_jpsi_all': {'nbins': 30, 'xmin': 2.6, 'xmax': 3.6},
                  'BToKEE_mll_raw_jpsi_pf': {'nbins': 30, 'xmin': 2.6, 'xmax': 3.6},
                  'BToKEE_mll_raw_jpsi_mix': {'nbins': 30, 'xmin': 2.6, 'xmax': 3.6},
                  'BToKEE_mll_raw_jpsi_low': {'nbins': 30, 'xmin': 2.6, 'xmax': 3.6},
                  'BToKEE_mll_raw_jpsi_mix_net': {'nbins': 30, 'xmin': 2.6, 'xmax': 3.6},
                  'BToKEE_mll_raw_jpsi_low_pfveto': {'nbins': 30, 'xmin': 2.6, 'xmax': 3.6},
                  'BToKEE_mll_fullfit_jpsi_all': {'nbins': 30, 'xmin': 2.6, 'xmax': 3.6},
                  'BToKEE_mll_fullfit_jpsi_pf': {'nbins': 30, 'xmin': 2.6, 'xmax': 3.6},
                  'BToKEE_mll_fullfit_jpsi_mix': {'nbins': 30, 'xmin': 2.6, 'xmax': 3.6},
                  'BToKEE_mll_fullfit_jpsi_low': {'nbins': 30, 'xmin': 2.6, 'xmax': 3.6},
                  'BToKEE_mll_fullfit_jpsi_mix_net': {'nbins': 30, 'xmin': 2.6, 'xmax': 3.6},
                  'BToKEE_mll_fullfit_jpsi_low_pfveto': {'nbins': 30, 'xmin': 2.6, 'xmax': 3.6},
                  'BToKEE_mass_all': {'nbins': 30, 'xmin': 4.7, 'xmax': 6.0},
                  'BToKEE_mass_pf': {'nbins': 30, 'xmin': 4.7, 'xmax': 6.0},
                  'BToKEE_mass_mix': {'nbins': 30, 'xmin': 4.7, 'xmax': 6.0},
                  'BToKEE_mass_low': {'nbins': 30, 'xmin': 4.7, 'xmax': 6.0},
                  'BToKEE_mass_mix_net': {'nbins': 30, 'xmin': 4.7, 'xmax': 6.0},
                  'BToKEE_mass_low_pfveto': {'nbins': 30, 'xmin': 4.7, 'xmax': 6.0},
                  'BToKEE_fit_mass_all': {'nbins': 30, 'xmin': 4.7, 'xmax': 6.0},
                  'BToKEE_fit_mass_pf': {'nbins': 30, 'xmin': 4.7, 'xmax': 6.0},
                  'BToKEE_fit_mass_mix': {'nbins': 30, 'xmin': 4.7, 'xmax': 6.0},
                  'BToKEE_fit_mass_low': {'nbins': 30, 'xmin': 4.7, 'xmax': 6.0},
                  'BToKEE_fit_mass_mix_net': {'nbins': 30, 'xmin': 4.7, 'xmax': 6.0},
                  'BToKEE_fit_mass_low_pfveto': {'nbins': 30, 'xmin': 4.7, 'xmax': 6.0},
                  'BToKEE_l1_pt_pf': {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0},
                  'BToKEE_l1_pt_low': {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0},
                  'BToKEE_l2_pt_pf': {'nbins': 50, 'xmin': 0.0, 'xmax': 15.0},
                  'BToKEE_l2_pt_low': {'nbins': 50, 'xmin': 0.0, 'xmax': 15.0},
                  'BToKEE_l1_pt_pf_sb': {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0},
                  'BToKEE_l1_pt_low_sb': {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0},
                  'BToKEE_l2_pt_pf_sb': {'nbins': 50, 'xmin': 0.0, 'xmax': 15.0},
                  'BToKEE_l2_pt_low_sb': {'nbins': 50, 'xmin': 0.0, 'xmax': 15.0},
                  'BToKEE_l1_pt_pf_sig': {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0},
                  'BToKEE_l1_pt_low_sig': {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0},
                  'BToKEE_l2_pt_pf_sig': {'nbins': 50, 'xmin': 0.0, 'xmax': 15.0},
                  'BToKEE_l2_pt_low_sig': {'nbins': 50, 'xmin': 0.0, 'xmax': 15.0},
                  'BToKEE_l1_mvaId_low': {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0},
                  'BToKEE_l2_mvaId_low': {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0},
                  'BToKEE_k_pt': {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0},
                  'BToKEE_pt': {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0},
                  'BToKEE_svprob': {'nbins': 50, 'xmin': 0.0, 'xmax': 1.0},
                  'BToKEE_cos2D': {'nbins': 50, 'xmin': 0.999, 'xmax': 1.0},
                  'BToKEE_l_xy_sig': {'nbins': 50, 'xmin': 0.0, 'xmax': 50.0},
                  }


branches = pd.read_hdf(args.inputfile, 'branches')

file_out = root_open(args.outputfile.replace('.root','')+'.root', 'recreate')
hist_list = {hist_name: Hist(hist_bins['nbins'], hist_bins['xmin'], hist_bins['xmax'], name=hist_name, title='', type='F') for hist_name, hist_bins in sorted(outputbranches.items())}

for hist_name, hist_bins in sorted(outputbranches.items()):
  if hist_name in branches.keys():
    branch_np = branches[hist_name].values
    fill_hist(hist_list[hist_name], branch_np[np.isfinite(branch_np)])
    hist_list[hist_name].write()
file_out.close()

