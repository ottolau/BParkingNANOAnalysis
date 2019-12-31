import pandas as pd
import numpy as np
from rootpy.io import root_open
from rootpy.plotting import Hist
from root_numpy import fill_hist, array2root
from root_pandas import to_root
import uproot

import argparse
parser = argparse.ArgumentParser(description="A simple ttree plotter")
parser.add_argument("-i", "--inputfile", dest="inputfile", default="DoubleMuonNtu_Run2016B.list", help="List of input ggNtuplizer files")
parser.add_argument("-o", "--outputfile", dest="outputfile", default="plots.root", help="Output file containing plots")
parser.add_argument("-s", "--hist", dest="hist", action='store_true', help="Store histograms or tree")
args = parser.parse_args()


outputbranches = {'BToKEE_mll_raw': {'nbins': 50, 'xmin': 2.5, 'xmax': 3.5},
                  'BToKEE_mll_fullfit': {'nbins': 50, 'xmin': 2.5, 'xmax': 3.5},
                  'BToKEE_mll_llfit': {'nbins': 50, 'xmin': 2.5, 'xmax': 3.5},
                  #'BToKEE_mass': {'nbins': 100, 'xmin': 4.5, 'xmax': 6.0},
                  'BToKEE_fit_mass': {'nbins': 50, 'xmin': 4.7, 'xmax': 6.0},
                  'BToKEE_fit_massErr': {'nbins': 50, 'xmin': 0.0, 'xmax': 0.5},
                  'BToKEE_fit_l1_pt': {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0},
                  'BToKEE_fit_l2_pt': {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0},
                  'BToKEE_fit_l1_normpt': {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0},
                  'BToKEE_fit_l2_normpt': {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0},
                  'BToKEE_fit_l1_eta': {'nbins': 50, 'xmin': -3.0, 'xmax': 3.0},
                  'BToKEE_fit_l2_eta': {'nbins': 50, 'xmin': -3.0, 'xmax': 3.0},
                  'BToKEE_fit_l1_phi': {'nbins': 50, 'xmin': -4.0, 'xmax': 4.0},
                  'BToKEE_fit_l2_phi': {'nbins': 50, 'xmin': -4.0, 'xmax': 4.0},
                  'BToKEE_l1_dxy_sig': {'nbins': 50, 'xmin': -30.0, 'xmax': 30.0},
                  'BToKEE_l2_dxy_sig': {'nbins': 50, 'xmin': -30.0, 'xmax': 30.0},
                  'BToKEE_l1_dz': {'nbins': 50, 'xmin': -1.0, 'xmax': 1.0},
                  'BToKEE_l2_dz': {'nbins': 50, 'xmin': -1.0, 'xmax': 1.0},
                  #'BToKEE_l1_unBiased': {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0},
                  #'BToKEE_l2_unBiased': {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0},
                  #'BToKEE_l1_ptBiased': {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0},
                  #'BToKEE_l2_ptBiased': {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0},
                  'BToKEE_l1_mvaId': {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0},
                  'BToKEE_l2_mvaId': {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0},
                  'BToKEE_l1_isPF': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                  'BToKEE_l2_isPF': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                  'BToKEE_l1_isLowPt': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                  'BToKEE_l2_isLowPt': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                  'BToKEE_l1_isPFoverlap': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                  'BToKEE_l2_isPFoverlap': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                  'BToKEE_fit_k_pt': {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0},
                  'BToKEE_fit_k_normpt': {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0},
                  'BToKEE_fit_k_eta': {'nbins': 50, 'xmin': -3.0, 'xmax': 3.0},
                  'BToKEE_fit_k_phi': {'nbins': 50, 'xmin': -4.0, 'xmax': 4.0},
                  'BToKEE_k_DCASig': {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0},
                  'BToKEE_fit_pt': {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0},
                  'BToKEE_fit_normpt': {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0},
                  'BToKEE_svprob': {'nbins': 50, 'xmin': 0.0, 'xmax': 1.0},
                  'BToKEE_fit_cos2D': {'nbins': 50, 'xmin': 0.999, 'xmax': 1.0},
                  'BToKEE_l_xy_sig': {'nbins': 50, 'xmin': 0.0, 'xmax': 50.0},
                  }

ELECTRON_MASS = 0.000511
K_MASS = 0.493677
JPSI_LOW = 2.9
JPSI_UP = 3.3
B_LOWSB_LOW = 4.75
B_LOWSB_UP = 5.0
B_UPSB_LOW = 5.5
B_UPSB_UP = 5.75
B_MIN = 4.7
B_MAX = 6.0


if __name__ == "__main__":
  inputfile = args.inputfile.replace('.root','').replace('.h5','')+'.root'
  outputfile = args.outputfile.replace('.root','').replace('.h5','')

  ele_type = {'all': True, 'pf': True, 'low': False, 'mix': False, 'low_pfveto': True, 'mix_net': True}
  ele_selection = {'all': 'all_selection', 'pf': 'pf_selection', 'low': 'low_selection', 'mix': 'mix_selection', 'low_pfveto': 'low_pfveto_selection', 'mix_net': 'mix_net_selection'}

  upfile = uproot.open(inputfile)
  params = upfile['tree'].arrays()
  branches = pd.DataFrame(params).sort_index(axis=1)

  output_branches = {}

  jpsi_selection = (branches['BToKEE_mll_llfit'] > JPSI_LOW) & (branches['BToKEE_mll_llfit'] < JPSI_UP)
  b_selection = jpsi_selection & (branches['BToKEE_fit_mass'] > B_LOWSB_UP) & (branches['BToKEE_fit_mass'] < B_UPSB_LOW)
  b_lowsb_selection = jpsi_selection & (branches['BToKEE_fit_mass'] > B_LOWSB_LOW) & (branches['BToKEE_fit_mass'] < B_LOWSB_UP)
  b_upsb_selection = jpsi_selection & (branches['BToKEE_fit_mass'] > B_UPSB_LOW) & (branches['BToKEE_fit_mass'] < B_UPSB_UP)
  b_sb_selection = b_lowsb_selection | b_upsb_selection

  
  #sv_selection = (branches['BToKEE_pt'] > 10.0) & (branches['BToKEE_l_xy_sig'] > 6.0 ) & (branches['BToKEE_svprob'] > 0.1) & (branches['BToKEE_cos2D'] > 0.999)
  #l1_selection = (branches['BToKEE_l1_convVeto']) & (branches['BToKEE_l1_pt'] > 1.5) & (branches['BToKEE_l1_mvaId'] > 3.0) #& (np.logical_not(branches['BToKEE_l1_isPFoverlap']))
  #l2_selection = (branches['BToKEE_l2_convVeto']) & (branches['BToKEE_l2_pt'] > 0.5) & (branches['BToKEE_l2_mvaId'] > 3.0) #& (np.logical_not(branches['BToKEE_l2_isPFoverlap']))
  #k_selection = (branches['BToKEE_k_pt'] > 1.0) #& (branches['BToKEE_k_DCASig'] > 2.0)
  #additional_selection = (branches['BToKEE_mass'] > B_LOW) & (branches['BToKEE_mass'] < B_UP)
  #general_selection = jpsi_selection & sv_selection & k_selection & (branches['BToKEE_l1_mvaId'] > 3.94) & (branches['BToKEE_l2_mvaId'] > 3.94)

  general_selection = (branches['BToKEE_l1_mvaId'] > 3.94) & (branches['BToKEE_l2_mvaId'] > 3.94)
  branches = branches[general_selection]

  # additional cuts, allows various lengths

  l1_pf_selection = (branches['BToKEE_l1_isPF'])
  l2_pf_selection = (branches['BToKEE_l2_isPF'])
  l1_low_selection = (branches['BToKEE_l1_isLowPt']) #& (branches['BToKEE_l1_pt'] < 5.0)
  l2_low_selection = (branches['BToKEE_l2_isLowPt']) #& (branches['BToKEE_l2_pt'] < 5.0)

  pf_selection = l1_pf_selection & l2_pf_selection # & (branches['BToKEE_k_pt'] > 1.5) & (branches['BToKEE_pt'] > 10.0)
  low_selection = l1_low_selection & l2_low_selection
  overlap_veto_selection = np.logical_not(branches['BToKEE_l1_isPFoverlap']) & np.logical_not(branches['BToKEE_l2_isPFoverlap'])
  mix_selection = ((l1_pf_selection & l2_low_selection) | (l2_pf_selection & l1_low_selection))
  low_pfveto_selection = low_selection & overlap_veto_selection
  mix_net_selection = overlap_veto_selection & np.logical_not(pf_selection | low_selection)
  all_selection = pf_selection | low_pfveto_selection | mix_net_selection 

  # count the number of b candidates passes the selection
  #count_selection = jpsi_selection 
  #nBToKEE_selected = branches['BToKEE_event'][count_selection].values
  #_, nBToKEE_selected = np.unique(nBToKEE_selected[np.isfinite(nBToKEE_selected)], return_counts=True)

  prepareMVA = False

  for eType, eBool in ele_type.items():
    if not eBool: continue
    if eType == 'all':
      output_branches[eType] = branches
    else:
      output_branches[eType] = branches[eval(ele_selection[eType])]

    if args.hist:
      file_out = root_open('{}_{}.root'.format(outputfile, eType), 'recreate')
      hist_list = {hist_name: Hist(hist_bins['nbins'], hist_bins['xmin'], hist_bins['xmax'], name=hist_name, title='', type='F') for hist_name, hist_bins in sorted(outputbranches.items())}
      for hist_name, hist_bins in sorted(outputbranches.items()):
        if hist_name in branches.keys():
          branch_np = output_branches[eType][hist_name].values
          fill_hist(hist_list[hist_name], branch_np[np.isfinite(branch_np)])
          hist_list[hist_name].write()
      file_out.close()

    else:
      if prepareMVA:
        output_branches[eType] = output_branches[eType].sample(frac=1)
        frac = 0.75
        training_branches = output_branches[eType].iloc[:int(frac*output_branches[eType].shape[0])]
        testing_branches = output_branches[eType].iloc[int(frac*output_branches[eType].shape[0]):]
        training_branches[outputbranches.keys()].to_root('{}_training_{}.root'.format(outputfile, eType), key='tree')
        testing_branches[outputbranches.keys()].to_root('{}_testing_{}.root'.format(outputfile, eType), key='tree')

      else:
        output_branches[eType][outputbranches.keys()].to_root('{}_{}.root'.format(outputfile, eType), key='tree')




