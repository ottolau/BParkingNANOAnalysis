#! /usr/bin/env python
import uproot
import uproot_methods
import pandas as pd
import numpy as np
import time
from helper import *
import xgboost as xgb
from BParkingNANOAnalysis.BParkingNANOAnalyzer.BaseAnalyzer import BParkingNANOAnalyzer

class BToKLLAnalyzer_postprocess(BParkingNANOAnalyzer):
  def __init__(self, inputfiles, outputfile, hist=False, evalMVA=False):
    self._evalMVA = evalMVA
    inputbranches= ['*',]
    outputbranches = {'BToKEE_mll_fullfit': {'nbins': 50, 'xmin': 2.6, 'xmax': 3.3},
                      'BToKEE_q2': {'nbins': 100, 'xmin': 0.0, 'xmax': 25.0},
                      'BToKEE_fit_mass': {'nbins': 100, 'xmin': 4.5, 'xmax': 6.0},
                      'BToKEE_fit_massErr': {'nbins': 100, 'xmin': 0.0, 'xmax': 0.5},
                      'BToKEE_fit_pt': {'nbins': 100, 'xmin': 0.0, 'xmax': 30.0},
                      'BToKEE_fit_normpt': {'nbins': 100, 'xmin': 0.0, 'xmax': 30.0},
                      'BToKEE_fit_eta': {'nbins': 100, 'xmin': -3.0, 'xmax': 3.0},
                      'BToKEE_fit_phi': {'nbins': 100, 'xmin': -4.0, 'xmax': 4.0},
                      'BToKEE_b_iso04_rel': {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0},
                      'BToKEE_fit_l1_pt': {'nbins': 100, 'xmin': 0.0, 'xmax': 30.0},
                      'BToKEE_fit_l2_pt': {'nbins': 100, 'xmin': 0.0, 'xmax': 30.0},
                      'BToKEE_fit_l1_normpt': {'nbins': 100, 'xmin': 0.0, 'xmax': 30.0},
                      'BToKEE_fit_l2_normpt': {'nbins': 100, 'xmin': 0.0, 'xmax': 30.0},
                      'BToKEE_fit_l1_eta': {'nbins': 100, 'xmin': -3.0, 'xmax': 3.0},
                      'BToKEE_fit_l2_eta': {'nbins': 100, 'xmin': -3.0, 'xmax': 3.0},
                      'BToKEE_fit_l1_phi': {'nbins': 100, 'xmin': -4.0, 'xmax': 4.0},
                      'BToKEE_fit_l2_phi': {'nbins': 100, 'xmin': -4.0, 'xmax': 4.0},
                      'BToKEE_l1_dxy_sig': {'nbins': 100, 'xmin': -30.0, 'xmax': 30.0},
                      'BToKEE_l2_dxy_sig': {'nbins': 100, 'xmin': -30.0, 'xmax': 30.0},
                      'BToKEE_l1_mvaId': {'nbins': 100, 'xmin': -2.0, 'xmax': 10.0},
                      'BToKEE_l2_mvaId': {'nbins': 100, 'xmin': -2.0, 'xmax': 10.0},
                      'BToKEE_l1_pfmvaId': {'nbins': 100, 'xmin': -10.0, 'xmax': 10.0},
                      'BToKEE_l2_pfmvaId': {'nbins': 100, 'xmin': -10.0, 'xmax': 10.0},
                      'BToKEE_l1_pfmvaCats': {'nbins': 2, 'xmin': 0.0, 'xmax': 2.0},
                      'BToKEE_l2_pfmvaCats': {'nbins': 2, 'xmin': 0.0, 'xmax': 2.0},
                      'BToKEE_l1_pfmvaId_lowPt': {'nbins': 100, 'xmin': -10.0, 'xmax': 10.0},
                      'BToKEE_l2_pfmvaId_lowPt': {'nbins': 100, 'xmin': -10.0, 'xmax': 10.0},
                      'BToKEE_l1_pfmvaId_highPt': {'nbins': 100, 'xmin': -10.0, 'xmax': 10.0},
                      'BToKEE_l2_pfmvaId_highPt': {'nbins': 100, 'xmin': -10.0, 'xmax': 10.0},
                      'BToKEE_l1_isPF': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                      'BToKEE_l2_isPF': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                      'BToKEE_l1_isPFoverlap': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                      'BToKEE_l2_isPFoverlap': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                      'BToKEE_l1_iso04_rel': {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0},
                      'BToKEE_l2_iso04_rel': {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0},
                      'BToKEE_fit_k_pt': {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0},
                      'BToKEE_fit_k_normpt': {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0},
                      'BToKEE_fit_k_eta': {'nbins': 100, 'xmin': -3.0, 'xmax': 3.0},
                      'BToKEE_fit_k_phi': {'nbins': 100, 'xmin': -4.0, 'xmax': 4.0},
                      'BToKEE_k_DCASig': {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0},
                      'BToKEE_k_nValidHits': {'nbins': 100, 'xmin': 0.0, 'xmax': 100.0},
                      'BToKEE_k_iso04_rel': {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0},
                      'BToKEE_svprob': {'nbins': 100, 'xmin': 0.0, 'xmax': 1.0},
                      'BToKEE_fit_cos2D': {'nbins': 100, 'xmin': 0.999, 'xmax': 1.0},
                      'BToKEE_l_xy_sig': {'nbins': 100, 'xmin': 0.0, 'xmax': 50.0},
                      'BToKEE_ptImbalance': {'nbins': 100, 'xmin': 0.0, 'xmax': 5.0},
                      'BToKEE_Dmass': {'nbins': 100, 'xmin': 0.0, 'xmax': 5.0},
                      'BToKEE_Dmass_flip': {'nbins': 100, 'xmin': 0.0, 'xmax': 5.0},
                      'BToKEE_dz': {'nbins': 100, 'xmin': -1.0, 'xmax': 1.0},
                      'BToKEE_eleEtaCats': {'nbins': 3, 'xmin': 0.0, 'xmax': 3.0},
                      'BToKEE_event': {'nbins': 10, 'xmin': 0.0, 'xmax': 10.0},
                      }

    outputbranches['BToKEE_ll_charge'] = {'nbins': 5, 'xmin': -2, 'xmax': 2}


    self._old = False
    outputbranches_old = {'BToKEE_maxDR': {'nbins': 100, 'xmin': 0.0, 'xmax': 4.0},
                          'BToKEE_minDR': {'nbins': 100, 'xmin': 0.0, 'xmax': 4.0},
                          }

    outputbranches_new = {'BToKEE_eleDR': {'nbins': 100, 'xmin': 0.0, 'xmax': 4.0},
                          'BToKEE_llkDR': {'nbins': 100, 'xmin': 0.0, 'xmax': 4.0},
                          }

    if self._old:
      outputbranches.update(outputbranches_old)
    else:
      outputbranches.update(outputbranches_new)

    outputbranches_mva = {'BToKEE_mva': {'nbins': 50, 'xmin': -20.0, 'xmax': 20.0},
                         }

    if evalMVA:
      outputbranches.update(outputbranches_mva)
    outputbranches.update(outputbranches_mva)

    self._yutaPR = False
    outputbranches_yutaPR = {'BToKEE_iso_sv_rel': {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0},
                                    'BToKEE_iso_ntrack': {'nbins': 50, 'xmin': 0.0, 'xmax': 50.0},
                                    'BToKEE_ptImbalance_yutaPR': {'nbins': 100, 'xmin': 0.0, 'xmax': 5.0},
                                    }

    if self._yutaPR:
      outputbranches.update(outputbranches_yutaPR)

    self._newVar = True
    outputbranches_newVar = {'BToKEE_svprob_rank': {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0},
                             'BToKEE_fit_pt_rank': {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0},
                             'BToKEE_fit_cos2D_rank': {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0},
                             'BToKEE_l_xy_rank': {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0},
                             'BToKEE_ptAsym': {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0},
                             'BToKEE_l1_pfmvaId_lowPt': {'nbins': 100, 'xmin': -10.0, 'xmax': 10.0},
                             'BToKEE_l2_pfmvaId_lowPt': {'nbins': 100, 'xmin': -10.0, 'xmax': 10.0},
                             'BToKEE_l1_pfmvaId_highPt': {'nbins': 100, 'xmin': -10.0, 'xmax': 10.0},
                             'BToKEE_l2_pfmvaId_highPt': {'nbins': 100, 'xmin': -10.0, 'xmax': 10.0},
                             }
    if self._newVar:
      outputbranches.update(outputbranches_newVar)

    super(BToKLLAnalyzer_postprocess, self).__init__(inputfiles, outputfile, inputbranches, outputbranches, hist)
    #outputbranches = {'BToKEE_fit_mass': {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0},}
    #super(BToKLLAnalyzer_postprocess, self).__init__(inputfiles, outputfile, inputbranches, outputbranches, hist)


  def run(self):
    print('[BToKLLAnalyzer_postprocess::run] INFO: Running the analyzer...')
    self.print_timestamp()
    self.init_output()
    if self._evalMVA:
      features = ['BToKEE_fit_l1_normpt', #'BToKEE_l1_dxy_sig',
                  'BToKEE_fit_l2_normpt', #'BToKEE_l2_dxy_sig',
                  'BToKEE_fit_k_normpt', #'BToKEE_k_DCASig',
                  'BToKEE_fit_normpt', 'BToKEE_svprob', 'BToKEE_fit_cos2D', 'BToKEE_l_xy_sig',
                  ]
      features += ['BToKEE_dz']
      features += ['BToKEE_eleDR', 'BToKEE_llkDR']
      features += ['BToKEE_l1_iso04_rel', 'BToKEE_l2_iso04_rel', 'BToKEE_k_iso04_rel', 'BToKEE_b_iso04_rel']
      #features += ['BToKEE_ptImbalance']
      features += ['BToKEE_l1_pfmvaId_lowPt', 'BToKEE_l2_pfmvaId_lowPt', 'BToKEE_l1_pfmvaId_highPt', 'BToKEE_l2_pfmvaId_highPt']
      #features += ['BToKEE_l1_mvaId', 'BToKEE_l2_mvaId']

      training_branches = sorted(features)
      mvaCut = -99
      ntree_limit = 1198
      model = xgb.Booster({'nthread': 6})
      model.load_model('../models/xgb_fulldata_26May2020_georgeq2_georgeVar_weighted_pauc2_pf.model')

    for (self._ifile, filename) in enumerate(self._file_in_name):
      print('[BToKLLAnalyzer_postprocess::run] INFO: FILE: {}/{}. Loading file...'.format(self._ifile+1, self._num_files))
      events = uproot.open(filename)['tree']
      print('[BToKLLAnalyzer_postprocess::run] INFO: FILE: {}/{}. Analyzing...'.format(self._ifile+1, self._num_files))
      startTime = time.time()
      for i, params in enumerate(events.iterate(outputtype=pd.DataFrame, entrysteps=1000000)):
        self._branches = params.copy()
        print('Reading chunk {}... Finished opening file in {} s'.format(i, time.time() - startTime))

        #self._branches = self._branches.sort_values('BToKEE_mva', ascending=False).drop_duplicates(['BToKEE_event'], keep='first')

        l1_pf_selection = (self._branches['BToKEE_l1_isPF'])
        l2_pf_selection = (self._branches['BToKEE_l2_isPF'])
        l1_low_selection = np.logical_not(self._branches['BToKEE_l1_isPF']) 
        l2_low_selection = np.logical_not(self._branches['BToKEE_l2_isPF']) 


        pf_selection = l1_pf_selection & l2_pf_selection 
        low_selection = l1_low_selection & l2_low_selection
        overlap_veto_selection = np.logical_not(self._branches['BToKEE_l1_isPFoverlap']) & np.logical_not(self._branches['BToKEE_l2_isPFoverlap'])
        mix_selection = ((l1_pf_selection & l2_low_selection) | (l2_pf_selection & l1_low_selection))
        low_pfveto_selection = low_selection & overlap_veto_selection
        mix_net_selection = overlap_veto_selection & np.logical_not(pf_selection | low_selection)
        all_selection = pf_selection | low_pfveto_selection | mix_net_selection 
        
        #low_notpf_selection = low_selection & np.logical_not(self._branches['BToKEE_l1_isPFoverlap'] & self._branches['BToKEE_l2_isPFoverlap'])
      
        # general selection
        mll_selection = (self._branches['BToKEE_mll_fullfit'] > NR_LOW) #& (self._branches['BToKEE_mll_fullfit'] < NR_UP) # all q2
        #mll_selection = (self._branches['BToKEE_mll_fullfit'] > NR_LOW) & (self._branches['BToKEE_mll_fullfit'] < PSI2S_UP) # full q2
        #mll_selection = (self._branches['BToKEE_mll_fullfit'] > NR_LOW) & (self._branches['BToKEE_mll_fullfit'] < JPSI_LOW) #low q2
        #mll_selection = (self._branches['BToKEE_mll_fullfit'] > JPSI_LOW) & (self._branches['BToKEE_mll_fullfit'] < JPSI_UP) # Jpsi
        #mll_selection = (self._branches['BToKEE_mll_fullfit'] > JPSI_UP) & (self._branches['BToKEE_mll_fullfit'] < PSI2S_UP) # psi(2S)
        #mll_selection = (self._branches['BToKEE_mll_fullfit'] > PSI2S_UP) # high q2
        #mll_selection = (self._branches['BToKEE_mll_fullfit'] > 0.6) & (self._branches['BToKEE_mll_fullfit'] < 2.8) # george q2
        b_upsb_selection = (self._branches['BToKEE_fit_mass'] > B_UP)
        b_bothsb_selection = ((self._branches['BToKEE_fit_mass'] > B_SB_LOW) & (self._branches['BToKEE_fit_mass'] < B_LOW)) | ((self._branches['BToKEE_fit_mass'] > B_UP) & (self._branches['BToKEE_fit_mass'] < B_SB_UP))
        d_veto_selection = self._branches['BToKEE_Dmass'] > D_MASS_CUT

        l1_selection = (self._branches['BToKEE_l1_mvaId'] > 2.0) 
        l2_selection = (self._branches['BToKEE_l2_mvaId'] > -3.0)
        #cutbased_selection = (self._branches['BToKEE_fit_pt'] > 10.0) & (self._branches['BToKEE_l_xy_sig'] > 0.0) & (self._branches['BToKEE_svprob'] > 0.1) & (self._branches['BToKEE_fit_cos2D'] > 0.999)
        #pfmvaId_selection = (self._branches['BToKEE_l1_pfmvaId_lowPt'] > -0.555556) & (self._branches['BToKEE_l2_pfmvaId_lowPt'] > -1.666667) & (self._branches['BToKEE_l1_pfmvaId_highPt'] > -2.777778) & (self._branches['BToKEE_l2_pfmvaId_highPt'] > -4.444444)

        mll_mean_jpsi = triCut_jpsi_mll_mean_pf
        fit_mass_mean_jpsi = triCut_jpsi_mKee_mean_pf
        triCut_jpsi_lower_bound = triCut_jpsi_lower_bound_pf
        triCut_jpsi_upper_bound = triCut_jpsi_upper_bound_pf
        eigVecs_jpsi = triCut_jpsi_rotMatrix_pf 
        self._branches['BToKEE_fit_mass_decorr_jpsi'], self._branches['BToKEE_mll_fullfit_decorr_jpsi'] = get_diagonalCut_var(self._branches, mll_mean_jpsi, fit_mass_mean_jpsi, triCut_jpsi_lower_bound, triCut_jpsi_upper_bound, eigVecs_jpsi)

        ll_charge_selection = (abs(self._branches['BToKEE_ll_charge']) != 0)

        general_selection = l1_selection & l2_selection
        #general_selection = d_veto_selection
        #general_selection &= (self._branches['BToKEE_eleEtaCats'] == 0)
        general_selection &= mll_selection
        general_selection &= pf_selection
        #general_selection &= low_selection
        #general_selection &= low_notpf_selection
        #general_selection &= mix_net_selection
        #general_selection &= low_pfveto_selection
        #general_selection &= b_upsb_selection
        #general_selection &= b_bothsb_selection
        #general_selection &= (self._branches['BToKEE_mva'] > 13.54)
        #general_selection &= (self._branches['BToKEE_fit_mass'] > 4.8) & (self._branches['BToKEE_fit_mass'] < 6.0)
        #general_selection &= (self._branches['BToKEE_l1_pfmvaId'] > -2.0) & (self._branches['BToKEE_l2_pfmvaId'] > -2.0)
        #general_selection &= cutbased_selection
        #general_selection &= (self._branches['BToKEE_l1_pfmvaId'] > 0.0) & (self._branches['BToKEE_l2_pfmvaId'] > 0.0)
        #general_selection &= pfmvaId_selection
        #general_selection &= (self._branches['BToKEE_mll_fullfit_decorr_jpsi'] < triCut_jpsi_lower_bound)
        general_selection &= ll_charge_selection

        self._branches = self._branches[general_selection]
        #self._branches = self._branches.sort_values('BToKEE_mva', ascending=False).drop_duplicates(['BToKEE_event'], keep='first')

        if self._evalMVA:
          #self._branches['BToKEE_mva'] = model.predict(xgb.DMatrix(self._branches[training_branches].sort_index(axis=1).values), ntree_limit=ntree_limit)
          self._branches['BToKEE_mva'] = model.predict(xgb.DMatrix(self._branches[training_branches].replace([np.inf, -np.inf], 0.0).sort_index(axis=1)), ntree_limit=ntree_limit)
          #self._branches = self._branches[(self._branches['BToKEE_mva'] > mvaCut)].sort_values('BToKEE_mva', ascending=False).drop_duplicates(['BToKEE_event'], keep='first')
          self._branches = self._branches[(self._branches['BToKEE_mva'] > mvaCut)]

        self.fill_output()
        startTime = time.time()

    self.finish()
    print('[BToKLLAnalyzer_postprocess::run] INFO: Finished')
    self.print_timestamp()






