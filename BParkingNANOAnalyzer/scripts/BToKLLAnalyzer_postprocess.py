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
    outputbranches = {'BToKEE_mll_raw': {'nbins': 100, 'xmin': 0.0, 'xmax': 4.5},
                      'BToKEE_mll_fullfit': {'nbins': 50, 'xmin': 2.6, 'xmax': 3.3},
                      'BToKEE_mll_llfit': {'nbins': 100, 'xmin': 0.0, 'xmax': 4.5},
                      'BToKEE_q2': {'nbins': 100, 'xmin': 0.0, 'xmax': 25.0},
                      'BToKEE_fit_mass': {'nbins': 100, 'xmin': 4.5, 'xmax': 6.0},
                      'BToKEE_fit_massErr': {'nbins': 100, 'xmin': 0.0, 'xmax': 0.5},
                      'BToKEE_fit_pt': {'nbins': 100, 'xmin': 0.0, 'xmax': 30.0},
                      'BToKEE_fit_normpt': {'nbins': 100, 'xmin': 0.0, 'xmax': 30.0},
                      'BToKEE_fit_eta': {'nbins': 100, 'xmin': -3.0, 'xmax': 3.0},
                      'BToKEE_fit_phi': {'nbins': 100, 'xmin': -4.0, 'xmax': 4.0},
                      'BToKEE_b_iso03_rel': {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0},
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
                      'BToKEE_l1_dz': {'nbins': 100, 'xmin': -1.0, 'xmax': 1.0},
                      'BToKEE_l2_dz': {'nbins': 100, 'xmin': -1.0, 'xmax': 1.0},
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
                      'BToKEE_l1_isLowPt': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                      'BToKEE_l2_isLowPt': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                      'BToKEE_l1_isPFoverlap': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                      'BToKEE_l2_isPFoverlap': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                      'BToKEE_l1_iso03_rel': {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0},
                      'BToKEE_l2_iso03_rel': {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0},
                      'BToKEE_l1_iso04_rel': {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0},
                      'BToKEE_l2_iso04_rel': {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0},
                      'BToKEE_fit_k_pt': {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0},
                      'BToKEE_fit_k_normpt': {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0},
                      'BToKEE_fit_k_eta': {'nbins': 100, 'xmin': -3.0, 'xmax': 3.0},
                      'BToKEE_fit_k_phi': {'nbins': 100, 'xmin': -4.0, 'xmax': 4.0},
                      'BToKEE_k_DCASig': {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0},
                      'BToKEE_k_dz': {'nbins': 100, 'xmin': -1.0, 'xmax': 1.0},
                      'BToKEE_k_nValidHits': {'nbins': 100, 'xmin': 0.0, 'xmax': 100.0},
                      'BToKEE_k_iso03_rel': {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0},
                      'BToKEE_k_iso04_rel': {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0},
                      'BToKEE_svprob': {'nbins': 100, 'xmin': 0.0, 'xmax': 1.0},
                      'BToKEE_fit_cos2D': {'nbins': 100, 'xmin': 0.999, 'xmax': 1.0},
                      'BToKEE_l_xy_sig': {'nbins': 100, 'xmin': 0.0, 'xmax': 50.0},
                      'BToKEE_ptImbalance': {'nbins': 100, 'xmin': 0.0, 'xmax': 50.0},
                      'BToKEE_Dmass': {'nbins': 100, 'xmin': 0.0, 'xmax': 5.0},
                      'BToKEE_Dmass_flip': {'nbins': 100, 'xmin': 0.0, 'xmax': 5.0},
                      #'BToKEE_pill_mass': {'nbins': 100, 'xmin': 0.0, 'xmax': 5.0},
                      'BToKEE_maxDR': {'nbins': 100, 'xmin': 0.0, 'xmax': 4.0},
                      'BToKEE_minDR': {'nbins': 100, 'xmin': 0.0, 'xmax': 4.0},
                      'BToKEE_fit_l1_dphi': {'nbins': 100, 'xmin': -4.0, 'xmax': 4.0},
                      'BToKEE_fit_l2_dphi': {'nbins': 100, 'xmin': -4.0, 'xmax': 4.0},
                      'BToKEE_fit_k_dphi': {'nbins': 100, 'xmin': -4.0, 'xmax': 4.0},
                      'BToKEE_fit_dphi': {'nbins': 100, 'xmin': -4.0, 'xmax': 4.0},
                      'BToKEE_trg_eta': {'nbins': 100, 'xmin': -3.0, 'xmax': 3.0},
                      'BToKEE_eleEtaCats': {'nbins': 3, 'xmin': 0.0, 'xmax': 3.0},
                      'BToKEE_event': {'nbins': 10, 'xmin': 0.0, 'xmax': 10.0},
                      }

    outputbranches_mva = {'BToKEE_xgb': {'nbins': 50, 'xmin': -20.0, 'xmax': 20.0},
                         }

    if evalMVA:
      outputbranches.update(outputbranches_mva)

    super(BToKLLAnalyzer_postprocess, self).__init__(inputfiles, outputfile, inputbranches, outputbranches, hist)

  def run(self):
    print('[BToKLLAnalyzer_postprocess::run] INFO: Running the analyzer...')
    self.print_timestamp()
    self.init_output()
    if self._evalMVA:
      features = ['BToKEE_fit_l1_normpt', 'BToKEE_fit_l1_eta', 'BToKEE_l1_dxy_sig', 'BToKEE_l1_dz',
                  'BToKEE_fit_l2_normpt', 'BToKEE_fit_l2_eta', 'BToKEE_l2_dxy_sig', 'BToKEE_l2_dz', 
                  'BToKEE_fit_k_normpt', 'BToKEE_fit_k_eta', 'BToKEE_k_DCASig', 'BToKEE_k_dz',
                  'BToKEE_fit_normpt', 'BToKEE_fit_eta', 'BToKEE_svprob', 'BToKEE_fit_cos2D', 'BToKEE_l_xy_sig',
                  ]
      #features += ['BToKEE_fit_l1_phi', 'BToKEE_fit_l2_phi', 'BToKEE_fit_k_phi', 'BToKEE_fit_phi']
      #features += ['BToKEE_fit_l1_dphi', 'BToKEE_fit_l2_dphi', 'BToKEE_fit_k_dphi', 'BToKEE_fit_dphi']
      features += ['BToKEE_minDR', 'BToKEE_maxDR']
      features += ['BToKEE_l1_iso04_rel', 'BToKEE_l2_iso04_rel', 'BToKEE_k_iso04_rel', 'BToKEE_b_iso04_rel']
      #features += ['BToKEE_l1_iso03_rel', 'BToKEE_l2_iso03_rel', 'BToKEE_k_iso03_rel', 'BToKEE_b_iso03_rel']
      features += ['BToKEE_l1_pfmvaId_lowPt', 'BToKEE_l2_pfmvaId_lowPt', 'BToKEE_l1_pfmvaId_highPt', 'BToKEE_l2_pfmvaId_highPt']
      features += ['BToKEE_ptImbalance']
      #features += ['BToKEE_trg_eta']
      #features += ['BToKEE_l1_mvaId', 'BToKEE_l2_mvaId']

      training_branches = sorted(features)
      mvaCut = 5.0
      ntree_limit = 765
      model = xgb.Booster({'nthread': 6})
      model.load_model('xgb_fulldata_01Mar2020_fullq2_isoPFMVACutDRptImb_weighted_pauc02_pf.model')

    for (self._ifile, filename) in enumerate(self._file_in_name):
      print('[BToKLLAnalyzer_postprocess::run] INFO: FILE: {}/{}. Loading file...'.format(self._ifile+1, self._num_files))
      events = uproot.open(filename)['tree']
      print('[BToKLLAnalyzer_postprocess::run] INFO: FILE: {}/{}. Analyzing...'.format(self._ifile+1, self._num_files))
      startTime = time.time()
      for i, params in enumerate(events.iterate(outputtype=pd.DataFrame, entrysteps=1000000)):
        self._branches = params.copy()
        print('Reading chunk {}... Finished opening file in {} s'.format(i, time.time() - startTime))

        self._branches['BToKEE_l1_pfmvaId_lowPt'] = np.where(self._branches['BToKEE_l1_pfmvaCats'] == 0, self._branches['BToKEE_l1_pfmvaId'], 20.0)
        self._branches['BToKEE_l2_pfmvaId_lowPt'] = np.where(self._branches['BToKEE_l2_pfmvaCats'] == 0, self._branches['BToKEE_l2_pfmvaId'], 20.0)
        self._branches['BToKEE_l1_pfmvaId_highPt'] = np.where(self._branches['BToKEE_l1_pfmvaCats'] == 1, self._branches['BToKEE_l1_pfmvaId'], 20.0)
        self._branches['BToKEE_l2_pfmvaId_highPt'] = np.where(self._branches['BToKEE_l2_pfmvaCats'] == 1, self._branches['BToKEE_l2_pfmvaId'], 20.0)

        l1_pf_selection = (self._branches['BToKEE_l1_isPF'])
        l2_pf_selection = (self._branches['BToKEE_l2_isPF'])
        l1_low_selection = (self._branches['BToKEE_l1_isLowPt']) 
        l2_low_selection = (self._branches['BToKEE_l2_isLowPt']) 

        pf_selection = l1_pf_selection & l2_pf_selection 
        low_selection = l1_low_selection & l2_low_selection
        overlap_veto_selection = np.logical_not(self._branches['BToKEE_l1_isPFoverlap']) & np.logical_not(self._branches['BToKEE_l2_isPFoverlap'])
        mix_selection = ((l1_pf_selection & l2_low_selection) | (l2_pf_selection & l1_low_selection))
        low_pfveto_selection = low_selection & overlap_veto_selection
        mix_net_selection = overlap_veto_selection & np.logical_not(pf_selection | low_selection)
        all_selection = pf_selection | low_pfveto_selection | mix_net_selection 

        # general selection
        jpsi_selection = (self._branches['BToKEE_mll_fullfit'] > NR_LOW) & (self._branches['BToKEE_mll_fullfit'] < JPSI_UP) # full q2
        #jpsi_selection = (self._branches['BToKEE_mll_fullfit'] > NR_LOW) & (self._branches['BToKEE_mll_fullfit'] < JPSI_LOW) #low q2
        #jpsi_selection = (self._branches['BToKEE_mll_fullfit'] > JPSI_LOW) & (self._branches['BToKEE_mll_fullfit'] < JPSI_UP) # Jpsi
        b_upsb_selection = (self._branches['BToKEE_fit_mass'] > B_UP)
        b_bothsb_selection = ((self._branches['BToKEE_fit_mass'] > B_SB_LOW) & (self._branches['BToKEE_fit_mass'] < B_LOW)) | ((self._branches['BToKEE_fit_mass'] > B_UP) & (self._branches['BToKEE_fit_mass'] < B_SB_UP))
        d_veto_selection = self._branches['BToKEE_Dmass'] > D_MASS_CUT

        l1_selection = (self._branches['BToKEE_l1_mvaId'] > 3.5) 
        l2_selection = (self._branches['BToKEE_l2_mvaId'] > 3.5)

        general_selection = l1_selection & l2_selection
        #general_selection = d_veto_selection
        #general_selection &= (self._branches['BToKEE_eleEtaCats'] == 0)
        general_selection &= jpsi_selection
        general_selection &= pf_selection
        #general_selection &= mix_net_selection
        #general_selection &= low_pfveto_selection
        #general_selection &= b_upsb_selection
        #general_selection &= b_bothsb_selection
        #general_selection &= (self._branches['BToKEE_xgb'] > 7.0)
        general_selection &= (self._branches['BToKEE_l1_pfmvaId'] > -2.0) & (self._branches['BToKEE_l2_pfmvaId'] > -2.0)

        self._branches = self._branches[general_selection]

        if self._evalMVA:
          self._branches['BToKEE_xgb'] = model.predict(xgb.DMatrix(self._branches[training_branches].sort_index(axis=1).values), ntree_limit=ntree_limit)
          self._branches = self._branches[(self._branches['BToKEE_xgb'] > mvaCut)].sort_values('BToKEE_xgb', ascending=False).drop_duplicates(['BToKEE_event'], keep='first')

        self.fill_output()
        startTime = time.time()

    self.finish()
    print('[BToKLLAnalyzer_postprocess::run] INFO: Finished')
    self.print_timestamp()






