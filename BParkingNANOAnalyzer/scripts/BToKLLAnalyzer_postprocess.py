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
  def __init__(self, inputfiles, outputfile, evalMVA=False):
    self._evalMVA = evalMVA
    inputbranches= ['*',]
    outputbranches = []
    outputbranches += ['BToKEE_mll_fullfit', 'BToKEE_q2', 'BToKEE_fit_mass', 'BToKEE_fit_massErr']
    outputbranches += ['BToKEE_fit_l1_pt', 'BToKEE_fit_l1_normpt', 'BToKEE_fit_l1_eta', 'BToKEE_fit_l1_phi']
    outputbranches += ['BToKEE_l1_dxy_sig', 'BToKEE_l1_dzTrg']
    outputbranches += ['BToKEE_l1_mvaId', 'BToKEE_l1_pfmvaId', 'BToKEE_l1_pfmvaCats', 'BToKEE_l1_isPF', 'BToKEE_l1_isPFoverlap']
    outputbranches += ['BToKEE_l1_pfmvaId_lowPt', 'BToKEE_l1_pfmvaId_highPt']
    outputbranches += ['BToKEE_l1_convOpen', 'BToKEE_l1_convLoose', 'BToKEE_l1_convTight']
    outputbranches += ['BToKEE_fit_l2_pt', 'BToKEE_fit_l2_normpt', 'BToKEE_fit_l2_eta', 'BToKEE_fit_l2_phi']
    outputbranches += ['BToKEE_l2_dxy_sig', 'BToKEE_l2_dzTrg']
    outputbranches += ['BToKEE_l2_mvaId', 'BToKEE_l2_pfmvaId', 'BToKEE_l2_pfmvaCats', 'BToKEE_l2_isPF', 'BToKEE_l2_isPFoverlap']
    outputbranches += ['BToKEE_l2_pfmvaId_lowPt', 'BToKEE_l2_pfmvaId_highPt']
    outputbranches += ['BToKEE_l2_convOpen', 'BToKEE_l2_convLoose', 'BToKEE_l2_convTight']
    outputbranches += ['BToKEE_fit_k_pt', 'BToKEE_fit_k_normpt', 'BToKEE_fit_k_eta', 'BToKEE_fit_k_phi']
    outputbranches += ['BToKEE_k_DCASig', 'BToKEE_k_nValidHits']
    outputbranches += ['BToKEE_k_svip2d', 'BToKEE_k_dzTrg', 'BToKEE_k_svip3d']
    outputbranches += ['BToKEE_fit_pt', 'BToKEE_fit_normpt', 'BToKEE_fit_eta', 'BToKEE_fit_phi']
    outputbranches += ['BToKEE_l_xy_sig', 'BToKEE_svprob', 'BToKEE_fit_cos2D']
    outputbranches += ['BToKEE_ptAsym', 'BToKEE_ptImbalance', 'BToKEE_Dmass', 'BToKEE_Dmass_flip']
    outputbranches += ['BToKEE_eleDR', 'BToKEE_llkDR']
    outputbranches += ['BToKEE_l1_iso04_rel', 'BToKEE_l2_iso04_rel', 'BToKEE_k_iso04_rel', 'BToKEE_b_iso04_rel']
    outputbranches += ['BToKEE_l1_n_isotrk', 'BToKEE_l2_n_isotrk', 'BToKEE_k_n_isotrk', 'BToKEE_b_n_isotrk']
    outputbranches += ['BToKEE_eleEtaCats', 'BToKEE_event']
    outputbranches += ['BToKEE_fold',]

    outputbranches_mva = ['BToKEE_mva',]

    if evalMVA:
      outputbranches += outputbranches_mva

    super(BToKLLAnalyzer_postprocess, self).__init__(inputfiles, outputfile, inputbranches, outputbranches)

  def run(self):
    print('[BToKLLAnalyzer_postprocess::run] INFO: Running the analyzer...')
    self.print_timestamp()
    self.init_output()
    if self._evalMVA:
      features = []
      features += ['BToKEE_fit_l1_normpt', 'BToKEE_fit_l2_normpt',
                  'BToKEE_l1_dxy_sig', 'BToKEE_l2_dxy_sig',
                  'BToKEE_fit_k_normpt', 'BToKEE_k_DCASig',
                  'BToKEE_fit_normpt', 'BToKEE_svprob', 'BToKEE_fit_cos2D', 'BToKEE_l_xy_sig', #'BToKEE_dz'
                  ]
      features += ['BToKEE_eleDR', 'BToKEE_llkDR']
      features += ['BToKEE_l1_iso04_rel', 'BToKEE_l2_iso04_rel', 'BToKEE_k_iso04_rel', 'BToKEE_b_iso04_rel']
      features += ['BToKEE_ptAsym']
      features += ['BToKEE_l1_pfmvaId_lowPt', 'BToKEE_l2_pfmvaId_lowPt', 'BToKEE_l1_pfmvaId_highPt', 'BToKEE_l2_pfmvaId_highPt']
      features += ['BToKEE_l1_mvaId', 'BToKEE_l2_mvaId']
      features += ['BToKEE_l1_dzTrg', 'BToKEE_l2_dzTrg', 'BToKEE_k_dzTrg']
      features += ['BToKEE_k_svip2d', 'BToKEE_k_svip3d']

      training_branches = sorted(features)
      mvaCut = 10.0
      ntree_limit = 800
      model = xgb.Booster({'nthread': 6})
      model.load_model('../models/xgb_fulldata_13March2021_lowq2_onePerEvent_pauc2_mix.model')

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
        #mll_selection = (self._branches['BToKEE_mll_fullfit'] > LOWQ2_LOW) #& (self._branches['BToKEE_mll_fullfit'] < NR_UP) # all q2
        #mll_selection = (self._branches['BToKEE_mll_fullfit'] > LOWQ2_LOW) & (self._branches['BToKEE_mll_fullfit'] < LOWQ2_UP) #low q2
        #mll_selection = (self._branches['BToKEE_mll_fullfit'] > JPSI_LOW) & (self._branches['BToKEE_mll_fullfit'] < JPSI_UP) # Jpsi
        #mll_selection = (self._branches['BToKEE_mll_fullfit'] > JPSI_UP) & (self._branches['BToKEE_mll_fullfit'] < PSI2S_UP) # psi(2S)
        mll_selection = (self._branches['BToKEE_mll_fullfit'] > PSI2S_UP) #high q2
        b_upsb_selection = (self._branches['BToKEE_fit_mass'] > B_UP)
        b_bothsb_selection = ((self._branches['BToKEE_fit_mass'] > B_SB_LOW) & (self._branches['BToKEE_fit_mass'] < B_LOW)) | ((self._branches['BToKEE_fit_mass'] > B_UP) & (self._branches['BToKEE_fit_mass'] < B_SB_UP))
        d_veto_selection = self._branches['BToKEE_Dmass'] > D_MASS_CUT

        l1_selection = (self._branches['BToKEE_l1_mvaId'] > -99.0) 
        l2_selection = (self._branches['BToKEE_l2_mvaId'] > -99.0)

        mll_mean_jpsi = triCut_jpsi_mll_mean_pf
        fit_mass_mean_jpsi = triCut_jpsi_mKee_mean_pf
        triCut_jpsi_lower_bound = triCut_jpsi_lower_bound_pf
        triCut_jpsi_upper_bound = triCut_jpsi_upper_bound_pf
        eigVecs_jpsi = triCut_jpsi_rotMatrix_pf 
        self._branches['BToKEE_fit_mass_decorr_jpsi'], self._branches['BToKEE_mll_fullfit_decorr_jpsi'] = get_diagonalCut_var(self._branches, mll_mean_jpsi, fit_mass_mean_jpsi, triCut_jpsi_lower_bound, triCut_jpsi_upper_bound, eigVecs_jpsi)

        #ll_charge_selection = (abs(self._branches['BToKEE_ll_charge']) != 0)

        general_selection = l1_selection & l2_selection
        general_selection &= mll_selection
        #general_selection &= pf_selection
        general_selection &= mix_net_selection
        #general_selection &= low_pfveto_selection
        #general_selection &= b_upsb_selection
        #general_selection &= b_bothsb_selection
        #general_selection &= (self._branches['BToKEE_fit_l1_pt'] > 2.0) & (self._branches['BToKEE_fit_l2_pt'] > 2.0)
        #general_selection &= (self._branches['BToKEE_fit_l1_pt'] > 2.0)
        #general_selection &= (self._branches['BToKEE_HLT_Mu9_IP6'])
        #general_selection &= (abs(self._branches['BToKEE_k_svip3d']) < 0.06)

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






