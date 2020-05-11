#! /usr/bin/env python
import uproot
import uproot_methods
import pandas as pd
import numpy as np
import time
from helper import *
import xgboost as xgb
from BParkingNANOAnalysis.BParkingNANOAnalyzer.BaseAnalyzer import BParkingNANOAnalyzer

class BToPhiLLAnalyzer_postprocess(BParkingNANOAnalyzer):
  def __init__(self, inputfiles, outputfile, hist=False, evalMVA=False):
    self._evalMVA = evalMVA
    inputbranches= ['*',]
    outputbranches = {'BToPhiEE_mll_fullfit': {'nbins': 50, 'xmin': 2.6, 'xmax': 3.3},
                      'BToPhiEE_q2': {'nbins': 100, 'xmin': 0.0, 'xmax': 25.0},
                      'BToPhiEE_fit_phi_mass': {'nbins': 100, 'xmin': 0.98, 'xmax': 1.06},
                      'BToPhiEE_fit_mass': {'nbins': 100, 'xmin': 4.5, 'xmax': 6.0},
                      'BToPhiEE_fit_massErr': {'nbins': 100, 'xmin': 0.0, 'xmax': 0.5},
                      'BToPhiEE_fit_pt': {'nbins': 100, 'xmin': 0.0, 'xmax': 30.0},
                      'BToPhiEE_fit_normpt': {'nbins': 100, 'xmin': 0.0, 'xmax': 30.0},
                      'BToPhiEE_fit_eta': {'nbins': 100, 'xmin': -3.0, 'xmax': 3.0},
                      'BToPhiEE_fit_phi': {'nbins': 100, 'xmin': -4.0, 'xmax': 4.0},
                      'BToPhiEE_b_iso04_rel': {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0},
                      'BToPhiEE_fit_l1_pt': {'nbins': 100, 'xmin': 0.0, 'xmax': 30.0},
                      'BToPhiEE_fit_l2_pt': {'nbins': 100, 'xmin': 0.0, 'xmax': 30.0},
                      'BToPhiEE_fit_l1_normpt': {'nbins': 100, 'xmin': 0.0, 'xmax': 30.0},
                      'BToPhiEE_fit_l2_normpt': {'nbins': 100, 'xmin': 0.0, 'xmax': 30.0},
                      'BToPhiEE_fit_l1_eta': {'nbins': 100, 'xmin': -3.0, 'xmax': 3.0},
                      'BToPhiEE_fit_l2_eta': {'nbins': 100, 'xmin': -3.0, 'xmax': 3.0},
                      'BToPhiEE_fit_l1_phi': {'nbins': 100, 'xmin': -4.0, 'xmax': 4.0},
                      'BToPhiEE_fit_l2_phi': {'nbins': 100, 'xmin': -4.0, 'xmax': 4.0},
                      'BToPhiEE_l1_dxy_sig': {'nbins': 100, 'xmin': -30.0, 'xmax': 30.0},
                      'BToPhiEE_l2_dxy_sig': {'nbins': 100, 'xmin': -30.0, 'xmax': 30.0},
                      'BToPhiEE_l1_mvaId': {'nbins': 100, 'xmin': -2.0, 'xmax': 10.0},
                      'BToPhiEE_l2_mvaId': {'nbins': 100, 'xmin': -2.0, 'xmax': 10.0},
                      'BToPhiEE_l1_pfmvaId': {'nbins': 100, 'xmin': -10.0, 'xmax': 10.0},
                      'BToPhiEE_l2_pfmvaId': {'nbins': 100, 'xmin': -10.0, 'xmax': 10.0},
                      'BToPhiEE_l1_pfmvaCats': {'nbins': 2, 'xmin': 0.0, 'xmax': 2.0},
                      'BToPhiEE_l2_pfmvaCats': {'nbins': 2, 'xmin': 0.0, 'xmax': 2.0},
                      'BToPhiEE_l1_pfmvaId_lowPt': {'nbins': 100, 'xmin': -10.0, 'xmax': 10.0},
                      'BToPhiEE_l2_pfmvaId_lowPt': {'nbins': 100, 'xmin': -10.0, 'xmax': 10.0},
                      'BToPhiEE_l1_pfmvaId_highPt': {'nbins': 100, 'xmin': -10.0, 'xmax': 10.0},
                      'BToPhiEE_l2_pfmvaId_highPt': {'nbins': 100, 'xmin': -10.0, 'xmax': 10.0},
                      'BToPhiEE_l1_isPF': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                      'BToPhiEE_l2_isPF': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                      'BToPhiEE_l1_isPFoverlap': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                      'BToPhiEE_l2_isPFoverlap': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                      'BToPhiEE_l1_iso04_rel': {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0},
                      'BToPhiEE_l2_iso04_rel': {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0},
                      'BToPhiEE_fit_trk1_pt': {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0},
                      'BToPhiEE_fit_trk1_normpt': {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0},
                      'BToPhiEE_fit_trk1_eta': {'nbins': 100, 'xmin': -3.0, 'xmax': 3.0},
                      'BToPhiEE_fit_trk1_phi': {'nbins': 100, 'xmin': -4.0, 'xmax': 4.0},
                      'BToPhiEE_trk1_DCASig': {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0},
                      'BToPhiEE_trk1_iso04_rel': {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0},
                      'BToPhiEE_fit_trk2_pt': {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0},
                      'BToPhiEE_fit_trk2_normpt': {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0},
                      'BToPhiEE_fit_trk2_eta': {'nbins': 100, 'xmin': -3.0, 'xmax': 3.0},
                      'BToPhiEE_fit_trk2_phi': {'nbins': 100, 'xmin': -4.0, 'xmax': 4.0},
                      'BToPhiEE_trk2_DCASig': {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0},
                      'BToPhiEE_trk2_iso04_rel': {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0},
                      'BToPhiEE_svprob': {'nbins': 100, 'xmin': 0.0, 'xmax': 1.0},
                      'BToPhiEE_fit_cos2D': {'nbins': 100, 'xmin': 0.999, 'xmax': 1.0},
                      'BToPhiEE_l_xy_sig': {'nbins': 100, 'xmin': 0.0, 'xmax': 50.0},
                      'BToPhiEE_ptImbalance': {'nbins': 100, 'xmin': 0.0, 'xmax': 5.0},
                      'BToPhiEE_dz': {'nbins': 100, 'xmin': -1.0, 'xmax': 1.0},
                      'BToPhiEE_eleEtaCats': {'nbins': 3, 'xmin': 0.0, 'xmax': 3.0},
                      'BToPhiEE_event': {'nbins': 10, 'xmin': 0.0, 'xmax': 10.0},
                      'BToPhiEE_eleDR': {'nbins': 100, 'xmin': 0.0, 'xmax': 4.0},
                      'BToPhiEE_trkDR': {'nbins': 100, 'xmin': 0.0, 'xmax': 4.0},
                      'BToPhiEE_llkkDR': {'nbins': 100, 'xmin': 0.0, 'xmax': 4.0},
                      'BToPhiEE_svprob_rank': {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0},
                      'BToPhiEE_fit_pt_rank': {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0},
                      'BToPhiEE_fit_cos2D_rank': {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0},
                      'BToPhiEE_l_xy_rank': {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0},
                      'BToPhiEE_ptAsym': {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0},
                      'BToPhiEE_l1_pfmvaId_lowPt': {'nbins': 100, 'xmin': -10.0, 'xmax': 10.0},
                      'BToPhiEE_l2_pfmvaId_lowPt': {'nbins': 100, 'xmin': -10.0, 'xmax': 10.0},
                      'BToPhiEE_l1_pfmvaId_highPt': {'nbins': 100, 'xmin': -10.0, 'xmax': 10.0},
                      'BToPhiEE_l2_pfmvaId_highPt': {'nbins': 100, 'xmin': -10.0, 'xmax': 10.0},
                      }

    outputbranches_mva = {'BToPhiEE_mva': {'nbins': 50, 'xmin': -20.0, 'xmax': 20.0},
                         }

    if evalMVA:
      outputbranches.update(outputbranches_mva)
    #outputbranches.update(outputbranches_mva)

    super(BToPhiLLAnalyzer_postprocess, self).__init__(inputfiles, outputfile, inputbranches, outputbranches, hist)

  def run(self):
    print('[BToPhiLLAnalyzer_postprocess::run] INFO: Running the analyzer...')
    self.print_timestamp()
    self.init_output()
    if self._evalMVA:
      features = ['BToPhiEE_fit_l1_normpt', 'BToPhiEE_l1_dxy_sig',
                  'BToPhiEE_fit_l2_normpt', 'BToPhiEE_l2_dxy_sig',
                  'BToPhiEE_fit_k_normpt', 'BToPhiEE_k_DCASig',
                  'BToPhiEE_fit_normpt', 'BToPhiEE_svprob', 'BToPhiEE_fit_cos2D', 'BToPhiEE_l_xy_sig',
                  ]
      features += ['BToPhiEE_dz']
      features += ['BToPhiEE_eleDR', 'BToPhiEE_llkDR']
      features += ['BToPhiEE_l1_iso04_rel', 'BToPhiEE_l2_iso04_rel', 'BToPhiEE_k_iso04_rel', 'BToPhiEE_b_iso04_rel']
      features += ['BToPhiEE_ptImbalance']
      features += ['BToPhiEE_l1_pfmvaId_lowPt', 'BToPhiEE_l2_pfmvaId_lowPt', 'BToPhiEE_l1_pfmvaId_highPt', 'BToPhiEE_l2_pfmvaId_highPt']
      #features += ['BToPhiEE_l1_mvaId', 'BToPhiEE_l2_mvaId']

      training_branches = sorted(features)
      mvaCut = -99
      ntree_limit = 1042
      model = xgb.Booster({'nthread': 6})
      model.load_model('xgb_fulldata_19Apr2020_allq2_isoPFMVANewDRptImb_weighted_pauc02_pf.model')

    for (self._ifile, filename) in enumerate(self._file_in_name):
      print('[BToPhiLLAnalyzer_postprocess::run] INFO: FILE: {}/{}. Loading file...'.format(self._ifile+1, self._num_files))
      events = uproot.open(filename)['tree']
      print('[BToPhiLLAnalyzer_postprocess::run] INFO: FILE: {}/{}. Analyzing...'.format(self._ifile+1, self._num_files))
      startTime = time.time()
      for i, params in enumerate(events.iterate(outputtype=pd.DataFrame, entrysteps=500000)):
        self._branches = params.copy()
        print('Reading chunk {}... Finished opening file in {} s'.format(i, time.time() - startTime))

        l1_pf_selection = (self._branches['BToPhiEE_l1_isPF'])
        l2_pf_selection = (self._branches['BToPhiEE_l2_isPF'])
        l1_low_selection = np.logical_not(self._branches['BToPhiEE_l1_isPF']) 
        l2_low_selection = np.logical_not(self._branches['BToPhiEE_l2_isPF']) 

        pf_selection = l1_pf_selection & l2_pf_selection 
        low_selection = l1_low_selection & l2_low_selection
        overlap_veto_selection = np.logical_not(self._branches['BToPhiEE_l1_isPFoverlap']) & np.logical_not(self._branches['BToPhiEE_l2_isPFoverlap'])
        mix_selection = ((l1_pf_selection & l2_low_selection) | (l2_pf_selection & l1_low_selection))
        low_pfveto_selection = low_selection & overlap_veto_selection
        mix_net_selection = overlap_veto_selection & np.logical_not(pf_selection | low_selection)
        all_selection = pf_selection | low_pfveto_selection | mix_net_selection 
        
        #low_notpf_selection = low_selection & np.logical_not(self._branches['BToPhiEE_l1_isPFoverlap'] & self._branches['BToPhiEE_l2_isPFoverlap'])
      
        # general selection
        #mll_selection = (self._branches['BToPhiEE_mll_fullfit'] > NR_LOW) #& (self._branches['BToPhiEE_mll_fullfit'] < NR_UP) # all q2
        #mll_selection = (self._branches['BToPhiEE_mll_fullfit'] > NR_LOW) & (self._branches['BToPhiEE_mll_fullfit'] < PSI2S_UP) # full q2
        #mll_selection = (self._branches['BToPhiEE_mll_fullfit'] > NR_LOW) & (self._branches['BToPhiEE_mll_fullfit'] < JPSI_LOW) #low q2
        mll_selection = (self._branches['BToPhiEE_mll_fullfit'] > JPSI_LOW) & (self._branches['BToPhiEE_mll_fullfit'] < JPSI_UP) # Jpsi
        #mll_selection = (self._branches['BToPhiEE_mll_fullfit'] > JPSI_UP) & (self._branches['BToPhiEE_mll_fullfit'] < PSI2S_UP) # psi(2S)
        mkk_selection = (self._branches['BToPhiEE_fit_phi_mass'] > PHI_LOW) & (self._branches['BToPhiEE_fit_phi_mass'] < PHI_UP)
        b_upsb_selection = (self._branches['BToPhiEE_fit_mass'] > B_UP)
        b_bothsb_selection = ((self._branches['BToPhiEE_fit_mass'] > B_SB_LOW) & (self._branches['BToPhiEE_fit_mass'] < B_LOW)) | ((self._branches['BToPhiEE_fit_mass'] > B_UP) & (self._branches['BToPhiEE_fit_mass'] < B_SB_UP))

        l1_selection = (self._branches['BToPhiEE_l1_mvaId'] > 3.0) 
        l2_selection = (self._branches['BToPhiEE_l2_mvaId'] > 0.0)

        general_selection = l1_selection & l2_selection
        general_selection &= mll_selection
        general_selection &= mkk_selection
        general_selection &= pf_selection
        #general_selection &= low_selection
        #general_selection &= b_upsb_selection

        self._branches = self._branches[general_selection]
        #self._branches = self._branches.sort_values('BToPhiEE_mva', ascending=False).drop_duplicates(['BToPhiEE_event'], keep='first')

        if self._evalMVA:
          #self._branches['BToPhiEE_mva'] = model.predict(xgb.DMatrix(self._branches[training_branches].sort_index(axis=1).values), ntree_limit=ntree_limit)
          self._branches['BToPhiEE_mva'] = model.predict(xgb.DMatrix(self._branches[training_branches].replace([np.inf, -np.inf], 0.0).sort_index(axis=1)), ntree_limit=ntree_limit)
          #self._branches = self._branches[(self._branches['BToPhiEE_mva'] > mvaCut)].sort_values('BToPhiEE_mva', ascending=False).drop_duplicates(['BToPhiEE_event'], keep='first')
          self._branches = self._branches[(self._branches['BToPhiEE_mva'] > mvaCut)]

        self.fill_output()
        startTime = time.time()

    self.finish()
    print('[BToPhiLLAnalyzer_postprocess::run] INFO: Finished')
    self.print_timestamp()






