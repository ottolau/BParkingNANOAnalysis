#! /usr/bin/env python
import uproot
import uproot_methods
import awkward
import pandas as pd
import numpy as np
import time
from functools import partial
from helper import *
import xgboost as xgb
#import lightgbm as lgb
from BParkingNANOAnalysis.BParkingNANOAnalyzer.BaseAnalyzer import BParkingNANOAnalyzer

class BToPhiLLAnalyzer(BParkingNANOAnalyzer):
  def __init__(self, inputfiles, outputfile, hist=False, isMC=False, evalMVA=False, model='xgb', modelfile='mva.model'):
    self._isMC = isMC
    self._evalMVA = evalMVA
    self._model = model
    self._modelfile = modelfile
    inputbranches = ['nBToPhiEE', 'BToPhiEE_mll_fullfit', 'BToPhiEE_fit_phi_mass', 'BToPhiEE_fit_mass', 'BToPhiEE_fit_massErr', 'BToPhiEE_l1_idx', 'BToPhiEE_l2_idx',
                     'BToPhiEE_trk1_idx', 'BToPhiEE_trk2_idx', 'BToPhiEE_l_xy', 'BToPhiEE_l_xy_unc', 'BToPhiEE_fit_pt', 'BToPhiEE_fit_eta', 'BToPhiEE_fit_phi', 'BToPhiEE_fit_phi_pt',
                     'BToPhiEE_fit_l1_pt', 'BToPhiEE_fit_l1_eta', 'BToPhiEE_fit_l1_phi', 'BToPhiEE_fit_l2_pt', 'BToPhiEE_fit_l2_eta', 'BToPhiEE_fit_l2_phi', 
                     'BToPhiEE_fit_trk1_pt', 'BToPhiEE_fit_trk1_eta', 'BToPhiEE_fit_trk1_phi', 'BToPhiEE_fit_trk2_pt', 'BToPhiEE_fit_trk2_eta', 'BToPhiEE_fit_trk2_phi', 
                     'BToPhiEE_svprob', 'BToPhiEE_fit_cos2D', 'BToPhiEE_trk1_iso04', 'BToPhiEE_trk2_iso04',
                     'BToPhiEE_l1_iso04', 'BToPhiEE_l2_iso04', 'BToPhiEE_b_iso04', 'BToPhiEE_vtx_x', 'BToPhiEE_vtx_y', 'BToPhiEE_vtx_z',
                     'Electron_pt', 'Electron_charge', 'Electron_dxy', 'Electron_dxyErr', 'Electron_convVeto', 'Electron_isPF', 'Electron_dz',
                     'Electron_isPFoverlap', 'Electron_mvaId', 'Electron_pfmvaId',
                     'ProbeTracks_charge', 'ProbeTracks_pt', 'ProbeTracks_DCASig', 'ProbeTracks_eta', 'ProbeTracks_phi',
                     'TriggerMuon_vz', 'PV_x', 'PV_y', 'PV_z', 'event',
                     #'HLT_Mu9_IP6_*', 'PV_npvsGood',
                     ]

    inputbranches_mc = ['GenPart_pdgId', 'GenPart_genPartIdxMother', 'Electron_genPartIdx', 'ProbeTracks_genPartIdx',
                       ]
    
    outputbranches = {}
    outputbranches['BToPhiEE_mll_fullfit'] = {'nbins': 30, 'xmin': 2.6, 'xmax': 3.6}
    outputbranches['BToPhiEE_q2'] = {'nbins': 50, 'xmin': 0.0, 'xmax': 20.0}
    outputbranches['BToPhiEE_fit_phi_mass'] = {'nbins': 50, 'xmin': 0.98, 'xmax': 1.06}
    outputbranches['BToPhiEE_fit_mass'] = {'nbins': 30, 'xmin': 4.7, 'xmax': 6.0}
    outputbranches['BToPhiEE_fit_massErr'] = {'nbins': 30, 'xmin': 0.0, 'xmax': 3.0}
    outputbranches['BToPhiEE_fit_l1_pt'] = {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0}
    outputbranches['BToPhiEE_fit_l2_pt'] = {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0}
    outputbranches['BToPhiEE_fit_l1_normpt'] = {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0}
    outputbranches['BToPhiEE_fit_l2_normpt'] = {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0}
    outputbranches['BToPhiEE_fit_l1_eta'] = {'nbins': 50, 'xmin': -3.0, 'xmax': 3.0}
    outputbranches['BToPhiEE_fit_l2_eta'] = {'nbins': 50, 'xmin': -3.0, 'xmax': 3.0}
    outputbranches['BToPhiEE_fit_l1_phi'] = {'nbins': 50, 'xmin': -4.0, 'xmax': 4.0}
    outputbranches['BToPhiEE_fit_l2_phi'] = {'nbins': 50, 'xmin': -4.0, 'xmax': 4.0}
    outputbranches['BToPhiEE_l1_dxy_sig'] = {'nbins': 50, 'xmin': -30.0, 'xmax': 30.0}
    outputbranches['BToPhiEE_l2_dxy_sig'] = {'nbins': 50, 'xmin': -30.0, 'xmax': 30.0}
    outputbranches['BToPhiEE_l1_mvaId'] = {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0}
    outputbranches['BToPhiEE_l2_mvaId'] = {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0}
    outputbranches['BToPhiEE_l1_pfmvaId'] = {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0}
    outputbranches['BToPhiEE_l2_pfmvaId'] = {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0}
    outputbranches['BToPhiEE_l1_pfmvaCats'] = {'nbins': 2, 'xmin': 0.0, 'xmax': 2.0}
    outputbranches['BToPhiEE_l2_pfmvaCats'] = {'nbins': 2, 'xmin': 0.0, 'xmax': 2.0}
    outputbranches['BToPhiEE_l1_isPF'] = {'nbins': 2, 'xmin': 0, 'xmax': 2}
    outputbranches['BToPhiEE_l2_isPF'] = {'nbins': 2, 'xmin': 0, 'xmax': 2}
    outputbranches['BToPhiEE_l1_isPFoverlap'] = {'nbins': 2, 'xmin': 0, 'xmax': 2}
    outputbranches['BToPhiEE_l2_isPFoverlap'] = {'nbins': 2, 'xmin': 0, 'xmax': 2}
    outputbranches['BToPhiEE_fit_trk1_pt'] = {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0}
    outputbranches['BToPhiEE_fit_trk1_normpt'] = {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0}
    outputbranches['BToPhiEE_fit_trk1_eta'] = {'nbins': 50, 'xmin': -3.0, 'xmax': 3.0}
    outputbranches['BToPhiEE_fit_trk1_phi'] = {'nbins': 50, 'xmin': -4.0, 'xmax': 4.0}
    outputbranches['BToPhiEE_trk1_DCASig'] = {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0}
    outputbranches['BToPhiEE_fit_trk2_pt'] = {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0}
    outputbranches['BToPhiEE_fit_trk2_normpt'] = {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0}
    outputbranches['BToPhiEE_fit_trk2_eta'] = {'nbins': 50, 'xmin': -3.0, 'xmax': 3.0}
    outputbranches['BToPhiEE_fit_trk2_phi'] = {'nbins': 50, 'xmin': -4.0, 'xmax': 4.0}
    outputbranches['BToPhiEE_trk2_DCASig'] = {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0}
    outputbranches['BToPhiEE_fit_pt'] = {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0}
    outputbranches['BToPhiEE_fit_eta'] = {'nbins': 50, 'xmin': -3.0, 'xmax': 3.0}
    outputbranches['BToPhiEE_fit_phi'] = {'nbins': 50, 'xmin': -4.0, 'xmax': 4.0}
    outputbranches['BToPhiEE_fit_normpt'] = {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0}
    outputbranches['BToPhiEE_svprob'] = {'nbins': 50, 'xmin': 0.0, 'xmax': 1.0}
    outputbranches['BToPhiEE_fit_cos2D'] = {'nbins': 50, 'xmin': 0.999, 'xmax': 1.0}
    outputbranches['BToPhiEE_l_xy_sig'] = {'nbins': 50, 'xmin': 0.0, 'xmax': 50.0}
    outputbranches['BToPhiEE_dz'] = {'nbins': 50, 'xmin': -1.0, 'xmax': 1.0}
    outputbranches['BToPhiEE_ptImbalance'] = {'nbins': 50, 'xmin': -1.0, 'xmax': 1.0}
    outputbranches['BToPhiEE_eleDR'] = {'nbins': 50, 'xmin': 0.0, 'xmax': 3.0}
    outputbranches['BToPhiEE_trkDR'] = {'nbins': 50, 'xmin': 0.0, 'xmax': 3.0}
    outputbranches['BToPhiEE_llkkDR'] = {'nbins': 50, 'xmin': 0.0, 'xmax': 3.0}
    outputbranches['BToPhiEE_trk1_iso04_rel'] = {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0}
    outputbranches['BToPhiEE_trk2_iso04_rel'] = {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0}
    outputbranches['BToPhiEE_l1_iso04_rel'] = {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0}
    outputbranches['BToPhiEE_l2_iso04_rel'] = {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0}
    outputbranches['BToPhiEE_b_iso04_rel'] = {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0}
    outputbranches['BToPhiEE_eleEtaCats'] = {'nbins': 3, 'xmin': 0.0, 'xmax': 3.0}
    outputbranches['BToPhiEE_event'] = {'nbins': 10, 'xmin': 0, 'xmax': 10}
    #outputbranches['BToPhiEE_PV_npvsGood'] = {'nbins': 10, 'xmin': 0, 'xmax': 10}
    outputbranches['BToPhiEE_svprob_rank'] = {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0}
    outputbranches['BToPhiEE_fit_pt_rank'] = {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0}
    outputbranches['BToPhiEE_fit_cos2D_rank'] = {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0}
    outputbranches['BToPhiEE_l_xy_rank'] = {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0}
    outputbranches['BToPhiEE_ptAsym'] = {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0}
    outputbranches['BToPhiEE_l1_pfmvaId_lowPt'] = {'nbins': 100, 'xmin': -10.0, 'xmax': 10.0}
    outputbranches['BToPhiEE_l2_pfmvaId_lowPt'] = {'nbins': 100, 'xmin': -10.0, 'xmax': 10.0}
    outputbranches['BToPhiEE_l1_pfmvaId_highPt'] = {'nbins': 100, 'xmin': -10.0, 'xmax': 10.0}
    outputbranches['BToPhiEE_l2_pfmvaId_highPt'] = {'nbins': 100, 'xmin': -10.0, 'xmax': 10.0}
    outputbranches['BToPhiEE_k1ll_mass'] = {'nbins': 100, 'xmin': 3.0, 'xmax': 7.0}
    outputbranches['BToPhiEE_k2ll_mass'] = {'nbins': 100, 'xmin': 3.0, 'xmax': 7.0}
    outputbranches['BToPhiEE_fit_phi_pt'] = {'nbins': 100, 'xmin': 0.0, 'xmax': 50.0}
    outputbranches['BToPhiEE_fit_phi_normpt'] = {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0}
    outputbranches['BToPhiEE_fit_ll_pt'] = {'nbins': 100, 'xmin': 0.0, 'xmax': 50.0}
    outputbranches['BToPhiEE_fit_ll_normpt'] = {'nbins': 100, 'xmin': 0.0, 'xmax': 10.0}

    outputbranches_mc = {}
    outputbranches_mc['BToPhiEE_l1_genPdgId'] = {'nbins': 10, 'xmin': 0, 'xmax': 10}
    outputbranches_mc['BToPhiEE_l2_genPdgId'] = {'nbins': 10, 'xmin': 0, 'xmax': 10}
    outputbranches_mc['BToPhiEE_trk1_genPdgId'] = {'nbins': 10, 'xmin': 0, 'xmax': 10}
    outputbranches_mc['BToPhiEE_trk2_genPdgId'] = {'nbins': 10, 'xmin': 0, 'xmax': 10}
    #outputbranches_mc['BToPhiEE_decay'] = {'nbins': 10, 'xmin': 0, 'xmax': 10}

    outputbranches_mva = {}
    outputbranches_mva['BToPhiEE_mva'] =  {'nbins': 100, 'xmin': -20.0, 'xmax': 20.0}
                                 
    if self._isMC:
      inputbranches += inputbranches_mc
      outputbranches.update(outputbranches_mc)
    if self._evalMVA:
      outputbranches.update(outputbranches_mva)

    super(BToPhiLLAnalyzer, self).__init__(inputfiles, outputfile, inputbranches, outputbranches, hist)

  def run(self):
    print('[BToPhiLLAnalyzer::run] INFO: Running the analyzer...')
    self.print_timestamp()
    self.init_output()
    if self._evalMVA:
      features = ['BToPhiEE_fit_l1_normpt', 'BToPhiEE_l1_dxy_sig',
                  'BToPhiEE_fit_l2_normpt', 'BToPhiEE_l2_dxy_sig',
                  'BToPhiEE_fit_trk1_normpt', 'BToPhiEE_trk1_DCASig', 'BToPhiEE_fit_trk2_normpt', 'BToPhiEE_trk2_DCASig',
                  'BToPhiEE_fit_normpt', 'BToPhiEE_svprob', 'BToPhiEE_fit_cos2D', 'BToPhiEE_l_xy_sig', 'BToPhiEE_dz',
                  'BToPhiEE_fit_phi_normpt',
                  ]
      features += ['BToPhiEE_eleDR', 'BToPhiEE_llkkDR', 'BToPhiEE_trkDR']
      features += ['BToPhiEE_l1_iso04_rel', 'BToPhiEE_l2_iso04_rel', 'BToPhiEE_trk1_iso04_rel', 'BToPhiEE_trk2_iso04_rel', 'BToPhiEE_b_iso04_rel']
      features += ['BToPhiEE_ptImbalance']
      features += ['BToPhiEE_l1_pfmvaId_lowPt', 'BToPhiEE_l2_pfmvaId_lowPt', 'BToPhiEE_l1_pfmvaId_highPt', 'BToPhiEE_l2_pfmvaId_highPt']
      #features += ['BToPhiEE_l1_mvaId', 'BToPhiEE_l2_mvaId']

      training_branches = sorted(features)
      mvaCut = 5.0
      ntree_limit = 659
      if self._model == 'xgb':
          model = xgb.Booster({'nthread': 6})
          model.load_model(self._modelfile)
      if self._model == 'lgb':
          model = lgb.Booster(model_file=self._modelfile)

    for (self._ifile, filename) in enumerate(self._file_in_name):
      print('[BToPhiLLAnalyzer::run] INFO: FILE: {}/{}. Loading file...'.format(self._ifile+1, self._num_files))
      events = self.get_events(filename, checkbranch='nBToPhiEE')
      if events is None:
        print('Null file. Skipping file {}...'.format(filename))
        continue
      print('[BToPhiLLAnalyzer::run] INFO: FILE: {}/{}. Analyzing...'.format(self._ifile+1, self._num_files))

      startTime = time.time()
      for i, params in enumerate(events.iterate(branches=self._inputbranches, entrysteps=50000)):
        #self._branches = {key: awkward.fromiter(branch) for key, branch in params.items()} # need this line for the old version of awkward/uproot (for condor job)
        self._branches = params.copy()
        print('Reading chunk {}... Finished opening file in {} s'.format(i, time.time() - startTime))

        if self._isMC:
          # reconstruct full decay chain
          self._branches['BToPhiEE_l1_genPdgId'] = self._branches['GenPart_pdgId'][self._branches['Electron_genPartIdx'][self._branches['BToPhiEE_l1_idx']]]
          self._branches['BToPhiEE_l2_genPdgId'] = self._branches['GenPart_pdgId'][self._branches['Electron_genPartIdx'][self._branches['BToPhiEE_l2_idx']]]
          self._branches['BToPhiEE_trk1_genPdgId'] = self._branches['GenPart_pdgId'][self._branches['ProbeTracks_genPartIdx'][self._branches['BToPhiEE_trk1_idx']]]
          self._branches['BToPhiEE_trk2_genPdgId'] = self._branches['GenPart_pdgId'][self._branches['ProbeTracks_genPartIdx'][self._branches['BToPhiEE_trk2_idx']]]

          self._branches['BToPhiEE_l1_genMotherIdx'] = self._branches['GenPart_genPartIdxMother'][self._branches['Electron_genPartIdx'][self._branches['BToPhiEE_l1_idx']]]
          self._branches['BToPhiEE_l2_genMotherIdx'] = self._branches['GenPart_genPartIdxMother'][self._branches['Electron_genPartIdx'][self._branches['BToPhiEE_l2_idx']]]
          self._branches['BToPhiEE_trk1_genMotherIdx'] = self._branches['GenPart_genPartIdxMother'][self._branches['ProbeTracks_genPartIdx'][self._branches['BToPhiEE_trk1_idx']]]
          self._branches['BToPhiEE_trk2_genMotherIdx'] = self._branches['GenPart_genPartIdxMother'][self._branches['ProbeTracks_genPartIdx'][self._branches['BToPhiEE_trk2_idx']]]

          self._branches['BToPhiEE_l1_genMotherPdgId'] = self._branches['GenPart_pdgId'][self._branches['BToPhiEE_l1_genMotherIdx']]
          self._branches['BToPhiEE_l2_genMotherPdgId'] = self._branches['GenPart_pdgId'][self._branches['BToPhiEE_l2_genMotherIdx']]
          self._branches['BToPhiEE_trk1_genMotherPdgId'] = self._branches['GenPart_pdgId'][self._branches['BToPhiEE_trk1_genMotherIdx']]
          self._branches['BToPhiEE_trk2_genMotherPdgId'] = self._branches['GenPart_pdgId'][self._branches['BToPhiEE_trk2_genMotherIdx']]

          self._branches['BToPhiEE_l1Mother_genMotherIdx'] = self._branches['GenPart_genPartIdxMother'][self._branches['BToPhiEE_l1_genMotherIdx']]
          self._branches['BToPhiEE_l2Mother_genMotherIdx'] = self._branches['GenPart_genPartIdxMother'][self._branches['BToPhiEE_l2_genMotherIdx']]
          self._branches['BToPhiEE_trk1Mother_genMotherIdx'] = self._branches['GenPart_genPartIdxMother'][self._branches['BToPhiEE_trk1_genMotherIdx']]
          self._branches['BToPhiEE_trk2Mother_genMotherIdx'] = self._branches['GenPart_genPartIdxMother'][self._branches['BToPhiEE_trk2_genMotherIdx']]

          self._branches['BToPhiEE_l1Mother_genMotherPdgId'] = self._branches['GenPart_pdgId'][self._branches['BToPhiEE_l1Mother_genMotherIdx']]
          self._branches['BToPhiEE_l2Mother_genMotherPdgId'] = self._branches['GenPart_pdgId'][self._branches['BToPhiEE_l2Mother_genMotherIdx']]
          self._branches['BToPhiEE_trk1Mother_genMotherPdgId'] = self._branches['GenPart_pdgId'][self._branches['BToPhiEE_trk1Mother_genMotherIdx']]
          self._branches['BToPhiEE_trk2Mother_genMotherPdgId'] = self._branches['GenPart_pdgId'][self._branches['BToPhiEE_trk2Mother_genMotherIdx']]

        # remove cross referencing
        for branch in self._branches.keys():
          if 'Electron_' in branch:
            self._branches['BToPhiEE_l1_'+branch.replace('Electron_','')] = self._branches[branch][self._branches['BToPhiEE_l1_idx']] 
            self._branches['BToPhiEE_l2_'+branch.replace('Electron_','')] = self._branches[branch][self._branches['BToPhiEE_l2_idx']] 
            del self._branches[branch]

          if 'ProbeTracks_' in branch:
            self._branches['BToPhiEE_trk1_'+branch.replace('ProbeTracks_','')] = self._branches[branch][self._branches['BToPhiEE_trk1_idx']] 
            self._branches['BToPhiEE_trk2_'+branch.replace('ProbeTracks_','')] = self._branches[branch][self._branches['BToPhiEE_trk2_idx']] 
            del self._branches[branch]

          if 'GenPart_' in branch:
            del self._branches[branch]

          if 'HLT_Mu9_IP6_' in branch:
            self._branches['BToPhiEE_'+branch] = np.repeat(self._branches[branch], self._branches['nBToPhiEE'])
            del self._branches[branch]

          if 'TriggerMuon_' in branch:
            self._branches['BToPhiEE_trg_'+branch.replace('TriggerMuon_','')] = np.repeat(self._branches[branch][:,0], self._branches['nBToPhiEE'])
            del self._branches[branch]

          if 'PV_' in branch:
            self._branches['BToPhiEE_'+branch] = np.repeat(self._branches[branch], self._branches['nBToPhiEE'])
            del self._branches[branch]

          if branch == 'event':
            self._branches['BToPhiEE_'+branch] = np.repeat(self._branches[branch], self._branches['nBToPhiEE'])
            del self._branches[branch]

        del self._branches['nBToPhiEE']

        # flatten the jagged arrays to a normal numpy array, turn the whole dictionary to pandas dataframe
        self._branches = pd.DataFrame.from_dict({branch: array.flatten() for branch, array in self._branches.items()})
        #self._branches = awkward.topandas(self._branches, flatten=True)

        # general selection
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

        # add ranking info
        idx_pf, idx_mix, idx_low = self._branches[pf_selection].index, self._branches[mix_selection].index, self._branches[low_selection].index

        svprob_rank_pf = self._branches[pf_selection].sort_values('BToPhiEE_svprob', ascending=False).groupby('BToPhiEE_event').cumcount()
        l_xy_rank_pf = self._branches[pf_selection].sort_values('BToPhiEE_l_xy', ascending=False).groupby('BToPhiEE_event').cumcount()
        fit_cos2D_rank_pf = self._branches[pf_selection].sort_values('BToPhiEE_fit_cos2D', ascending=False).groupby('BToPhiEE_event').cumcount()
        fit_pt_rank_pf = self._branches[pf_selection].sort_values('BToPhiEE_fit_pt', ascending=False).groupby('BToPhiEE_event').cumcount()

        svprob_rank_mix = self._branches[mix_selection].sort_values('BToPhiEE_svprob', ascending=False).groupby('BToPhiEE_event').cumcount()
        l_xy_rank_mix = self._branches[mix_selection].sort_values('BToPhiEE_l_xy', ascending=False).groupby('BToPhiEE_event').cumcount()
        fit_cos2D_rank_mix = self._branches[mix_selection].sort_values('BToPhiEE_fit_cos2D', ascending=False).groupby('BToPhiEE_event').cumcount()
        fit_pt_rank_mix = self._branches[mix_selection].sort_values('BToPhiEE_fit_pt', ascending=False).groupby('BToPhiEE_event').cumcount()

        svprob_rank_low = self._branches[low_selection].sort_values('BToPhiEE_svprob', ascending=False).groupby('BToPhiEE_event').cumcount()
        l_xy_rank_low = self._branches[low_selection].sort_values('BToPhiEE_l_xy', ascending=False).groupby('BToPhiEE_event').cumcount()
        fit_cos2D_rank_low = self._branches[low_selection].sort_values('BToPhiEE_fit_cos2D', ascending=False).groupby('BToPhiEE_event').cumcount()
        fit_pt_rank_low = self._branches[low_selection].sort_values('BToPhiEE_fit_pt', ascending=False).groupby('BToPhiEE_event').cumcount()

        self._branches.loc[idx_pf, 'BToPhiEE_svprob_rank'] = svprob_rank_pf
        self._branches.loc[idx_pf, 'BToPhiEE_l_xy_rank'] = l_xy_rank_pf
        self._branches.loc[idx_pf, 'BToPhiEE_fit_cos2D_rank'] = fit_cos2D_rank_pf
        self._branches.loc[idx_pf, 'BToPhiEE_fit_pt_rank'] = fit_pt_rank_pf

        self._branches.loc[idx_mix, 'BToPhiEE_svprob_rank'] = svprob_rank_mix
        self._branches.loc[idx_mix, 'BToPhiEE_l_xy_rank'] = l_xy_rank_mix
        self._branches.loc[idx_mix, 'BToPhiEE_fit_cos2D_rank'] = fit_cos2D_rank_mix
        self._branches.loc[idx_mix, 'BToPhiEE_fit_pt_rank'] = fit_pt_rank_mix

        self._branches.loc[idx_low, 'BToPhiEE_svprob_rank'] = svprob_rank_low
        self._branches.loc[idx_low, 'BToPhiEE_l_xy_rank'] = l_xy_rank_low
        self._branches.loc[idx_low, 'BToPhiEE_fit_cos2D_rank'] = fit_cos2D_rank_low
        self._branches.loc[idx_low, 'BToPhiEE_fit_pt_rank'] = fit_pt_rank_low

        eleType_selection = pf_selection
        #eleType_selection = low_selection
        self._branches = self._branches[eleType_selection]

        # general selection
        mll_selection = (self._branches['BToPhiEE_mll_fullfit'] > NR_LOW) #& (self._branches['BToPhiEE_mll_fullfit'] < NR_UP)# all q2
        #mll_selection = (self._branches['BToPhiEE_mll_fullfit'] > NR_LOW) & (self._branches['BToPhiEE_mll_fullfit'] < PSI2S_UP) # full q2
        #mll_selection = (self._branches['BToPhiEE_mll_fullfit'] > NR_LOW) & (self._branches['BToPhiEE_mll_fullfit'] < JPSI_LOW) #low q2
        #mll_selection = (self._branches['BToPhiEE_mll_fullfit'] > JPSI_LOW) & (self._branches['BToPhiEE_mll_fullfit'] < JPSI_UP) # Jpsi
        #mll_selection = (self._branches['BToPhiEE_mll_fullfit'] > JPSI_UP) & (self._branches['BToPhiEE_mll_fullfit'] < PSI2S_UP) # psi(2S)

        #mkk_selection = (self._branches['BToPhiEE_fit_phi_mass'] > PHI_LOW) & (self._branches['BToPhiEE_fit_phi_mass'] < PHI_UP)
        mkk_selection = (self._branches['BToPhiEE_fit_phi_mass'] > 0.98) & (self._branches['BToPhiEE_fit_phi_mass'] < 1.06)

        b_upsb_selection = (self._branches['BToPhiEE_fit_mass'] > BS_UP)

        l1_selection = (self._branches['BToPhiEE_l1_convVeto']) #& (self._branches['BToPhiEE_l1_mvaId'] > 3.0)
        l2_selection = (self._branches['BToPhiEE_l2_convVeto']) #& (self._branches['BToPhiEE_l2_mvaId'] > 0.0)
        additional_selection = (self._branches['BToPhiEE_fit_mass'] > B_MIN) & (self._branches['BToPhiEE_fit_mass'] < B_MAX)

        selection = l1_selection & l2_selection
        selection &= mll_selection
        selection &= mkk_selection
        selection &= additional_selection
        #selection &= b_upsb_selection

        if self._isMC:
          selection &= (self._branches['BToPhiEE_l1_genPartIdx'] > -0.5) & (self._branches['BToPhiEE_l2_genPartIdx'] > -0.5) & (self._branches['BToPhiEE_trk1_genPartIdx'] > -0.5) & (self._branches['BToPhiEE_trk2_genPartIdx'] > -0.5)

        self._branches = self._branches[selection]
        
        if not self._branches.empty:          
          # add additional branches
          self._branches['BToPhiEE_l_xy_sig'] = self._branches['BToPhiEE_l_xy'] / self._branches['BToPhiEE_l_xy_unc']
          self._branches['BToPhiEE_l1_dxy_sig'] = self._branches['BToPhiEE_l1_dxy'] / self._branches['BToPhiEE_l1_dxyErr']
          self._branches['BToPhiEE_l2_dxy_sig'] = self._branches['BToPhiEE_l2_dxy'] / self._branches['BToPhiEE_l2_dxyErr']
          self._branches['BToPhiEE_fit_l1_normpt'] = self._branches['BToPhiEE_fit_l1_pt'] / self._branches['BToPhiEE_fit_mass']
          self._branches['BToPhiEE_fit_l2_normpt'] = self._branches['BToPhiEE_fit_l2_pt'] / self._branches['BToPhiEE_fit_mass']
          self._branches['BToPhiEE_fit_trk1_normpt'] = self._branches['BToPhiEE_fit_trk1_pt'] / self._branches['BToPhiEE_fit_mass']
          self._branches['BToPhiEE_fit_trk2_normpt'] = self._branches['BToPhiEE_fit_trk2_pt'] / self._branches['BToPhiEE_fit_mass']
          self._branches['BToPhiEE_fit_phi_normpt'] = self._branches['BToPhiEE_fit_phi_pt'] / self._branches['BToPhiEE_fit_mass']
          self._branches['BToPhiEE_fit_normpt'] = self._branches['BToPhiEE_fit_pt'] / self._branches['BToPhiEE_fit_mass']
          self._branches['BToPhiEE_q2'] = self._branches['BToPhiEE_mll_fullfit'] * self._branches['BToPhiEE_mll_fullfit']
          self._branches['BToPhiEE_b_iso04_rel'] = self._branches['BToPhiEE_b_iso04'] / self._branches['BToPhiEE_fit_pt']
          self._branches['BToPhiEE_l1_iso04_rel'] = self._branches['BToPhiEE_l1_iso04'] / self._branches['BToPhiEE_fit_l1_pt']
          self._branches['BToPhiEE_l2_iso04_rel'] = self._branches['BToPhiEE_l2_iso04'] / self._branches['BToPhiEE_fit_l2_pt']
          self._branches['BToPhiEE_trk1_iso04_rel'] = self._branches['BToPhiEE_trk1_iso04'] / self._branches['BToPhiEE_fit_trk1_pt']
          self._branches['BToPhiEE_trk2_iso04_rel'] = self._branches['BToPhiEE_trk2_iso04'] / self._branches['BToPhiEE_fit_trk2_pt']
          self._branches['BToPhiEE_eleEtaCats'] = map(self.EleEtaCats, self._branches['BToPhiEE_fit_l1_eta'], self._branches['BToPhiEE_fit_l2_eta'])
          self._branches['BToPhiEE_dz'] = self._branches['BToPhiEE_vtx_z'] - self._branches['BToPhiEE_trg_vz']
          #self._branches['BToPhiEE_dz'] = self._branches['BToPhiEE_l1_dz']
          self._branches['BToPhiEE_l1_pfmvaCats'] = np.where(self._branches['BToPhiEE_l1_pt'] < 5.0, 0, 1)
          self._branches['BToPhiEE_l2_pfmvaCats'] = np.where(self._branches['BToPhiEE_l2_pt'] < 5.0, 0, 1)
          self._branches['BToPhiEE_l1_pfmvaId_lowPt'] = np.where(self._branches['BToPhiEE_l1_pfmvaCats'] == 0, self._branches['BToPhiEE_l1_pfmvaId'], 20.0)
          self._branches['BToPhiEE_l2_pfmvaId_lowPt'] = np.where(self._branches['BToPhiEE_l2_pfmvaCats'] == 0, self._branches['BToPhiEE_l2_pfmvaId'], 20.0)
          self._branches['BToPhiEE_l1_pfmvaId_highPt'] = np.where(self._branches['BToPhiEE_l1_pfmvaCats'] == 1, self._branches['BToPhiEE_l1_pfmvaId'], 20.0)
          self._branches['BToPhiEE_l2_pfmvaId_highPt'] = np.where(self._branches['BToPhiEE_l2_pfmvaCats'] == 1, self._branches['BToPhiEE_l2_pfmvaId'], 20.0)


          if self._isMC:
            self._branches['BToPhiEE_decay'] = map(self.DecayCats_BToPhiLL_vectorized, self._branches['BToPhiEE_l1_genPartIdx'], self._branches['BToPhiEE_l2_genPartIdx'],
                                                 self._branches['BToPhiEE_trk1_genPartIdx'], self._branches['BToPhiEE_trk2_genPartIdx'],
                                                 self._branches['BToPhiEE_l1_genPdgId'], self._branches['BToPhiEE_l2_genPdgId'], 
                                                 self._branches['BToPhiEE_trk1_genPdgId'], self._branches['BToPhiEE_trk2_genPdgId'],
                                                 self._branches['BToPhiEE_l1_genMotherPdgId'], self._branches['BToPhiEE_l2_genMotherPdgId'],
                                                 self._branches['BToPhiEE_trk1_genMotherPdgId'], self._branches['BToPhiEE_trk2_genMotherPdgId'],
                                                 self._branches['BToPhiEE_l1Mother_genMotherPdgId'], self._branches['BToPhiEE_l2Mother_genMotherPdgId'],
                                                 self._branches['BToPhiEE_trk1Mother_genMotherPdgId'], self._branches['BToPhiEE_trk2Mother_genMotherPdgId'])

            #self._branches.query('BToPhiEE_decay == 0', inplace=True) # B->Phi ll
            self._branches.query('BToPhiEE_decay == 1', inplace=True) # B->Phi J/psi(ll)

          # mass hypothesis to veto fake event from semi-leptonic decay D
          l1_p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(self._branches['BToPhiEE_fit_l1_pt'], self._branches['BToPhiEE_fit_l1_eta'], self._branches['BToPhiEE_fit_l1_phi'], ELECTRON_MASS)
          l2_p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(self._branches['BToPhiEE_fit_l2_pt'], self._branches['BToPhiEE_fit_l2_eta'], self._branches['BToPhiEE_fit_l2_phi'], ELECTRON_MASS)
          trk1_p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(self._branches['BToPhiEE_fit_trk1_pt'], self._branches['BToPhiEE_fit_trk1_eta'], self._branches['BToPhiEE_fit_trk1_phi'], K_MASS)
          trk2_p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(self._branches['BToPhiEE_fit_trk2_pt'], self._branches['BToPhiEE_fit_trk2_eta'], self._branches['BToPhiEE_fit_trk2_phi'], K_MASS)

          self._branches['BToPhiEE_eleDR'] = l1_p4.delta_r(l2_p4)
          self._branches['BToPhiEE_trkDR'] = trk1_p4.delta_r(trk2_p4)
          self._branches['BToPhiEE_llkkDR'] = (l1_p4 + l2_p4).delta_r(trk1_p4 + trk2_p4)
          self._branches['BToPhiEE_k1ll_mass'] = (l1_p4 + l2_p4 + trk1_p4).mass
          self._branches['BToPhiEE_k2ll_mass'] = (l1_p4 + l2_p4 + trk2_p4).mass
          self._branches['BToPhiEE_fit_ll_pt'] = (l1_p4 + l2_p4).pt
          self._branches['BToPhiEE_fit_ll_normpt'] = self._branches['BToPhiEE_fit_ll_pt'] / self._branches['BToPhiEE_fit_mass']

          diele_p3 = (l1_p4 + l2_p4).p3
          phi_p3 = (trk1_p4 + trk2_p4).p3
          pv2sv_p3 = uproot_methods.TVector3Array.from_cartesian(self._branches['BToPhiEE_PV_x'] - self._branches['BToPhiEE_vtx_x'], self._branches['BToPhiEE_PV_y'] - self._branches['BToPhiEE_vtx_y'], self._branches['BToPhiEE_PV_z'] - self._branches['BToPhiEE_vtx_z'])
          self._branches['BToPhiEE_ptImbalance'] = np.array([p1.cross(p2).mag for p1, p2 in zip(diele_p3, pv2sv_p3)]) / np.array([p1.cross(p2).mag for p1, p2 in zip(phi_p3, pv2sv_p3)])
          self._branches['BToPhiEE_ptAsym'] = (np.array([p1.cross(p2).mag for p1, p2 in zip(diele_p3, pv2sv_p3)]) - np.array([p1.cross(p2).mag for p1, p2 in zip(phi_p3, pv2sv_p3)])) / (np.array([p1.cross(p2).mag for p1, p2 in zip(diele_p3, pv2sv_p3)]) + np.array([p1.cross(p2).mag for p1, p2 in zip(phi_p3, pv2sv_p3)])) 


          if self._evalMVA:
            if self._model == 'xgb':
              self._branches['BToPhiEE_mva'] = model.predict(xgb.DMatrix(self._branches[training_branches].replace([np.inf, -np.inf], 0.0).sort_index(axis=1)), ntree_limit=ntree_limit)
            if self._model == 'lgb':
              self._branches['BToPhiEE_mva'] = model.predict(self._branches[training_branches].replace([np.inf, -np.inf], 0.0).sort_index(axis=1), raw_score=True)
            #self._branches = self._branches[(self._branches['BToPhiEE_xgb'] > mvaCut)].sort_values('BToPhiEE_xgb', ascending=False).drop_duplicates(['BToPhiEE_event'], keep='first')
            self._branches = self._branches[(self._branches['BToPhiEE_mva'] > mvaCut)]

          # fill output
          self.fill_output()

        startTime = time.time()

    self.finish()
    print('[BToPhiLLAnalyzer::run] INFO: Finished')
    self.print_timestamp()






