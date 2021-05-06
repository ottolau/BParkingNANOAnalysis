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

class BToKLLAnalyzer(BParkingNANOAnalyzer):
  def __init__(self, inputfiles, outputfile, fold=-1, random=False, isMC=False, evalMVA=False, model='xgb', modelfile='mva.model'):
    self._isMC = isMC
    self._fold = fold
    self._random = random
    self._evalMVA = evalMVA
    self._model = model
    self._modelfile = modelfile
    inputbranches = []
    inputbranches += ['nBToKEE', 'BToKEE_mll_fullfit', 'BToKEE_fit_mass', 'BToKEE_fit_massErr']
    inputbranches += ['BToKEE_l1Idx', 'BToKEE_l2Idx', 'BToKEE_kIdx']
    inputbranches += ['BToKEE_fit_pt', 'BToKEE_fit_eta', 'BToKEE_fit_phi']
    inputbranches += ['BToKEE_fit_l1_pt', 'BToKEE_fit_l1_eta', 'BToKEE_fit_l1_phi', 'BToKEE_fit_l2_pt', 'BToKEE_fit_l2_eta', 'BToKEE_fit_l2_phi']
    inputbranches += ['BToKEE_fit_k_pt', 'BToKEE_fit_k_eta', 'BToKEE_fit_k_phi']
    inputbranches += ['BToKEE_k_svip2d', 'BToKEE_k_svip3d']
    inputbranches += ['BToKEE_l_xy', 'BToKEE_l_xy_unc', 'BToKEE_svprob', 'BToKEE_fit_cos2D']
    inputbranches += ['BToKEE_vtx_x', 'BToKEE_vtx_y', 'BToKEE_vtx_z']
    inputbranches += ['BToKEE_l1_iso04', 'BToKEE_l2_iso04', 'BToKEE_k_iso04', 'BToKEE_b_iso04']
    inputbranches += ['BToKEE_l1_n_isotrk', 'BToKEE_l2_n_isotrk', 'BToKEE_k_n_isotrk', 'BToKEE_b_n_isotrk']
    inputbranches += ['Electron_pt', 'Electron_charge', 'Electron_dxy', 'Electron_dxyErr', 'Electron_convVeto', 'Electron_isPF']
    inputbranches += ['Electron_isPFoverlap', 'Electron_mvaId', 'Electron_pfmvaId']
    inputbranches += ['Electron_convOpen', 'Electron_convLoose', 'Electron_convTight']
    inputbranches += ['ProbeTracks_charge', 'ProbeTracks_pt', 'ProbeTracks_DCASig', 'ProbeTracks_eta', 'ProbeTracks_phi', 'ProbeTracks_nValidHits']
    inputbranches += ['Electron_dzTrg', 'ProbeTracks_dzTrg']
    inputbranches += ['PV_x', 'PV_y', 'PV_z', 'event', 'HLT_Mu9_IP6*']
    inputbranches += ['BToKEE_pt']

    inputbranches_mc = []
    inputbranches_mc += ['GenPart_pdgId', 'GenPart_genPartIdxMother', 'Electron_genPartIdx', 'ProbeTracks_genPartIdx']
    
    outputbranches = []
    outputbranches += ['BToKEE_mll_fullfit', 'BToKEE_q2', 'BToKEE_fit_mass', 'BToKEE_fit_massErr']
    outputbranches += ['BToKEE_fit_l1_pt', 'BToKEE_fit_l1_normpt', 'BToKEE_fit_l1_eta', 'BToKEE_fit_l1_phi']
    outputbranches += ['BToKEE_l1_dxy_sig']
    outputbranches += ['BToKEE_l1_mvaId', 'BToKEE_l1_pfmvaId', 'BToKEE_l1_pfmvaCats', 'BToKEE_l1_isPF', 'BToKEE_l1_isPFoverlap']
    outputbranches += ['BToKEE_l1_pfmvaId_lowPt', 'BToKEE_l1_pfmvaId_highPt']
    outputbranches += ['BToKEE_l1_convOpen', 'BToKEE_l1_convLoose', 'BToKEE_l1_convTight']
    outputbranches += ['BToKEE_fit_l2_pt', 'BToKEE_fit_l2_normpt', 'BToKEE_fit_l2_eta', 'BToKEE_fit_l2_phi']
    outputbranches += ['BToKEE_l2_dxy_sig']
    outputbranches += ['BToKEE_l2_mvaId', 'BToKEE_l2_pfmvaId', 'BToKEE_l2_pfmvaCats', 'BToKEE_l2_isPF', 'BToKEE_l2_isPFoverlap']
    outputbranches += ['BToKEE_l2_pfmvaId_lowPt', 'BToKEE_l2_pfmvaId_highPt']
    outputbranches += ['BToKEE_l2_convOpen', 'BToKEE_l2_convLoose', 'BToKEE_l2_convTight']
    outputbranches += ['BToKEE_fit_k_pt', 'BToKEE_fit_k_normpt', 'BToKEE_fit_k_eta', 'BToKEE_fit_k_phi']
    outputbranches += ['BToKEE_k_DCASig', 'BToKEE_k_nValidHits']
    outputbranches += ['BToKEE_k_svip2d', 'BToKEE_k_svip3d']
    outputbranches += ['BToKEE_fit_pt', 'BToKEE_fit_normpt', 'BToKEE_fit_eta', 'BToKEE_fit_phi']
    outputbranches += ['BToKEE_l_xy_sig', 'BToKEE_svprob', 'BToKEE_fit_cos2D']
    outputbranches += ['BToKEE_ptAsym', 'BToKEE_ptImbalance', 'BToKEE_Dmass', 'BToKEE_Dmass_flip']
    outputbranches += ['BToKEE_eleDR', 'BToKEE_llkDR']
    outputbranches += ['BToKEE_l1_iso04_rel', 'BToKEE_l2_iso04_rel', 'BToKEE_k_iso04_rel', 'BToKEE_b_iso04_rel']
    outputbranches += ['BToKEE_l1_n_isotrk', 'BToKEE_l2_n_isotrk', 'BToKEE_k_n_isotrk', 'BToKEE_b_n_isotrk']
    outputbranches += ['BToKEE_eleEtaCats', 'BToKEE_event']
    outputbranches += ['BToKEE_l1_dzTrg', 'BToKEE_l2_dzTrg', 'BToKEE_k_dzTrg']
    outputbranches += ['BToKEE_fold',]
    outputbranches += ['BToKEE_pt', 'BToKEE_l1_pt', 'BToKEE_l2_pt', 'BToKEE_k_pt']
    outputbranches += ['BToKEE_nCand']
    outputbranches += ['BToKEE_Dmass_ll', 'BToKEE_Dmass_ll_flip']

    outputbranches_mc = []
    outputbranches_mc += ['BToKEE_l1_isGen', 'BToKEE_l2_isGen', 'BToKEE_k_isGen', 'BToKEE_l1_genPdgId', 'BToKEE_l2_genPdgId', 'BToKEE_k_genPdgId']
    outputbranches_mc += ['BToKEE_k_isKaon', 'BToKEE_HLT_Mu9_IP6']

    outputbranches_mva = ['BToKEE_mva',]
                                 
    if self._isMC:
      inputbranches += inputbranches_mc
      outputbranches += outputbranches_mc
    if self._evalMVA:
      outputbranches += outputbranches_mva

    super(BToKLLAnalyzer, self).__init__(inputfiles, outputfile, inputbranches, outputbranches)

  def run(self):
    print('[BToKLLAnalyzer::run] INFO: Running the analyzer...')
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
      #features += ['BToKEE_l1_mvaId', 'BToKEE_l2_mvaId']
      features += ['BToKEE_l1_dzTrg', 'BToKEE_l2_dzTrg', 'BToKEE_k_dzTrg']
      features += ['BToKEE_k_svip2d', 'BToKEE_k_svip3d']
      #features += ['BToKEE_Dmass', 'BToKEE_Dmass_flip']

      training_branches = sorted(features)
      mvaCut = 5.0
      ntree_limit = 774
      model = xgb.Booster({'nthread': 6})
      model.load_model(self._modelfile)

    for (self._ifile, filename) in enumerate(self._file_in_name):
      print('[BToKLLAnalyzer::run] INFO: FILE: {}/{}. Loading file...'.format(self._ifile+1, self._num_files))
      print('[BToKLLAnalyzer::run] INFO: FILE: file nmae: {}'.format(filename))
      events = self.get_events(filename, checkbranch='nBToKEE')
      if events is None:
        print('Null file. Skipping file {}...'.format(filename))
        continue
      print('[BToKLLAnalyzer::run] INFO: FILE: {}/{}. Analyzing...'.format(self._ifile+1, self._num_files))

      startTime = time.time()
      for i, params in enumerate(events.iterate(branches=self._inputbranches, entrysteps=50000)):
        #self._branches = {key: awkward.fromiter(branch) for key, branch in params.items()} # need this line for the old version of awkward/uproot (for condor job)
        self._branches = params.copy()
        print('Reading chunk {}... Finished opening file in {} s'.format(i, time.time() - startTime))

        if self._isMC:
          # reconstruct full decay chain
          self._branches['BToKEE_l1_genPdgId'] = self._branches['GenPart_pdgId'][self._branches['Electron_genPartIdx'][self._branches['BToKEE_l1Idx']]]
          self._branches['BToKEE_l2_genPdgId'] = self._branches['GenPart_pdgId'][self._branches['Electron_genPartIdx'][self._branches['BToKEE_l2Idx']]]
          self._branches['BToKEE_k_genPdgId'] = self._branches['GenPart_pdgId'][self._branches['ProbeTracks_genPartIdx'][self._branches['BToKEE_kIdx']]]

          self._branches['BToKEE_l1_genMotherIdx'] = self._branches['GenPart_genPartIdxMother'][self._branches['Electron_genPartIdx'][self._branches['BToKEE_l1Idx']]]
          self._branches['BToKEE_l2_genMotherIdx'] = self._branches['GenPart_genPartIdxMother'][self._branches['Electron_genPartIdx'][self._branches['BToKEE_l2Idx']]]
          self._branches['BToKEE_k_genMotherIdx'] = self._branches['GenPart_genPartIdxMother'][self._branches['ProbeTracks_genPartIdx'][self._branches['BToKEE_kIdx']]]

          self._branches['BToKEE_l1_genMotherPdgId'] = self._branches['GenPart_pdgId'][self._branches['BToKEE_l1_genMotherIdx']]
          self._branches['BToKEE_l2_genMotherPdgId'] = self._branches['GenPart_pdgId'][self._branches['BToKEE_l2_genMotherIdx']]
          self._branches['BToKEE_k_genMotherPdgId'] = self._branches['GenPart_pdgId'][self._branches['BToKEE_k_genMotherIdx']]

          self._branches['BToKEE_l1Mother_genMotherIdx'] = self._branches['GenPart_genPartIdxMother'][self._branches['BToKEE_l1_genMotherIdx']]
          self._branches['BToKEE_l2Mother_genMotherIdx'] = self._branches['GenPart_genPartIdxMother'][self._branches['BToKEE_l2_genMotherIdx']]
          self._branches['BToKEE_kMother_genMotherIdx'] = self._branches['GenPart_genPartIdxMother'][self._branches['BToKEE_k_genMotherIdx']]

          self._branches['BToKEE_l1Mother_genMotherPdgId'] = self._branches['GenPart_pdgId'][self._branches['BToKEE_l1Mother_genMotherIdx']]
          self._branches['BToKEE_l2Mother_genMotherPdgId'] = self._branches['GenPart_pdgId'][self._branches['BToKEE_l2Mother_genMotherIdx']]
          self._branches['BToKEE_kMother_genMotherPdgId'] = self._branches['GenPart_pdgId'][self._branches['BToKEE_kMother_genMotherIdx']]


        # remove cross referencing
        for branch in self._branches.keys():
          if 'Electron_' in branch:
            self._branches['BToKEE_l1_'+branch.replace('Electron_','')] = self._branches[branch][self._branches['BToKEE_l1Idx']] 
            self._branches['BToKEE_l2_'+branch.replace('Electron_','')] = self._branches[branch][self._branches['BToKEE_l2Idx']] 
            del self._branches[branch]

          if 'ProbeTracks_' in branch:
            self._branches['BToKEE_k_'+branch.replace('ProbeTracks_','')] = self._branches[branch][self._branches['BToKEE_kIdx']] 
            del self._branches[branch]

          if 'GenPart_' in branch:
            del self._branches[branch]

          if 'HLT_Mu9_IP6' in branch:
            self._branches['BToKEE_'+branch] = np.repeat(self._branches[branch], self._branches['nBToKEE'])
            del self._branches[branch]

          if 'TriggerMuon_' in branch:
            self._branches['BToKEE_trg_'+branch.replace('TriggerMuon_','')] = np.repeat(self._branches[branch][:,0], self._branches['nBToKEE'])
            del self._branches[branch]

          if 'PV_' in branch:
            self._branches['BToKEE_'+branch] = np.repeat(self._branches[branch], self._branches['nBToKEE'])
            del self._branches[branch]

          if branch == 'event':
            self._branches['BToKEE_'+branch] = np.repeat(self._branches[branch], self._branches['nBToKEE'])
            del self._branches[branch]

        del self._branches['nBToKEE']

        # flatten the jagged arrays to a normal numpy array, turn the whole dictionary to pandas dataframe
        self._branches = pd.DataFrame.from_dict({branch: array.flatten() for branch, array in self._branches.items()})
        #self._branches = awkward.topandas(self._branches, flatten=True)

        # general selection
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

        # add ranking info
        #idx_pf, idx_mix, idx_low = self._branches[pf_selection].index, self._branches[mix_selection].index, self._branches[low_selection].index
        #
        #svprob_rank_pf = self._branches[pf_selection].sort_values('BToKEE_svprob', ascending=False).groupby('BToKEE_event').cumcount()
        #l_xy_rank_pf = self._branches[pf_selection].sort_values('BToKEE_l_xy', ascending=False).groupby('BToKEE_event').cumcount()
        #fit_cos2D_rank_pf = self._branches[pf_selection].sort_values('BToKEE_fit_cos2D', ascending=False).groupby('BToKEE_event').cumcount()
        #fit_pt_rank_pf = self._branches[pf_selection].sort_values('BToKEE_fit_pt', ascending=False).groupby('BToKEE_event').cumcount()
        #
        #svprob_rank_mix = self._branches[mix_selection].sort_values('BToKEE_svprob', ascending=False).groupby('BToKEE_event').cumcount()
        #l_xy_rank_mix = self._branches[mix_selection].sort_values('BToKEE_l_xy', ascending=False).groupby('BToKEE_event').cumcount()
        #fit_cos2D_rank_mix = self._branches[mix_selection].sort_values('BToKEE_fit_cos2D', ascending=False).groupby('BToKEE_event').cumcount()
        #fit_pt_rank_mix = self._branches[mix_selection].sort_values('BToKEE_fit_pt', ascending=False).groupby('BToKEE_event').cumcount()
        #
        #svprob_rank_low = self._branches[low_selection].sort_values('BToKEE_svprob', ascending=False).groupby('BToKEE_event').cumcount()
        #l_xy_rank_low = self._branches[low_selection].sort_values('BToKEE_l_xy', ascending=False).groupby('BToKEE_event').cumcount()
        #fit_cos2D_rank_low = self._branches[low_selection].sort_values('BToKEE_fit_cos2D', ascending=False).groupby('BToKEE_event').cumcount()
        #fit_pt_rank_low = self._branches[low_selection].sort_values('BToKEE_fit_pt', ascending=False).groupby('BToKEE_event').cumcount()
        #
        #self._branches.loc[idx_pf, 'BToKEE_svprob_rank'] = svprob_rank_pf
        #self._branches.loc[idx_pf, 'BToKEE_l_xy_rank'] = l_xy_rank_pf
        #self._branches.loc[idx_pf, 'BToKEE_fit_cos2D_rank'] = fit_cos2D_rank_pf
        #self._branches.loc[idx_pf, 'BToKEE_fit_pt_rank'] = fit_pt_rank_pf
        #
        #self._branches.loc[idx_mix, 'BToKEE_svprob_rank'] = svprob_rank_mix
        #self._branches.loc[idx_mix, 'BToKEE_l_xy_rank'] = l_xy_rank_mix
        #self._branches.loc[idx_mix, 'BToKEE_fit_cos2D_rank'] = fit_cos2D_rank_mix
        #self._branches.loc[idx_mix, 'BToKEE_fit_pt_rank'] = fit_pt_rank_mix
        #
        #self._branches.loc[idx_low, 'BToKEE_svprob_rank'] = svprob_rank_low
        #self._branches.loc[idx_low, 'BToKEE_l_xy_rank'] = l_xy_rank_low
        #self._branches.loc[idx_low, 'BToKEE_fit_cos2D_rank'] = fit_cos2D_rank_low
        #self._branches.loc[idx_low, 'BToKEE_fit_pt_rank'] = fit_pt_rank_low

        eleType_selection = pf_selection
        #eleType_selection = low_selection
        #eleType_selection = mix_net_selection
        #eleType_selection = low_pfveto_selection
        self._branches = self._branches[eleType_selection]

        self._branches['BToKEE_l1_pfmvaCats'] = np.where(self._branches['BToKEE_l1_pt'] < 5.0, 0, 1)
        self._branches['BToKEE_l2_pfmvaCats'] = np.where(self._branches['BToKEE_l2_pt'] < 5.0, 0, 1)
        self._branches['BToKEE_l1_pfmvaId_lowPt'] = np.where(self._branches['BToKEE_l1_pfmvaCats'] == 0, self._branches['BToKEE_l1_pfmvaId'], 20.0)
        self._branches['BToKEE_l2_pfmvaId_lowPt'] = np.where(self._branches['BToKEE_l2_pfmvaCats'] == 0, self._branches['BToKEE_l2_pfmvaId'], 20.0)
        self._branches['BToKEE_l1_pfmvaId_highPt'] = np.where(self._branches['BToKEE_l1_pfmvaCats'] == 1, self._branches['BToKEE_l1_pfmvaId'], 20.0)
        self._branches['BToKEE_l2_pfmvaId_highPt'] = np.where(self._branches['BToKEE_l2_pfmvaCats'] == 1, self._branches['BToKEE_l2_pfmvaId'], 20.0)

        l1_pihypo_p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(self._branches['BToKEE_fit_l1_pt'], self._branches['BToKEE_fit_l1_eta'], self._branches['BToKEE_fit_l1_phi'], PI_MASS)
        l2_pihypo_p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(self._branches['BToKEE_fit_l2_pt'], self._branches['BToKEE_fit_l2_eta'], self._branches['BToKEE_fit_l2_phi'], PI_MASS)
        l1_khypo_p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(self._branches['BToKEE_fit_l1_pt'], self._branches['BToKEE_fit_l1_eta'], self._branches['BToKEE_fit_l1_phi'], K_MASS)
        l2_khypo_p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(self._branches['BToKEE_fit_l2_pt'], self._branches['BToKEE_fit_l2_eta'], self._branches['BToKEE_fit_l2_phi'], K_MASS)
        k_pihypo_p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(self._branches['BToKEE_fit_k_pt'], self._branches['BToKEE_fit_k_eta'], self._branches['BToKEE_fit_k_phi'], PI_MASS)
        k_p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(self._branches['BToKEE_fit_k_pt'], self._branches['BToKEE_fit_k_eta'], self._branches['BToKEE_fit_k_phi'], K_MASS)

        self._branches['BToKEE_Dmass_l1'] = (l1_pihypo_p4 + k_p4).mass
        self._branches['BToKEE_Dmass_l2'] = (l2_pihypo_p4 + k_p4).mass
        self._branches['BToKEE_Dmass_flip_l1'] = (l1_khypo_p4 + k_pihypo_p4).mass
        self._branches['BToKEE_Dmass_flip_l2'] = (l2_khypo_p4 + k_pihypo_p4).mass
        self._branches['BToKEE_Dmass'] = np.where((self._branches['BToKEE_k_charge'] * self._branches['BToKEE_l1_charge']) < 0.0, self._branches['BToKEE_Dmass_l1'], self._branches['BToKEE_Dmass_l2'])
        self._branches['BToKEE_Dmass_flip'] = np.where((self._branches['BToKEE_k_charge'] * self._branches['BToKEE_l1_charge']) < 0.0, self._branches['BToKEE_Dmass_flip_l1'], self._branches['BToKEE_Dmass_flip_l2'])
        self._branches['BToKEE_Dmass_ll'] = (l1_khypo_p4 + l2_pihypo_p4).mass
        self._branches['BToKEE_Dmass_ll_flip'] = (l1_pihypo_p4 + l2_khypo_p4).mass

        #self._branches = self._branches.sort_values('BToKEE_fit_pt', ascending=False).groupby('BToKEE_event').head(2)

        # general selection
        mll_selection = (self._branches['BToKEE_mll_fullfit'] > LOWQ2_LOW) #& (self._branches['BToKEE_mll_fullfit'] < HIGHQ2_UP)# all q2
        #mll_selection = (self._branches['BToKEE_mll_fullfit'] > LOWQ2_LOW) & (self._branches['BToKEE_mll_fullfit'] < LOWQ2_UP) #low q2
        #mll_selection = (self._branches['BToKEE_mll_fullfit'] > JPSI_LOW) & (self._branches['BToKEE_mll_fullfit'] < JPSI_UP) # Jpsi
        #mll_selection = (self._branches['BToKEE_mll_fullfit'] > JPSI_UP) & (self._branches['BToKEE_mll_fullfit'] < PSI2S_UP) # psi(2S)
        #mll_selection = (self._branches['BToKEE_mll_fullfit'] > NR_LOW) & (self._branches['BToKEE_mll_fullfit'] < JPSI_UP)
        #mll_selection = (self._branches['BToKEE_mll_fullfit'] > np.sqrt(1.1)) & (self._branches['BToKEE_mll_fullfit'] < LOWQ2_UP) #low q2, RK = 1
        #mll_selection = (self._branches['BToKEE_mll_fullfit'] > np.sqrt(1.1)) & (self._branches['BToKEE_mll_fullfit'] < JPSI_UP) #low q2, RK = 1 and Jpsi

        b_upsb_selection = (self._branches['BToKEE_fit_mass'] > B_UP)
        b_bothsb_selection = ( (self._branches['BToKEE_fit_mass'] > 4.75) & (self._branches['BToKEE_fit_mass'] < 5.0) ) | ( (self._branches['BToKEE_fit_mass'] > 5.45) & (self._branches['BToKEE_fit_mass'] < 5.7) )

        l1_selection = (self._branches['BToKEE_l1_convVeto']) #& (self._branches['BToKEE_l1_mvaId'] > 2.0) #& (self._branches['BToKEE_l1_mvaId'] > 2.0)
        l2_selection = (self._branches['BToKEE_l2_convVeto']) #& (self._branches['BToKEE_l2_mvaId'] > -1.0) #& (self._branches['BToKEE_l2_mvaId'] > -3.0) 
        additional_selection = (self._branches['BToKEE_fit_mass'] > B_MIN) & (self._branches['BToKEE_fit_mass'] < B_MAX)

        selection = l1_selection & l2_selection
        selection &= mll_selection
        selection &= additional_selection
        #selection &= b_upsb_selection
        #selection &= b_bothsb_selection
        #selection &= (self._branches['BToKEE_Dmass'] > 2.0) & (self._branches['BToKEE_Dmass_flip'] > 2.0)
        #selection &= (abs(self._branches['BToKEE_k_svip3d']) < 0.06) & (self._branches['BToKEE_fit_cos2D'] > 0.95) & (self._branches['BToKEE_Dmass'] > 2.0) & (self._branches['BToKEE_Dmass_flip'] > 2.0)
        selection &= (abs(self._branches['BToKEE_k_svip3d']) < 0.06) & (self._branches['BToKEE_fit_cos2D'] > 0.95) 

        if self._isMC:
          selection &= (self._branches['BToKEE_HLT_Mu9_IP6'])
          selection &= (self._branches['BToKEE_l1_genPartIdx'] > -0.5) & (self._branches['BToKEE_l2_genPartIdx'] > -0.5) & (self._branches['BToKEE_k_genPartIdx'] > -0.5)

        self._branches = self._branches[selection]

        if not self._branches.empty:          
          if self._isMC:
            self._branches['BToKEE_k_isKaon'] = np.where(abs(self._branches['BToKEE_k_genPdgId']) == 321, True, False)
            self._branches['BToKEE_decay'] = map(self.DecayCats_vectorized, self._branches['BToKEE_l1_genPartIdx'], self._branches['BToKEE_l2_genPartIdx'], self._branches['BToKEE_k_genPartIdx'],
                                                 self._branches['BToKEE_l1_genPdgId'], self._branches['BToKEE_l2_genPdgId'], self._branches['BToKEE_k_genPdgId'],
                                                 self._branches['BToKEE_l1_genMotherPdgId'], self._branches['BToKEE_l2_genMotherPdgId'], self._branches['BToKEE_k_genMotherPdgId'],
                                                 self._branches['BToKEE_l1Mother_genMotherPdgId'], self._branches['BToKEE_l2Mother_genMotherPdgId'], self._branches['BToKEE_kMother_genMotherPdgId'])
            #self._branches['BToKEE_decay'] = self._branches.apply(self.DecayCats, axis=1, prefix='BToKEE')

            #self._branches.query('BToKEE_decay == 0', inplace=True) # B->K ll
            self._branches.query('BToKEE_decay == 1', inplace=True) # B->K J/psi(ll)
            #self._branches.query('BToKEE_decay == 2', inplace=True) # B->K*(K pi) ll
            #self._branches.query('BToKEE_decay == 3', inplace=True) # B->K*(K pi) J/psi(ll)
            self._branches['BToKEE_l1_isGen'] = np.where((self._branches['BToKEE_l1_genPartIdx'] > -0.5) & (abs(self._branches['BToKEE_l1_genMotherPdgId']) == 521), True, False)
            self._branches['BToKEE_l2_isGen'] = np.where((self._branches['BToKEE_l2_genPartIdx'] > -0.5) & (abs(self._branches['BToKEE_l2_genMotherPdgId']) == 521), True, False)
            self._branches['BToKEE_k_isGen'] = np.where((self._branches['BToKEE_k_genPartIdx'] > -0.5) & (abs(self._branches['BToKEE_k_genMotherPdgId']) == 521), True, False)

          self._branches = self._branches.join(self._branches.groupby('BToKEE_event')['BToKEE_event'].count(), on='BToKEE_event', rsuffix='_r').rename(columns={'BToKEE_event_r': 'BToKEE_nCand'})
          # random select one from each event
          if (not self._isMC) and self._random:
            self._branches = self._branches.groupby('BToKEE_event').apply(pd.DataFrame.sample, n=1).reset_index(drop=True)

          # add additional branches
          self._branches['BToKEE_l_xy_sig'] = self._branches['BToKEE_l_xy'] / self._branches['BToKEE_l_xy_unc']
          self._branches['BToKEE_l1_dxy_sig'] = self._branches['BToKEE_l1_dxy'] / self._branches['BToKEE_l1_dxyErr']
          self._branches['BToKEE_l2_dxy_sig'] = self._branches['BToKEE_l2_dxy'] / self._branches['BToKEE_l2_dxyErr']
          self._branches['BToKEE_fit_l1_normpt'] = self._branches['BToKEE_fit_l1_pt'] / self._branches['BToKEE_fit_mass']
          self._branches['BToKEE_fit_l2_normpt'] = self._branches['BToKEE_fit_l2_pt'] / self._branches['BToKEE_fit_mass']
          self._branches['BToKEE_fit_k_normpt'] = self._branches['BToKEE_fit_k_pt'] / self._branches['BToKEE_fit_mass']
          self._branches['BToKEE_fit_normpt'] = self._branches['BToKEE_fit_pt'] / self._branches['BToKEE_fit_mass']
          self._branches['BToKEE_q2'] = self._branches['BToKEE_mll_fullfit'] * self._branches['BToKEE_mll_fullfit']
          self._branches['BToKEE_b_iso04_rel'] = self._branches['BToKEE_b_iso04'] / self._branches['BToKEE_fit_pt']
          self._branches['BToKEE_l1_iso04_rel'] = self._branches['BToKEE_l1_iso04'] / self._branches['BToKEE_fit_l1_pt']
          self._branches['BToKEE_l2_iso04_rel'] = self._branches['BToKEE_l2_iso04'] / self._branches['BToKEE_fit_l2_pt']
          self._branches['BToKEE_k_iso04_rel'] = self._branches['BToKEE_k_iso04'] / self._branches['BToKEE_fit_k_pt']
          self._branches['BToKEE_eleEtaCats'] = map(self.EleEtaCats, self._branches['BToKEE_fit_l1_eta'], self._branches['BToKEE_fit_l2_eta'])
          self._branches['BToKEE_ll_charge'] = self._branches['BToKEE_l1_charge'] + self._branches['BToKEE_l2_charge']
          self._branches['BToKEE_fold'] = np.repeat(self._fold, self._branches.shape[0])


          # mass hypothesis to veto fake event from semi-leptonic decay D
          l1_p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(self._branches['BToKEE_fit_l1_pt'], self._branches['BToKEE_fit_l1_eta'], self._branches['BToKEE_fit_l1_phi'], ELECTRON_MASS)
          l2_p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(self._branches['BToKEE_fit_l2_pt'], self._branches['BToKEE_fit_l2_eta'], self._branches['BToKEE_fit_l2_phi'], ELECTRON_MASS)
          #l1_pihypo_p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(self._branches['BToKEE_fit_l1_pt'], self._branches['BToKEE_fit_l1_eta'], self._branches['BToKEE_fit_l1_phi'], PI_MASS)
          #l2_pihypo_p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(self._branches['BToKEE_fit_l2_pt'], self._branches['BToKEE_fit_l2_eta'], self._branches['BToKEE_fit_l2_phi'], PI_MASS)
          #l1_khypo_p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(self._branches['BToKEE_fit_l1_pt'], self._branches['BToKEE_fit_l1_eta'], self._branches['BToKEE_fit_l1_phi'], K_MASS)
          #l2_khypo_p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(self._branches['BToKEE_fit_l2_pt'], self._branches['BToKEE_fit_l2_eta'], self._branches['BToKEE_fit_l2_phi'], K_MASS)
          k_p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(self._branches['BToKEE_fit_k_pt'], self._branches['BToKEE_fit_k_eta'], self._branches['BToKEE_fit_k_phi'], K_MASS)
          #k_pihypo_p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(self._branches['BToKEE_fit_k_pt'], self._branches['BToKEE_fit_k_eta'], self._branches['BToKEE_fit_k_phi'], PI_MASS)

          self._branches['BToKEE_eleDR'] = l1_p4.delta_r(l2_p4)
          self._branches['BToKEE_llkDR'] = (l1_p4 + l2_p4).delta_r(k_p4)

          #self._branches['BToKEE_Dmass_l1'] = (l1_pihypo_p4 + k_p4).mass
          #self._branches['BToKEE_Dmass_l2'] = (l2_pihypo_p4 + k_p4).mass
          #self._branches['BToKEE_Dmass_flip_l1'] = (l1_khypo_p4 + k_pihypo_p4).mass
          #self._branches['BToKEE_Dmass_flip_l2'] = (l2_khypo_p4 + k_pihypo_p4).mass
          #self._branches['BToKEE_Dmass'] = np.where((self._branches['BToKEE_k_charge'] * self._branches['BToKEE_l1_charge']) < 0.0, self._branches['BToKEE_Dmass_l1'], self._branches['BToKEE_Dmass_l2'])
          #self._branches['BToKEE_Dmass_flip'] = np.where((self._branches['BToKEE_k_charge'] * self._branches['BToKEE_l1_charge']) < 0.0, self._branches['BToKEE_Dmass_flip_l1'], self._branches['BToKEE_Dmass_flip_l2'])

          diele_p3 = (l1_p4 + l2_p4).p3
          pv2sv_p3 = uproot_methods.TVector3Array.from_cartesian(self._branches['BToKEE_PV_x'] - self._branches['BToKEE_vtx_x'], self._branches['BToKEE_PV_y'] - self._branches['BToKEE_vtx_y'], self._branches['BToKEE_PV_z'] - self._branches['BToKEE_vtx_z'])
          self._branches['BToKEE_ptImbalance'] = np.array([p1.cross(p2).mag for p1, p2 in zip(diele_p3, pv2sv_p3)]) / np.array([p1.cross(p2).mag for p1, p2 in zip(k_p4.p3, pv2sv_p3)])
          self._branches['BToKEE_ptAsym'] = (np.array([p1.cross(p2).mag for p1, p2 in zip(diele_p3, pv2sv_p3)]) - np.array([p1.cross(p2).mag for p1, p2 in zip(k_p4.p3, pv2sv_p3)])) / (np.array([p1.cross(p2).mag for p1, p2 in zip(diele_p3, pv2sv_p3)]) + np.array([p1.cross(p2).mag for p1, p2 in zip(k_p4.p3, pv2sv_p3)])) 

          if self._evalMVA:
            #self._branches['BToKEE_mva'] = model.predict(xgb.DMatrix(self._branches[training_branches].replace([np.inf, -np.inf], 0.0).sort_index(axis=1)), ntree_limit=ntree_limit)
            self._branches['BToKEE_mva'] = model.predict(xgb.DMatrix(self._branches[training_branches].replace([np.inf, -np.inf], 0.0).sort_index(axis=1)))
            #self._branches = self._branches[(self._branches['BToKEE_mva'] > mvaCut)].sort_values('BToKEE_mva', ascending=False).drop_duplicates(['BToKEE_event'], keep='first')
            self._branches = self._branches[(self._branches['BToKEE_mva'] > mvaCut)]

          # fill output
          #self._branches = self._branches.sort_values('BToKEE_fit_pt', ascending=False).drop_duplicates(['BToKEE_event'], keep='first')
          self.fill_output()

        startTime = time.time()

    self.finish()
    print('[BToKLLAnalyzer::run] INFO: Finished')
    self.print_timestamp()






