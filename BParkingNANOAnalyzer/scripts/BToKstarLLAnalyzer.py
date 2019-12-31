#! /usr/bin/env python

#import ROOT
from math import ceil
import awkward
import uproot
import uproot_methods
import pandas as pd
import numpy as np
from BParkingNANOAnalysis.BParkingNANOAnalyzer.BaseAnalyzer import BParkingNANOAnalyzer


class BToKstarLLAnalyzer(BParkingNANOAnalyzer):
  def __init__(self, inputfiles, outputfile, hist=False, isMC=False):
    self._isMC = isMC
    inputbranches_BToKsEE = ['nBToKsEE',
                            'BToKsEE_mll_raw',
                            'BToKsEE_mll_llfit',
                            #'BToKsEE_mllErr_llfit',
                            'BToKsEE_mll_fullfit',
                            'BToKsEE_mass',
                            'BToKsEE_fit_mass',
                            'BToKsEE_fit_massErr',
                            'BToKsEE_fitted_barMass',
                            'BToKsEE_mkstar_fullfit',
                            'BToKsEE_barMkstar_fullfit',
                            'BToKsEE_l1_idx',
                            'BToKsEE_l2_idx',
                            'BToKsEE_trk1_idx',
                            'BToKsEE_trk2_idx',
                            'BToKsEE_l_xy',
                            'BToKsEE_l_xy_unc',
                            'BToKsEE_fit_pt',
                            'BToKsEE_fit_eta',
                            'BToKsEE_fit_phi',
                            'BToKsEE_lep1pt_fullfit',
                            'BToKsEE_lep1eta_fullfit',
                            'BToKsEE_lep1phi_fullfit',
                            'BToKsEE_lep2pt_fullfit',
                            'BToKsEE_lep2eta_fullfit',
                            'BToKsEE_lep2phi_fullfit',
                            'BToKsEE_trk1pt_fullfit',
                            'BToKsEE_trk1eta_fullfit',
                            'BToKsEE_trk1phi_fullfit',
                            'BToKsEE_trk2pt_fullfit',
                            'BToKsEE_trk2eta_fullfit',
                            'BToKsEE_trk2phi_fullfit',
                            'BToKsEE_svprob',
                            'BToKsEE_fit_cos2D',
                            #'Electron_pt',
                            #'Electron_eta',
                            #'Electron_phi',
                            'Electron_dz',
                            'Electron_dxy',
                            'Electron_dxyErr',
                            'Electron_ptBiased',
                            'Electron_unBiased',
                            'Electron_convVeto',
                            'Electron_isLowPt',
                            'Electron_isPF',
                            'Electron_isPFoverlap',
                            'Electron_mvaId',
                            #'Electron_pfmvaId',
                            #'Electron_lostHits',
                            'ProbeTracks_pt',
                            'ProbeTracks_DCASig',
                            'ProbeTracks_eta',
                            'ProbeTracks_phi',
                            'ProbeTracks_dz',
                            #'ProbeTracks_isLostTrk',
                            #'ProbeTracks_isPacked',
                            #'HLT_Mu9_IP6_*',
                            'event'
                            ]

    inputbranches_BToKsEE_mc = ['GenPart_pdgId',
                               'GenPart_genPartIdxMother',
                               'Electron_genPartIdx',
                               'ProbeTracks_genPartIdx',
                               ]

    if self._isMC:
      inputbranches_BToKsEE += inputbranches_BToKsEE_mc

 
    outputbranches_BToKsEE = {'BToKsEE_mll_raw': {'nbins': 50, 'xmin': 0.0, 'xmax': 5.0},
                             'BToKsEE_mll_llfit': {'nbins': 30, 'xmin': 2.6, 'xmax': 3.6},
                             #'BToKsEE_mllErr_llfit': {'nbins': 30, 'xmin': 0.0, 'xmax': 3.0},
                             'BToKsEE_mll_fullfit': {'nbins': 30, 'xmin': 2.6, 'xmax': 3.6},
                             'BToKsEE_mass': {'nbins': 30, 'xmin': 4.7, 'xmax': 6.0},
                             'BToKsEE_fit_mass': {'nbins': 30, 'xmin': 4.7, 'xmax': 6.0},
                             'BToKsEE_fit_massErr': {'nbins': 30, 'xmin': 0.0, 'xmax': 3.0},
                             'BToKsEE_fit_barmass': {'nbins': 30, 'xmin': 4.7, 'xmax': 6.0},
                             'BToKsEE_mkstar_fullfit': {'nbins': 30, 'xmin': 4.7, 'xmax': 6.0},
                             'BToKsEE_barmkstar_fullfit': {'nbins': 30, 'xmin': 4.7, 'xmax': 6.0},
                             'BToKsEE_fit_l1_pt': {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0},
                             'BToKsEE_fit_l2_pt': {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0},
                             'BToKsEE_fit_l1_normpt': {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0},
                             'BToKsEE_fit_l2_normpt': {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0},
                             'BToKsEE_fit_l1_eta': {'nbins': 50, 'xmin': -3.0, 'xmax': 3.0},
                             'BToKsEE_fit_l2_eta': {'nbins': 50, 'xmin': -3.0, 'xmax': 3.0},
                             'BToKsEE_fit_l1_phi': {'nbins': 50, 'xmin': -4.0, 'xmax': 4.0},
                             'BToKsEE_fit_l2_phi': {'nbins': 50, 'xmin': -4.0, 'xmax': 4.0},
                             'BToKsEE_l1_dxy_sig': {'nbins': 50, 'xmin': -30.0, 'xmax': 30.0},
                             'BToKsEE_l2_dxy_sig': {'nbins': 50, 'xmin': -30.0, 'xmax': 30.0},
                             'BToKsEE_l1_dz': {'nbins': 50, 'xmin': -1.0, 'xmax': 1.0},
                             'BToKsEE_l2_dz': {'nbins': 50, 'xmin': -1.0, 'xmax': 1.0},
                             'BToKsEE_l1_unBiased': {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0},
                             'BToKsEE_l2_unBiased': {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0},
                             'BToKsEE_l1_ptBiased': {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0},
                             'BToKsEE_l2_ptBiased': {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0},
                             'BToKsEE_l1_mvaId': {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0},
                             'BToKsEE_l2_mvaId': {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0},
                             #'BToKsEE_l1_pfmvaId': {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0},
                             #'BToKsEE_l2_pfmvaId': {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0},
                             'BToKsEE_l1_isPF': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                             'BToKsEE_l2_isPF': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                             'BToKsEE_l1_isLowPt': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                             'BToKsEE_l2_isLowPt': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                             'BToKsEE_l1_isPFoverlap': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                             'BToKsEE_l2_isPFoverlap': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                             'BToKsEE_fit_trk1_pt': {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0},
                             'BToKsEE_fit_trk1_normpt': {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0},
                             'BToKsEE_fit_trk1_eta': {'nbins': 50, 'xmin': -3.0, 'xmax': 3.0},
                             'BToKsEE_fit_trk1_phi': {'nbins': 50, 'xmin': -4.0, 'xmax': 4.0},
                             'BToKsEE_trk1_DCASig': {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0},
                             'BToKsEE_fit_trk2_pt': {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0},
                             'BToKsEE_fit_trk2_normpt': {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0},
                             'BToKsEE_fit_trk2_eta': {'nbins': 50, 'xmin': -3.0, 'xmax': 3.0},
                             'BToKsEE_fit_trk2_phi': {'nbins': 50, 'xmin': -4.0, 'xmax': 4.0},
                             'BToKsEE_trk2_DCASig': {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0},
                             'BToKsEE_fit_pt': {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0},
                             'BToKsEE_fit_eta': {'nbins': 50, 'xmin': -3.0, 'xmax': 3.0},
                             'BToKsEE_fit_phi': {'nbins': 50, 'xmin': -4.0, 'xmax': 4.0},
                             'BToKsEE_fit_normpt': {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0},
                             'BToKsEE_svprob': {'nbins': 50, 'xmin': 0.0, 'xmax': 1.0},
                             'BToKsEE_fit_cos2D': {'nbins': 50, 'xmin': 0.999, 'xmax': 1.0},
                             'BToKsEE_l_xy_sig': {'nbins': 50, 'xmin': 0.0, 'xmax': 50.0},
                             'BToKsEE_event': {'nbins': 10, 'xmin': 0, 'xmax': 10},
                             }

    super(BToKstarLLAnalyzer, self).__init__(inputfiles, outputfile, inputbranches_BToKsEE, outputbranches_BToKsEE, hist)

  def run(self):
    ELECTRON_MASS = 0.000511
    K_MASS = 0.493677
    JPSI_LOW = 2.9
    JPSI_UP = 3.3
    B_MASS = 5.245
    B_SIGMA = 9.155e-02
    B_LOWSB_LOW = B_MASS - 6.0*B_SIGMA
    B_LOWSB_UP = B_MASS - 3.0*B_SIGMA
    B_UPSB_LOW = B_MASS + 3.0*B_SIGMA
    B_UPSB_UP = B_MASS + 6.0*B_SIGMA
    B_LOW = 4.5
    B_UP = 6.0

    print('[BToKstarLLAnalyzer::run] INFO: Running the analyzer...')
    self.print_timestamp()
    self.init_output()
    for (self._ifile, filename) in enumerate(self._file_in_name):
      print('[BToKstarLLAnalyzer::run] INFO: FILE: {}/{}. Loading file...'.format(self._ifile+1, self._num_files))
      tree = uproot.open(filename)['Events']
      self._branches = tree.arrays(self._inputbranches)
      self._branches = {key: awkward.fromiter(branch) for key, branch in self._branches.items()} # need this line for the old version of awkward/uproot (for condor job)

      print('[BToKstarLLAnalyzer::run] INFO: FILE: {}/{}. Analyzing...'.format(self._ifile+1, self._num_files))

      if self._isMC:
        # reconstruct full decay chain
        self._branches['BToKsEE_l1_genMotherIdx'] = self._branches['GenPart_genPartIdxMother'][self._branches['Electron_genPartIdx'][self._branches['BToKsEE_l1Idx']]]
        self._branches['BToKsEE_l2_genMotherIdx'] = self._branches['GenPart_genPartIdxMother'][self._branches['Electron_genPartIdx'][self._branches['BToKsEE_l2Idx']]]
        self._branches['BToKsEE_k_genMotherIdx'] = self._branches['GenPart_genPartIdxMother'][self._branches['ProbeTracks_genPartIdx'][self._branches['BToKsEE_kIdx']]]

        self._branches['BToKsEE_l1_genMotherPdgId'] = self._branches['GenPart_pdgId'][self._branches['BToKsEE_l1_genMotherIdx']]
        self._branches['BToKsEE_l2_genMotherPdgId'] = self._branches['GenPart_pdgId'][self._branches['BToKsEE_l2_genMotherIdx']]
        self._branches['BToKsEE_k_genMotherPdgId'] = self._branches['GenPart_pdgId'][self._branches['BToKsEE_k_genMotherIdx']]

        self._branches['BToKsEE_l1Mother_genMotherIdx'] = self._branches['GenPart_genPartIdxMother'][self._branches['BToKsEE_l1_genMotherIdx']]
        self._branches['BToKsEE_l2Mother_genMotherIdx'] = self._branches['GenPart_genPartIdxMother'][self._branches['BToKsEE_l2_genMotherIdx']]
        self._branches['BToKsEE_kMother_genMotherIdx'] = self._branches['GenPart_genPartIdxMother'][self._branches['BToKsEE_k_genMotherIdx']]

        self._branches['BToKsEE_l1Mother_genMotherPdgId'] = self._branches['GenPart_pdgId'][self._branches['BToKsEE_l1Mother_genMotherIdx']]
        self._branches['BToKsEE_l2Mother_genMotherPdgId'] = self._branches['GenPart_pdgId'][self._branches['BToKsEE_l2Mother_genMotherIdx']]
        self._branches['BToKsEE_kMother_genMotherPdgId'] = self._branches['GenPart_pdgId'][self._branches['BToKsEE_kMother_genMotherIdx']]


      # remove cross referencing
      for branch in self._branches.keys():
        if 'Electron_' in branch:
          self._branches['BToKsEE_l1_'+branch.replace('Electron_','')] = self._branches[branch][self._branches['BToKsEE_l1_idx']] 
          self._branches['BToKsEE_l2_'+branch.replace('Electron_','')] = self._branches[branch][self._branches['BToKsEE_l2_idx']] 
          del self._branches[branch]

        if 'ProbeTracks_' in branch:
          self._branches['BToKsEE_trk1_'+branch.replace('ProbeTracks_','')] = self._branches[branch][self._branches['BToKsEE_trk1_idx']] 
          self._branches['BToKsEE_trk2_'+branch.replace('ProbeTracks_','')] = self._branches[branch][self._branches['BToKsEE_trk2_idx']] 
          del self._branches[branch]
        
        if 'GenPart_' in branch:
          del self._branches[branch]

        if 'HLT_Mu9_IP6_' in branch:
          self._branches['BToKsEE_'+branch] = np.repeat(self._branches[branch], self._branches['nBToKsEE'])
          del self._branches[branch]

        if branch == 'event':
          self._branches['BToKsEE_'+branch] = np.repeat(self._branches[branch], self._branches['nBToKsEE'])
          del self._branches[branch]
        

      del self._branches['nBToKsEE']



      #l1_p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(self._branches['BToKsEE_l1_pt'], self._branches['BToKsEE_l1_eta'], self._branches['BToKsEE_l1_phi'], ELECTRON_MASS)
      #l2_p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(self._branches['BToKsEE_l2_pt'], self._branches['BToKsEE_l2_eta'], self._branches['BToKsEE_l2_phi'], ELECTRON_MASS)
      #k_p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(self._branches['BToKsEE_k_pt'], self._branches['BToKsEE_k_eta'], self._branches['BToKsEE_k_phi'], K_MASS)
      #self._branches['BToKsEE_mll_raw'] = (l1_p4 + l2_p4).mass
      #self._branches['BToKsEE_mass'] = (l1_p4 + l2_p4 + k_p4).mass

      # flatten the jagged arrays to a normal numpy array, turn the whole dictionary to pandas dataframe
      self._branches = pd.DataFrame.from_dict({branch: array.flatten() for branch, array in self._branches.items()})
      #self._branches = awkward.topandas(self._branches, flatten=True)

      rename_column = {'BToKsEE_lep1pt_fullfit': 'BToKsEE_fit_l1_pt',
                       'BToKsEE_lep1eta_fullfit': 'BToKsEE_fit_l1_eta',
                       'BToKsEE_lep1phi_fullfit': 'BToKsEE_fit_l1_phi',
                       'BToKsEE_lep2pt_fullfit': 'BToKsEE_fit_l2_pt',
                       'BToKsEE_lep2eta_fullfit': 'BToKsEE_fit_l2_eta',
                       'BToKsEE_lep2phi_fullfit': 'BToKsEE_fit_l2_phi',
                       'BToKsEE_trk1pt_fullfit': 'BToKsEE_fit_trk1_pt',
                       'BToKsEE_trk1eta_fullfit': 'BToKsEE_fit_trk1_eta',
                       'BToKsEE_trk1phi_fullfit': 'BToKsEE_fit_trk1_phi',
                       'BToKsEE_trk2pt_fullfit': 'BToKsEE_fit_trk2_pt',
                       'BToKsEE_trk2eta_fullfit': 'BToKsEE_fit_trk2_eta',
                       'BToKsEE_trk2phi_fullfit': 'BToKsEE_fit_trk2_phi',
                       'BToKsEE_fitted_barMass': 'BToKsEE_fit_barmass',
                       'BToKsEE_barMkstar_fullfit': 'BToKsEE_barmkstar_fullfit',
                       }
      
      self._branches.rename(columns=rename_column, inplace=True)
      self._branches['BToKsEE_fit_mass'] = np.where((self._branches['BToKsEE_mkstar_fullfit'] > 0.742) & (self._branches['BToKsEE_mkstar_fullfit'] < 1.042), self._branches['BToKsEE_fit_mass'], self._branches['BToKsEE_fit_barmass'])
      self._branches['BToKsEE_fit_barmass'] = np.where((self._branches['BToKsEE_mkstar_fullfit'] > 0.742) & (self._branches['BToKsEE_mkstar_fullfit'] < 1.042), self._branches['BToKsEE_fit_barmass'], self._branches['BToKsEE_fit_mass'])

      # add additional branches
      self._branches['BToKsEE_l_xy_sig'] = self._branches['BToKsEE_l_xy'] / self._branches['BToKsEE_l_xy_unc']
      self._branches['BToKsEE_l1_dxy_sig'] = self._branches['BToKsEE_l1_dxy'] / self._branches['BToKsEE_l1_dxyErr']
      self._branches['BToKsEE_l2_dxy_sig'] = self._branches['BToKsEE_l2_dxy'] / self._branches['BToKsEE_l2_dxyErr']
      self._branches['BToKsEE_fit_l1_normpt'] = self._branches['BToKsEE_fit_l1_pt'] / self._branches['BToKsEE_fit_mass']
      self._branches['BToKsEE_fit_l2_normpt'] = self._branches['BToKsEE_fit_l2_pt'] / self._branches['BToKsEE_fit_mass']
      self._branches['BToKsEE_fit_trk1_normpt'] = self._branches['BToKsEE_fit_trk1_pt'] / self._branches['BToKsEE_fit_mass']
      self._branches['BToKsEE_fit_trk2_normpt'] = self._branches['BToKsEE_fit_trk2_pt'] / self._branches['BToKsEE_fit_mass']
      self._branches['BToKsEE_fit_normpt'] = self._branches['BToKsEE_fit_pt'] / self._branches['BToKsEE_fit_mass']

      # general selection
      
      sv_selection = (self._branches['BToKsEE_fit_pt'] > 3.0) #& (self._branches['BToKsEE_l_xy_sig'] > 6.0 ) & (self._branches['BToKsEE_svprob'] > 0.01) & (self._branches['BToKsEE_fit_cos2D'] > 0.9)
      l1_selection = (self._branches['BToKsEE_l1_convVeto']) & (self._branches['BToKsEE_fit_l1_pt'] > 1.5) & (self._branches['BToKsEE_l1_mvaId'] > 3.94) #& (np.logical_not(self._branches['BToKsEE_l1_isPFoverlap']))
      l2_selection = (self._branches['BToKsEE_l2_convVeto']) & (self._branches['BToKsEE_fit_l2_pt'] > 0.5) & (self._branches['BToKsEE_l2_mvaId'] > 3.94) #& (np.logical_not(self._branches['BToKsEE_l2_isPFoverlap']))
      #k_selection = (self._branches['BToKsEE_fit_k_pt'] > 0.5) #& (self._branches['BToKsEE_k_DCASig'] > 2.0)
      additional_selection = (self._branches['BToKsEE_fit_mass'] > B_LOW) & (self._branches['BToKsEE_fit_mass'] < B_UP)

      b_lowsb_selection = (self._branches['BToKsEE_fit_mass'] > B_LOWSB_LOW) & (self._branches['BToKsEE_fit_mass'] < B_LOWSB_UP)
      b_upsb_selection = (self._branches['BToKsEE_fit_mass'] > B_UPSB_LOW) & (self._branches['BToKsEE_fit_mass'] < B_UPSB_UP)
      b_sb_selection = b_lowsb_selection | b_upsb_selection

      selection = sv_selection & l1_selection & l2_selection & additional_selection

      self._branches = self._branches[selection]
      if self._isMC:
        self._branches['BToKsEE_decay'] = self._branches.apply(self.DecayCats, axis=1)
        #self._branches.query('BToKsEE_decay == 1', inplace=True) # B->K J/psi(ll)
        #self._branches.query('BToKsEE_decay == 3', inplace=True) # B->K*(K pi) J/psi(ll)
            

      # fill output
      self.fill_output()

    self.finish()
    print('[BToKstarLLAnalyzer::run] INFO: Finished')
    self.print_timestamp()

  def DecayCats(self, row):    
    mc_matched_selection = (row['BToKsEE_l1_genPartIdx'] > -0.5) & (row['BToKsEE_l2_genPartIdx'] > -0.5) & (row['BToKsEE_k_genPartIdx'] > -0.5)
    # B->K ll
    RK_nonresonant_chain_selection = (abs(row['BToKsEE_l1_genMotherPdgId']) == 521) & (abs(row['BToKsEE_k_genMotherPdgId']) == 521)
    RK_nonresonant_chain_selection &= (row['BToKsEE_l1_genMotherPdgId'] == row['BToKsEE_l2_genMotherPdgId']) & (row['BToKsEE_k_genMotherPdgId'] == row['BToKsEE_l1_genMotherPdgId'])
    RK_nonresonant_chain_selection &= mc_matched_selection

    # B->K J/psi(ll)
    RK_resonant_chain_selection = (abs(row['BToKsEE_l1_genMotherPdgId']) == 443) & (abs(row['BToKsEE_k_genMotherPdgId']) == 521)
    RK_resonant_chain_selection &= (row['BToKsEE_l1_genMotherPdgId'] == row['BToKsEE_l2_genMotherPdgId']) & (row['BToKsEE_k_genMotherPdgId'] == row['BToKsEE_l1Mother_genMotherPdgId']) & (row['BToKsEE_k_genMotherPdgId'] == row['BToKsEE_l2Mother_genMotherPdgId'])
    RK_resonant_chain_selection &= mc_matched_selection

    # B->K*(K pi) ll
    RKstar_nonresonant_chain_selection = (abs(row['BToKsEE_l1_genMotherPdgId']) == 511) & (abs(row['BToKsEE_k_genMotherPdgId']) == 313)
    RKstar_nonresonant_chain_selection &= (row['BToKsEE_l1_genMotherPdgId'] == row['BToKsEE_l2_genMotherPdgId']) & (row['BToKsEE_l1_genMotherPdgId'] == row['BToKsEE_kMother_genMotherPdgId']) 
    RKstar_nonresonant_chain_selection &= mc_matched_selection

    # B->K*(K pi) J/psi(ll)
    RKstar_resonant_chain_selection = (abs(row['BToKsEE_l1_genMotherPdgId']) == 443) & (abs(row['BToKsEE_k_genMotherPdgId']) == 313)
    RKstar_resonant_chain_selection &= (row['BToKsEE_l1_genMotherPdgId'] == row['BToKsEE_l2_genMotherPdgId']) & (row['BToKsEE_l1Mother_genMotherPdgId'] == row['BToKsEE_kMother_genMotherPdgId']) 
    RKstar_resonant_chain_selection &= mc_matched_selection

    # Bs->phi(K K) ll
    Rphi_nonresonant_chain_selection = (abs(row['BToKsEE_l1_genMotherPdgId']) == 531) & (abs(row['BToKsEE_k_genMotherPdgId']) == 333)
    Rphi_nonresonant_chain_selection &= (row['BToKsEE_l1_genMotherPdgId'] == row['BToKsEE_l2_genMotherPdgId']) & (row['BToKsEE_l1_genMotherPdgId'] == row['BToKsEE_kMother_genMotherPdgId']) 
    Rphi_nonresonant_chain_selection &= mc_matched_selection

    # Bs->phi(K K) J/psi(ll)
    Rphi_resonant_chain_selection = (abs(row['BToKsEE_l1_genMotherPdgId']) == 443) & (abs(row['BToKsEE_k_genMotherPdgId']) == 333)
    Rphi_resonant_chain_selection &= (row['BToKsEE_l1_genMotherPdgId'] == row['BToKsEE_l2_genMotherPdgId']) & (row['BToKsEE_l1Mother_genMotherPdgId'] == row['BToKsEE_kMother_genMotherPdgId']) 
    Rphi_resonant_chain_selection &= mc_matched_selection

    if RK_nonresonant_chain_selection: return 0
    elif RK_resonant_chain_selection: return 1
    elif RKstar_nonresonant_chain_selection: return 2
    elif RKstar_resonant_chain_selection: return 3
    elif Rphi_nonresonant_chain_selection: return 4
    elif Rphi_resonant_chain_selection: return 5
    else: return -1








