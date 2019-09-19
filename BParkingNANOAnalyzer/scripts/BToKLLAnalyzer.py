#! /usr/bin/env python

#import ROOT
from math import ceil
import awkward
import uproot
import uproot_methods
import pandas as pd
import numpy as np
from BParkingNANOAnalysis.BParkingNANOAnalyzer.BaseAnalyzer import BParkingNANOAnalyzer


class BToKLLAnalyzer(BParkingNANOAnalyzer):
  def __init__(self, inputfiles, outputfile, hist=False):
    inputbranches_BToKEE = ['nBToKEE',
                            'BToKEE_mll_raw',
                            'BToKEE_mll_fullfit',
                            'BToKEE_mass',
                            'BToKEE_fit_mass',
                            'BToKEE_l1Idx',
                            'BToKEE_l2Idx',
                            'BToKEE_kIdx',
                            'BToKEE_l_xy',
                            'BToKEE_l_xy_unc',
                            'BToKEE_pt',
                            #'BToKEE_eta',
                            'BToKEE_svprob',
                            'BToKEE_cos2D',
                            'Electron_pt',
                            #'Electron_eta',
                            #'Electron_phi',
                            #'Electron_dz',
                            #'Electron_unBiased',
                            'Electron_convVeto',
                            'Electron_isLowPt',
                            'Electron_isPF',
                            'Electron_isPFoverlap',
                            'Electron_mvaId',
                            #'Electron_lostHits',
                            'ProbeTracks_pt',
                            'ProbeTracks_DCASig',
                            #'ProbeTracks_eta',
                            #'ProbeTracks_phi',
                            #'ProbeTracks_dz',
                            #'ProbeTracks_isLostTrk',
                            #'ProbeTracks_isPacked',
                            #'HLT_Mu9_IP6_*',
                            ]

    
    outputbranches_BToKEE = {'BToKEE_mll_raw': {'nbins': 50, 'xmin': 0.0, 'xmax': 5.0},
                             'BToKEE_mll_raw_jpsi_all': {'nbins': 30, 'xmin': 2.6, 'xmax': 3.6},
                             'BToKEE_mll_raw_jpsi_pf': {'nbins': 30, 'xmin': 2.6, 'xmax': 3.6},
                             'BToKEE_mll_raw_jpsi_mix': {'nbins': 30, 'xmin': 2.6, 'xmax': 3.6},
                             'BToKEE_mll_raw_jpsi_low': {'nbins': 30, 'xmin': 2.6, 'xmax': 3.6},
                             'BToKEE_mll_fullfit_jpsi_all': {'nbins': 30, 'xmin': 2.6, 'xmax': 3.6},
                             'BToKEE_mll_fullfit_jpsi_pf': {'nbins': 30, 'xmin': 2.6, 'xmax': 3.6},
                             'BToKEE_mll_fullfit_jpsi_mix': {'nbins': 30, 'xmin': 2.6, 'xmax': 3.6},
                             'BToKEE_mll_fullfit_jpsi_low': {'nbins': 30, 'xmin': 2.6, 'xmax': 3.6},
                             'BToKEE_mass_all': {'nbins': 30, 'xmin': 4.7, 'xmax': 6.0},
                             'BToKEE_mass_pf': {'nbins': 30, 'xmin': 4.7, 'xmax': 6.0},
                             'BToKEE_mass_mix': {'nbins': 30, 'xmin': 4.7, 'xmax': 6.0},
                             'BToKEE_mass_low': {'nbins': 30, 'xmin': 4.7, 'xmax': 6.0},
                             'BToKEE_fit_mass_all': {'nbins': 30, 'xmin': 4.7, 'xmax': 6.0},
                             'BToKEE_fit_mass_pf': {'nbins': 30, 'xmin': 4.7, 'xmax': 6.0},
                             'BToKEE_fit_mass_mix': {'nbins': 30, 'xmin': 4.7, 'xmax': 6.0},
                             'BToKEE_fit_mass_low': {'nbins': 30, 'xmin': 4.7, 'xmax': 6.0},
                             'BToKEE_l1_pt_pf': {'nbins': 50, 'xmin': 0.0, 'xmax': 20.0},
                             'BToKEE_l1_pt_low': {'nbins': 50, 'xmin': 0.0, 'xmax': 20.0},
                             'BToKEE_l2_pt_pf': {'nbins': 50, 'xmin': 0.0, 'xmax': 5.0},
                             'BToKEE_l2_pt_low': {'nbins': 50, 'xmin': 0.0, 'xmax': 5.0},
                             'BToKEE_l1_mvaId_low': {'nbins': 50, 'xmin': -10.0, 'xmax': 10.0},
                             'BToKEE_l2_mvaId_low': {'nbins': 50, 'xmin': -10.0, 'xmax': 10.0},
                             'BToKEE_k_pt': {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0},
                             'BToKEE_pt': {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0},
                             'BToKEE_svprob': {'nbins': 50, 'xmin': 0.0, 'xmax': 1.0},
                             'BToKEE_cos2D': {'nbins': 50, 'xmin': 0.999, 'xmax': 1.0},
                             'BToKEE_l_xy_sig': {'nbins': 80, 'xmin': 0.0, 'xmax': 500.0},
                             }
    '''
    outputbranches_BToKEE = ['BToKEE_l1_pt', 'BToKEE_l2_pt', 'BToKEE_k_pt', 'BToKEE_k_DCASig', 'BToKEE_pt', 'BToKEE_eta', 'BToKEE_svprob', 'BToKEE_cos2D', 'BToKEE_l_xy_sig', 'BToKEE_mll_raw', 'BToKEE_mass']
    outputbranches_BToKEE = {key:1 for key in outputbranches_BToKEE}
    '''

    super(BToKLLAnalyzer, self).__init__(inputfiles, outputfile, inputbranches_BToKEE, outputbranches_BToKEE, hist)

  def run(self):
    ELECTRON_MASS = 0.000511
    K_MASS = 0.493677

    print('[BToKLLAnalyzer::run] INFO: Running the analyzer...')
    self.print_timestamp()
    self.init_output()
    for (self._ifile, filename) in enumerate(self._file_in_name):
      print('[BToKLLAnalyzer::run] INFO: FILE: {}/{}. Loading file...'.format(self._ifile+1, self._num_files))
      tree = uproot.open(filename)['Events']
      nTree = 0
      #for self._branches in tree.iterate(self._inputbranches, entrysteps=10000):
      #for self._branches in tree.iterate(self._inputbranches, entrysteps=float("inf")):
        #print('[BToKLLAnalyzer::run] INFO: FILE: {}/{}. Iterating tree {}...'.format(self._ifile+1, self._num_files, nTree+1))
      self._branches = tree.arrays(self._inputbranches)
      self._branches = {key: awkward.fromiter(branch) for key, branch in self._branches.items()}

      print('[BToKLLAnalyzer::run] INFO: FILE: {}/{}. Analyzing...'.format(self._ifile+1, self._num_files))

      # remove cross referencing
      for branch in self._branches.keys():
        if 'Electron_' in branch:
          self._branches['BToKEE_l1_'+branch.replace('Electron_','')] = self._branches[branch][self._branches['BToKEE_l1Idx']] 
          self._branches['BToKEE_l2_'+branch.replace('Electron_','')] = self._branches[branch][self._branches['BToKEE_l2Idx']] 
          del self._branches[branch]

        if 'ProbeTracks_' in branch:
          self._branches['BToKEE_k_'+branch.replace('ProbeTracks_','')] = self._branches[branch][self._branches['BToKEE_kIdx']] 
          del self._branches[branch]

        if 'HLT_Mu9_IP6_' in branch:
          self._branches['BToKEE_'+branch] = np.repeat(self._branches[branch], self._branches['nBToKEE'])
          del self._branches[branch]

      del self._branches['nBToKEE']


      #l1_p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(self._branches['BToKEE_l1_pt'], self._branches['BToKEE_l1_eta'], self._branches['BToKEE_l1_phi'], ELECTRON_MASS)
      #l2_p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(self._branches['BToKEE_l2_pt'], self._branches['BToKEE_l2_eta'], self._branches['BToKEE_l2_phi'], ELECTRON_MASS)
      #k_p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(self._branches['BToKEE_k_pt'], self._branches['BToKEE_k_eta'], self._branches['BToKEE_k_phi'], K_MASS)

      #self._branches['BToKEE_mll_raw'] = (l1_p4 + l2_p4).mass
      #self._branches['BToKEE_mass'] = (l1_p4 + l2_p4 + k_p4).mass

      # flatten the jagged arrays to a normal numpy array, turn the whole dictionary to pandas dataframe
      self._branches = pd.DataFrame.from_dict({branch: array.flatten() for branch, array in self._branches.items()})
      #self._branches = awkward.topandas(self._branches, flatten=True)

      # add additional branches
      self._branches['BToKEE_l_xy_sig'] = self._branches['BToKEE_l_xy'] / self._branches['BToKEE_l_xy_unc']

      # general selection
      sv_selection = (self._branches['BToKEE_pt'] > 3.0) & (self._branches['BToKEE_l_xy_sig'] > 6.0 ) & (self._branches['BToKEE_svprob'] > 0.1) & (self._branches['BToKEE_cos2D'] > 0.999)

      l1_selection = (self._branches['BToKEE_l1_convVeto']) & (self._branches['BToKEE_l1_pt'] > 1.5) & (self._branches['BToKEE_l1_mvaId'] > 3.96) #& (np.logical_not(self._branches['BToKEE_l1_isPFoverlap']))
      l2_selection = (self._branches['BToKEE_l2_convVeto']) & (self._branches['BToKEE_l2_pt'] > 0.5) & (self._branches['BToKEE_l2_mvaId'] > 3.96) #& (np.logical_not(self._branches['BToKEE_l2_isPFoverlap']))
      k_selection = (self._branches['BToKEE_k_pt'] > 3.0) & (self._branches['BToKEE_k_DCASig'] > 2.0)
      #additional_selection = (self._branches['BToKEE_l1_isPF']) & (self._branches['BToKEE_l2_isPF'])
      selection = sv_selection & l1_selection & l2_selection & k_selection #& additional_selection

      self._branches = self._branches[selection]

      
      # additional cuts, allows various lengths

      jpsi_selection = (self._branches['BToKEE_mll_raw'] > 2.9) & (self._branches['BToKEE_mll_raw'] < 3.3)
      l1_pf_selection = (self._branches['BToKEE_l1_isPF'])
      l2_pf_selection = (self._branches['BToKEE_l2_isPF'])
      l1_low_selection = (self._branches['BToKEE_l1_isLowPt']) #& (self._branches['BToKEE_l1_pt'] < 5.0)
      l2_low_selection = (self._branches['BToKEE_l2_isLowPt']) #& (self._branches['BToKEE_l2_pt'] < 5.0)

      pf_selection = l1_pf_selection & l2_pf_selection
      low_selection = l1_low_selection & l2_low_selection
      overlap_veto_selection = np.logical_not(self._branches['BToKEE_l1_isPFoverlap']) & np.logical_not(self._branches['BToKEE_l2_isPFoverlap'])
      mix_selection = ((l1_pf_selection & l2_low_selection) | (l2_pf_selection & l1_low_selection)) #& overlap_veto_selection


      self._branches['BToKEE_mll_raw_jpsi_all'] = self._branches['BToKEE_mll_raw'][overlap_veto_selection]
      self._branches['BToKEE_mll_raw_jpsi_pf'] = self._branches['BToKEE_mll_raw'][pf_selection]
      self._branches['BToKEE_mll_raw_jpsi_mix'] = self._branches['BToKEE_mll_raw'][mix_selection]
      self._branches['BToKEE_mll_raw_jpsi_low'] = self._branches['BToKEE_mll_raw'][low_selection]

      self._branches['BToKEE_mll_fullfit_jpsi_all'] = self._branches['BToKEE_mll_fullfit'][overlap_veto_selection]
      self._branches['BToKEE_mll_fullfit_jpsi_pf'] = self._branches['BToKEE_mll_fullfit'][pf_selection]
      self._branches['BToKEE_mll_fullfit_jpsi_mix'] = self._branches['BToKEE_mll_fullfit'][mix_selection]
      self._branches['BToKEE_mll_fullfit_jpsi_low'] = self._branches['BToKEE_mll_fullfit'][low_selection]


      self._branches['BToKEE_mass_all'] = self._branches['BToKEE_mass'][jpsi_selection & overlap_veto_selection]
      self._branches['BToKEE_mass_pf'] = self._branches['BToKEE_mass'][jpsi_selection & pf_selection]
      self._branches['BToKEE_mass_mix'] = self._branches['BToKEE_mass'][jpsi_selection & mix_selection]
      self._branches['BToKEE_mass_low'] = self._branches['BToKEE_mass'][jpsi_selection & low_selection]

      
      self._branches['BToKEE_fit_mass_all'] = self._branches['BToKEE_fit_mass'][jpsi_selection & overlap_veto_selection]
      self._branches['BToKEE_fit_mass_pf'] = self._branches['BToKEE_fit_mass'][jpsi_selection & pf_selection]
      self._branches['BToKEE_fit_mass_mix'] = self._branches['BToKEE_fit_mass'][jpsi_selection & mix_selection]
      self._branches['BToKEE_fit_mass_low'] = self._branches['BToKEE_fit_mass'][jpsi_selection & low_selection]

      self._branches['BToKEE_l1_pt_pf'] = self._branches['BToKEE_l1_pt'][l1_pf_selection]
      self._branches['BToKEE_l1_pt_low'] = self._branches['BToKEE_l1_pt'][l1_low_selection]
      self._branches['BToKEE_l2_pt_pf'] = self._branches['BToKEE_l2_pt'][l2_pf_selection]
      self._branches['BToKEE_l2_pt_low'] = self._branches['BToKEE_l2_pt'][l2_low_selection]

      self._branches['BToKEE_l1_mvaId_low'] = self._branches['BToKEE_l1_mvaId'][l1_low_selection]
      self._branches['BToKEE_l2_mvaId_low'] = self._branches['BToKEE_l2_mvaId'][l2_low_selection]
      

      # fill output
      self.fill_output()
      #nTree += 1

    self.finish()
    print('[BToKLLAnalyzer::run] INFO: Finished')
    self.print_timestamp()


