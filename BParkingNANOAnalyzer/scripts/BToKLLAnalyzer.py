#! /usr/bin/env python

#import ROOT
from math import ceil
import awkward
import pandas as pd
import numpy as np
from BParkingNANOAnalysis.BParkingNANOAnalyzer.BaseAnalyzer import BParkingNANOAnalyzer

class BToKLLAnalyzer(BParkingNANOAnalyzer):
  def __init__(self, inputfiles, outputfile, hist=False):
    inputbranches_BToKEE = ['nBToKEE',
                            'BToKEE_mll_raw',
                            'BToKEE_mass',
                            'BToKEE_l1Idx',
                            'BToKEE_l2Idx',
                            'BToKEE_kIdx',
                            'BToKEE_l_xy',
                            'BToKEE_l_xy_unc',
                            'BToKEE_pt',
                            'BToKEE_svprob',
                            'BToKEE_cos2D',
                            'Electron_pt',
                            'Electron_dz',
                            'Electron_unBiased',
                            'Electron_convVeto',
                            'Electron_isLowPt',
                            'Electron_isPF',
                            'ProbeTracks_pt',
                            'ProbeTracks_dz',
                            'ProbeTracks_isLostTrk',
                            'ProbeTracks_isPacked',
                            #'HLT_Mu9_IP6_*',
                            ]

    outputbranches_BToKEE = {'BToKEE_mll_raw': {'nbins': 50, 'xmin': 0.0, 'xmax': 5.0},
                             'BToKEE_mass_all': {'nbins': 30, 'xmin': 4.7, 'xmax': 6.0},
                             'BToKEE_mass_pf': {'nbins': 30, 'xmin': 4.7, 'xmax': 6.0},
                             'BToKEE_mass_low': {'nbins': 30, 'xmin': 4.7, 'xmax': 6.0},
                             'BToKEE_l1_pt_pf': {'nbins': 50, 'xmin': 0.0, 'xmax': 15.0},
                             'BToKEE_l1_pt_low': {'nbins': 50, 'xmin': 0.0, 'xmax': 15.0},
                             'BToKEE_l2_pt_pf': {'nbins': 50, 'xmin': 0.0, 'xmax': 5.0},
                             'BToKEE_l2_pt_low': {'nbins': 50, 'xmin': 0.0, 'xmax': 5.0},
                             'BToKEE_k_pt': {'nbins': 50, 'xmin': 0.0, 'xmax': 5.0},
                           }

    super(BToKLLAnalyzer, self).__init__(inputfiles, outputfile, inputbranches_BToKEE, outputbranches_BToKEE, hist)

  def run(self):
    print('[BParkingNANOAnalyzer::run] INFO: Running the analyzer...')
    self.print_timestamp()
    #self.load_files()

    while self._ifile < self._num_files:
      self.load_branches()
      self.init_output()

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

      # flatten the jagged arrays to a normal numpy array, turn the whole dictionary to pandas dataframe
      self._branches = pd.DataFrame.from_dict({branch: array.flatten() for branch, array in self._branches.items()})

      # add additional branches
      self._branches['BToKEE_l_xy_sig'] = self._branches['BToKEE_l_xy'] / self._branches['BToKEE_l_xy_unc']

      # general selection
      sv_selection = (self._branches['BToKEE_pt'] > 10) & (self._branches['BToKEE_l_xy_sig'] > 6 ) & (self._branches['BToKEE_svprob'] > 0.1) & (self._branches['BToKEE_cos2D'] > 0.999)
      l1_selection = (self._branches['BToKEE_l1_convVeto'] == 1) & (self._branches['BToKEE_l1_pt'] > 1) & (self._branches['BToKEE_l1_unBiased'] > 2)
      l2_selection = (self._branches['BToKEE_l2_convVeto'] == 1) & (self._branches['BToKEE_l2_pt'] > 0.5) & (self._branches['BToKEE_l2_unBiased'] > 2)
      k_selection = (self._branches['BToKEE_k_pt'] > 1.5)
      selection = sv_selection & l1_selection & l2_selection & k_selection

      self._branches = self._branches[selection]

      # additional cuts, allows various lengths

      self._branches['BToKEE_mass_all'] = self._branches['BToKEE_mass'][(self._branches['BToKEE_mll_raw'] > 2.9) & (self._branches['BToKEE_mll_raw'] < 3.3)]
      self._branches['BToKEE_mass_pf'] = self._branches['BToKEE_mass'][(self._branches['BToKEE_mll_raw'] > 2.9) & (self._branches['BToKEE_mll_raw'] < 3.3) & (self._branches['BToKEE_l1_isPF'] == 1) & (self._branches['BToKEE_l2_isPF'] == 1)]
      self._branches['BToKEE_mass_low'] = self._branches['BToKEE_mass'][(self._branches['BToKEE_mll_raw'] > 2.9) & (self._branches['BToKEE_mll_raw'] < 3.3) & ((self._branches['BToKEE_l1_isPF'] == 0) | (self._branches['BToKEE_l2_isPF'] == 0))]

      self._branches['BToKEE_l1_pt_pf'] = self._branches['BToKEE_l1_pt'][(self._branches['BToKEE_l1_isPF'] == 1)]
      self._branches['BToKEE_l1_pt_low'] = self._branches['BToKEE_l1_pt'][(self._branches['BToKEE_l1_isLowPt'] == 1)]
      self._branches['BToKEE_l2_pt_pf'] = self._branches['BToKEE_l2_pt'][(self._branches['BToKEE_l2_isPF'] == 1)]
      self._branches['BToKEE_l2_pt_low'] = self._branches['BToKEE_l2_pt'][(self._branches['BToKEE_l2_isLowPt'] == 1)]


      # fill output
      self.fill_output()
      self._ifile += 1

    self.finish()
    print('[BParkingNANOAnalyzer::run] INFO: Finished')
    self.print_timestamp()


