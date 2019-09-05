#! /usr/bin/env python

import ROOT
from math import ceil
import awkward
from BParkingNANOAnalysis.BParkingNANOAnalyzer.BaseAnalyzer import BParkingNANOAnalyzer

class BToKLLAnalyzer(BParkingNANOAnalyzer):
  def __init__(self, inputfiles, outputfile, hist=False):
    inputbranches_BToKEE = ['BToKEE_mll_raw',
                          'BToKEE_mass',
                          'BToKEE_l1Idx',
                          'BToKEE_l2Idx',
                          'BToKEE_kIdx',
                          'Electron_pt',
                          'ProbeTracks_pt',
                          ]

    outputbranches_BToKEE = {'BToKEE_mll_raw': {'nbins': 50, 'xmin': 2.6, 'xmax': 3.6},
                           'BToKEE_mass': {'nbins': 50, 'xmin': 4.0, 'xmax': 6.0},
                           'BToKEE_l1_pt': {'nbins': 50, 'xmin': 0.0, 'xmax': 20.0},
                           'BToKEE_l2_pt': {'nbins': 50, 'xmin': 0.0, 'xmax': 20.0},
                           'BToKEE_k_pt': {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0},
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

      # selection

      # fill output
      self.fill_output()
      self._ifile += 1

    self.finish()
    print('[BParkingNANOAnalyzer::run] INFO: Finished')
    self.print_timestamp()


