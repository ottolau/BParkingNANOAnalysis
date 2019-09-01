#! /usr/bin/env python

import ROOT
from math import ceil
from BParkingNANOAnalysis.BParkingNANOAnalyzer.BaseAnalyzer import BParkingNANOAnalyzer

class BToKLLAnalyzer(BParkingNANOAnalyzer):
  def __init__(self, tchain, outputfile):
    super(BToKLLAnalyzer, self).__init__(tchain, outputfile)


  def start(self, hist):
    # selected input branches to be turned on
    inputbranches_BToKEE = ['nBToKEE',
                            'BToKEE_kIdx',
                            'BToKEE_l1Idx',
                            'BToKEE_l2Idx',
                            'BToKEE_mass',
                            'BToKEE_mll_raw',
                            'BToKEE_svprob',
                            'BToKEE_l_xy',
                            'BToKEE_l_xy_unc',
                            'BToKEE_fit_pt',
                            'Electron_pt',
                            'ProbeTracks_pt',
                            ]

    # 'Name of histogram/tree': [nbins, xmin, xmax]
    outputbranches_BToKEE = {'m_ee': [50, 2.6, 3.6],
                             'm_Kee': [50, 4.0, 6.0],
                             'elePt_lead': [50, 0.0, 20.0],
                             'elePt_sublead': [50, 0.0, 20.0],
                             'kaonPt': [50, 0.0, 10.0],
                             }

    outputbranches = outputbranches_BToKEE # we can append BToKMM later
    self.initialization(inputbranches_BToKEE, outputbranches, hist)


  def loop(self, max_nevents=-1, first_event=0, hist=False):
    self.start(hist)
    current_file_name = self._tree.GetCurrentFile().GetName()
    print("Loading file: {}".format(current_file_name))

    if max_nevents > 0:
      limit_nevents = min(max_nevents, self._tot_nevents)
    else:
      limit_nevents = self._tot_nevents

    n_checkpoints = 5
    print_every = int(ceil(1. * limit_nevents / n_checkpoints))
    print "[PedestalAnalysis::run] INFO : Running loop over tree from event {} to {}".format(first_event, limit_nevents - 1)
    self.start_timer()

    for ievent,event in enumerate(self._tree):
      if ievent < first_event: continue
      if ievent > max_nevents and max_nevents != -1: break
      self.print_progress(ievent, first_event, limit_nevents, print_every)
      #if ievent % 100 == 0: print('Processing entry {}'.format(ievent))
      self.analyze(event, hist)
    
    self.write_outputfile()


  def analyze(self, event, hist):
    for i in range(event.nBToKEE):
      l1Idx = event.BToKEE_l1Idx[i]
      l2Idx = event.BToKEE_l2Idx[i]
      kIdx = event.BToKEE_kIdx[i]
  
      # Selection


      # Fill the output tree
      self._output_list['m_ee'] = event.BToKEE_mll_raw[i]
      if 2.9 < event.BToKEE_mll_raw[i] < 3.3:
        self._output_list['m_Kee'] = event.BToKEE_mass[i]
      self._output_list['elePt_lead'] = event.Electron_pt[l1Idx]
      self._output_list['elePt_sublead'] = event.Electron_pt[l2Idx]
      self._output_list['kaonPt'] = event.ProbeTracks_pt[kIdx]
    
      self.fill_output(hist)





