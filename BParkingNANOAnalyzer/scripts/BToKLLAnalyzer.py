#! /usr/bin/env python

import ROOT
from BParkingNANOAnalysis.BParkingNANOAnalyzer.BaseAnalyzer import BParkingNANOAnalyzer

class BToKLLAnalyzer(BParkingNANOAnalyzer):
  def __init__(self, tchain, outputfile):
    super(BToKLLAnalyzer, self).__init__(tchain, outputfile)
    self.inputbranches_BToKEE = ['nBToKEE',
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

    self.outputbranches_BToKEE = {'m_ee': [50, 2.6, 3.6],
                                  'm_Kee': [50, 4.0, 6.0],
                                  'elePt_lead': [50, 0.0, 20.0],
                                  'elePt_sublead': [50, 0.0, 20.0],
                                  'kaonPt': [50, 0.0, 10.0],
                                  }

  def loop(self, maxevents, hist=False):
    self.outputbranches = self.outputbranches_BToKEE # we can append BToKMM later

    self.initialization(self.inputbranches_BToKEE, self.outputbranches, hist)
    current_file_name = self.tree.GetCurrentFile().GetName()
    print("Loading file: {}".format(current_file_name))

    for ievent,event in enumerate(self.tree):
      if ievent > maxevents and maxevents != -1: break
      if ievent % 100 == 0: print('Processing entry {}'.format(ievent))
      self.analyze_BToKEE(event, hist)
    
    self.file_out.cd()
    self.file_out.Write()
    self.file_out.Close()

  def analyze_BToKEE(self, event, hist):
    for i in range(event.nBToKEE):
      l1Idx = event.BToKEE_l1Idx[i]
      l2Idx = event.BToKEE_l2Idx[i]
      kIdx = event.BToKEE_kIdx[i]
  
      # Selection


      # Fill the output tree
      self.output_list['m_ee'] = event.BToKEE_mll_raw[i]
      if 2.9 < event.BToKEE_mll_raw[i] < 3.3:
        self.output_list['m_Kee'] = event.BToKEE_mass[i]
      self.output_list['elePt_lead'] = event.Electron_pt[l1Idx]
      self.output_list['elePt_sublead'] = event.Electron_pt[l2Idx]
      self.output_list['kaonPt'] = event.ProbeTracks_pt[kIdx]

      if hist:
        self.fill_hist()
      else:
        self.fill_tree()






