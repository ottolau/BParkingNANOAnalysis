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
                                 'BToKEE_fit_mass',
                                 'Electron_pt',
                                 'ProbeTracks_pt',
                                 ]

    self.outputbranches_BToKEE = ['m_Kee',
                                  'elePtLead',
                                  'elePtSublead',
                                  'kaonPt',
                                  ]

  def loop(self, maxevents):

    self.outputbranches = self.outputbranches_BToKEE # we can append BToKMM later
    self.initialization(self.inputbranches_BToKEE, self.outputbranches)
    current_file_name = self.tree.GetCurrentFile().GetName()
    print("Loading file: {}".format(current_file_name))

    for ievent,event in enumerate(self.tree):
      if ievent > maxevents and maxevents != -1: break
      if ievent % 100 == 0: print('Processing entry {}'.format(ievent))
      self.analyze_BToKEE(event)
    
    self.file_out.cd()
    self.file_out.Write()
    self.file_out.Close()

  def analyze_BToKEE(self, event):
    for i in range(event.nBToKEE):
      l1Idx = event.BToKEE_l1Idx[i]
      l2Idx = event.BToKEE_l2Idx[i]
      kIdx = event.BToKEE_kIdx[i]
  
      # Selection


      # Fill the output tree

      self.outputbranch.m_Kee = event.BToKEE_fit_mass[i]
      self.outputbranch.elePtLead = event.Electron_pt[l1Idx]
      self.outputbranch.elePtSublead = event.Electron_pt[l2Idx]
      self.outputbranch.kaonPt = event.ProbeTracks_pt[kIdx]

      self.outputtree.Fill()






