#! /usr/bin/env python

import ROOT
from array import array
from collections import OrderedDict

class BParkingNANOAnalyzer(object):
  def __init__(self, tchain, outputfile):
    self.tree = tchain
    print('Total number of events: {}'.format(self.tree.GetEntries()))
    self.file_out = ROOT.TFile(outputfile, 'recreate')

  def set_branchstatus(self, branches):
    for branch in branches:
      self.tree.SetBranchStatus(branch, 1)

  def initialize_outputdict(self):
    self.output_list = OrderedDict(zip(self.outputbranches, [-99.]*len(self.outputbranches)))

  def fill_hist(self):
    for hist_name, var in self.output_list.items():
      self.hist_list[hist_name].Fill(var)
    self.initialize_outputdict()

  def fill_tree(self):
    self.outputtree.Fill(array('f',self.output_list.values()))
    self.initialize_outputdict()

  def initialization(self, inputbranches, outputbranches, hist=False):
    self.tree.SetBranchStatus("*", 0)
    self.set_branchstatus(inputbranches)
    self.outputbranches = outputbranches
    self.initialize_outputdict()

    if hist:
      self.hist_list = {}
      for hist_name, hist_bins in outputbranches.items():
        self.hist_list[hist_name] = ROOT.TH1D(hist_name, "", hist_bins[0], hist_bins[1], hist_bins[2])

    else:
      self.outputtree = ROOT.TNtuple('tree', 'tree', ':'.join(outputbranches))



