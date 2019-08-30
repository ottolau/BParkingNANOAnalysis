#! /usr/bin/env python

import ROOT

class BParkingNANOAnalyzer(object):
  def __init__(self, tchain, outputfile):
    self.tree = tchain
    print('Total number of events: {}'.format(self.tree.GetEntries()))
    self.file_out = ROOT.TFile(outputfile, 'recreate')

  def set_branchstatus(self, branches):
    for branch in branches:
      self.tree.SetBranchStatus(branch, 1)

  def build_outputbranches(self, nameOfTree, branches):
    output = "struct {} {{".format(nameOfTree)
    output += "".join(["Float_t {};".format(nameOfBranch) for nameOfBranch in branches])
    output += "};"
    ROOT.gROOT.ProcessLine(output)

  def initialization(self, inputbranches, outputbranches, hist=False):
    self.tree.SetBranchStatus("*", 0)
    self.set_branchstatus(inputbranches)
    if hist:
      pass
    else:
      self.build_outputbranches("outputbranch_t", outputbranches)
      self.outputbranch = ROOT.outputbranch_t()
      self.outputtree = ROOT.TTree('tree', 'tree')
      for var in outputbranches:
        self.outputtree.Branch(var, ROOT.AddressOf(self.outputbranch, var), var+'/F')



