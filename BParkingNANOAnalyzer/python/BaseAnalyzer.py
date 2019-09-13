#! /usr/bin/env python

import ROOT
from collections import OrderedDict
from abc import ABCMeta, abstractmethod
import time
import datetime
import uproot
import awkward
import sys
import os
import numpy as np
#import rootpy.ROOT as ROOT
from rootpy.io import root_open
from rootpy.plotting import Hist
from rootpy.tree import Tree
from root_numpy import fill_hist, array2root, array2tree

ROOT.gErrorIgnoreLevel=ROOT.kError

class BParkingNANOAnalyzer(object):
  def __init__(self, inputfiles, outputfile, inputbranches, outputbranches, hist=False):
    __metaclass__ = ABCMeta
    self._file_out_name = outputfile.replace('.root','')
    self._file_in_name = inputfiles
    self._num_files = len(self._file_in_name)
    self._inputbranches = inputbranches
    self._outputbranches = outputbranches
    self._hist = hist
    self._ifile = 0


  def init_output(self):
    print('[BParkingNANOAnalyzer::init_output] INFO: FILE: {}/{}. Initializing output {}...'.format(self._ifile+1, self._num_files, 'histograms' if self._hist else 'tree'))
    self._file_out = root_open(self._file_out_name+'.root', 'recreate')
    # can choose output to be histograms or a tree
    if self._hist:
      self._hist_list = {}
      for hist_name, hist_bins in self._outputbranches.items():
        # define the output histograms (assuming all of them to be TH1F)
        self._hist_list[hist_name] = Hist(hist_bins['nbins'], hist_bins['xmin'], hist_bins['xmax'], name=hist_name, title='', type='F')

    else:
      # define the output tree
      self._outputtree = Tree("tree")
      self._outputtree.create_branches({branch: 'F' for branch in sorted(self._outputbranches.keys())})
      self._outputtree.write()
      self._file_out.close()


  def fill_output(self):
    print('[BParkingNANOAnalyzer::fill_output] INFO: FILE: {}/{}. Filling the output {}...'.format(self._ifile+1, self._num_files, 'histograms' if self._hist else 'tree'))
    if self._hist:
      for hist_name, hist_bins in sorted(self._outputbranches.items()):
        if hist_name in self._branches.keys():
          branch_np = self._branches[hist_name].values
          fill_hist(self._hist_list[hist_name], branch_np[np.isfinite(branch_np)])
    else:
      for branch_name in sorted(self._outputbranches.keys()):
        if branch_name in self._branches.keys():
          branch_np = self._branches[branch_name].values
          new_column = np.array(branch_np, dtype=[(branch_name, 'f4')])
          array2root(new_column, self._file_out_name+'_subset{}.root'.format(self._ifile), 'tree')


  def finish(self):
    print('[BParkingNANOAnalyzer::finish] INFO: Merging the output files...')
    #os.system("hadd -k -f {}.root {}_subset*.root".format(self._file_out_name, self._file_out_name))
    #os.system("rm {}_subset*.root".format(self._file_out_name))
    if self._hist:
      for hist_name, hist in sorted(self._hist_list.items()):
        hist.write()
    else:
      pass
    self._file_out.close()


  def print_timestamp(self):
    ts_start = time.time()
    print("[BParkingNANOAnalysis::print_timestamp] INFO : Time: {}".format(datetime.datetime.fromtimestamp(ts_start).strftime('%Y-%m-%d %H:%M:%S')))


  @abstractmethod
  def run(self):
    pass








