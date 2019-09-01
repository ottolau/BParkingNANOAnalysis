#! /usr/bin/env python

import ROOT
from array import array
from collections import OrderedDict
from abc import ABCMeta, abstractmethod
import time
import datetime

class BParkingNANOAnalyzer(object):
  def __init__(self, tchain, outputfile):
    __metaclass__ = ABCMeta
    self._tree = tchain
    self._tot_nevents = self._tree.GetEntries()
    print('Total number of events: {}'.format(self._tot_nevents))
    self._file_out = ROOT.TFile(outputfile, 'recreate')


  def set_branchstatus(self, branches):
    # switch on the selected input branches
    for branch in branches:
      self._tree.SetBranchStatus(branch, 1)


  def initialize_outputdict(self):
    # initialize the dictionary for storing the output variables
    self._output_list = OrderedDict(zip(self._outputbranches, [-99.]*len(self._outputbranches)))


  def initialization(self, inputbranches, outputbranches, hist=False):
    # trun off all the input branches
    self._tree.SetBranchStatus("*", 0)
    # trun on the selected output branches
    self.set_branchstatus(inputbranches)

    self._outputbranches = outputbranches
    self.initialize_outputdict()

    # can choose output to be histograms or a tree
    if hist:
      self._hist_list = {}
      for hist_name, hist_bins in outputbranches.items():
        # define the output histograms (assuming all of them to be TH1D)
        self._hist_list[hist_name] = ROOT.TH1D(hist_name, "", hist_bins[0], hist_bins[1], hist_bins[2])

    else:
      # define the output tree
      self._outputtree = ROOT.TNtuple('tree', 'tree', ':'.join(outputbranches))


  def fill_output(self, hist):
    if hist:
      for hist_name, var in self._output_list.items():
        self._hist_list[hist_name].Fill(var)
    else:
      self._outputtree.Fill(array('f',self._output_list.values()))
    self.initialize_outputdict()



  def write_outputfile(self):
    self._file_out.cd()
    self._file_out.Write()
    self._file_out.Close()


  def start_timer(self):
    self._ts_start = time.time()
    print "[BParkingNANOAnalysis::start_timer] INFO : Start time: {}".format(datetime.datetime.fromtimestamp(self._ts_start).strftime('%Y-%m-%d %H:%M:%S'))


  def print_progress(self, this_event, first_event, last_event, print_every):
    if this_event % print_every == 0:
      print "[BParkingNANOAnalysis::print_progress] INFO : Processing event {} / {}".format(this_event + 1, last_event)
      if this_event != first_event:
        if self._ts_start > 0 :
          elapsed_time = time.time() - self._ts_start
          estimated_completion = self._ts_start + (1. * elapsed_time * (last_event - first_event) / (this_event - first_event))
          m, s = divmod(elapsed_time, 60)
          h, m = divmod(m, 60)
          print "[BParkingNANOAnalysis::print_progress] INFO : \tElapsed time: {} : {} : {:.3}".format(int(round(h, 0)), int(round(m, 0)), s)
          print "[BParkingNANOAnalysis::print_progress] INFO : \tEstimated finish time: {}".format(datetime.datetime.fromtimestamp(estimated_completion).strftime('%Y-%m-%d %H:%M:%S'))
        else:
          print "[BParkingNANOAnalysis::print_progress] INFO : \tFor time estimates, call self.start_timer() right before starting the event loop"


  # Anything before looping over the tree
  # Histogram definitions go here
  @abstractmethod
  def start(self, hist):
    pass

  # Main event loop
  @abstractmethod
  def loop(self, max_nevents=-1, first_event=0, hist=False):
    pass

  # Analyze within one event
  # Selection and filling histograms go here
  @abstractmethod
  def analyze(self, event, hist):
    pass



